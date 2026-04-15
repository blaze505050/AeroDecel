"""
src/canopy_gnn.py — Graph Neural Network for Canopy Stress/Deformation
=======================================================================
Represents the canopy fabric as a graph where:
  Nodes = fabric panel control points  (position, material properties)
  Edges = seam connections, gore boundaries, riser attachment

A GNN (Message Passing Neural Network) predicts per-node stress and
displacement under aerodynamic loading. Unlike fixed-mesh FEM, this
handles arbitrary canopy topologies.

Graph structure
---------------
  Node features: [x, y, z, area, porosity, material_k, pressure_local]
  Edge features: [dx, dy, dz, seam_strength, gore_type]
  Output:        [displacement_x, displacement_y, displacement_z, von_Mises_stress]

Message Passing (MPNN)
----------------------
  h_v^(k+1) = UPDATE(h_v^(k), AGGREGATE({m_uv | (u,v) ∈ E}))
  m_uv = MESSAGE(h_u^(k), h_v^(k), e_uv)

  MESSAGE:    MLP(concat[h_u, h_v, e_uv]) → latent message
  AGGREGATE:  mean pooling
  UPDATE:     GRU(h_v, aggregated) or MLP+residual

Pure numpy fallback uses a simplified spectral GNN approach.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


@dataclass
class CanopyGraph:
    """Canopy represented as a graph for GNN input."""
    # Node arrays (N × features)
    node_pos:      np.ndarray   # (N, 3)  x,y,z positions
    node_area:     np.ndarray   # (N,)    panel areas [m²]
    node_porosity: np.ndarray   # (N,)    fabric porosity [0,1]
    node_pressure: np.ndarray   # (N,)    local aerodynamic pressure [Pa]
    # Edge arrays (M × features)
    edge_src:      np.ndarray   # (M,)  source node indices
    edge_dst:      np.ndarray   # (M,)  dest node indices
    edge_length:   np.ndarray   # (M,)  seam length [m]
    edge_strength: np.ndarray   # (M,)  seam rated strength [N/m]

    @property
    def N(self): return len(self.node_pos)
    @property
    def M(self): return len(self.edge_src)

    @property
    def node_features(self) -> np.ndarray:
        return np.column_stack([
            self.node_pos,               # x, y, z
            self.node_area[:, None],
            self.node_porosity[:, None],
            self.node_pressure[:, None],
        ])  # (N, 7)

    @property
    def edge_features(self) -> np.ndarray:
        dx = self.node_pos[self.edge_dst] - self.node_pos[self.edge_src]
        return np.column_stack([dx, self.edge_length[:, None],
                                 self.edge_strength[:, None]])   # (M, 5)


def generate_canopy_graph(canopy_radius_m: float = 5.0,
                           n_gores: int = 12,
                           n_radial: int = 6,
                           q_dyn: float = 100.0) -> CanopyGraph:
    """
    Generate a realistic canopy graph from geometric parameters.
    Gores arranged radially around the apex.
    """
    R  = canopy_radius_m
    nodes_pos  = []
    nodes_area = []
    nodes_por  = []
    nodes_pres = []

    # Apex node
    nodes_pos.append([0, 0, 0])
    nodes_area.append(0.1)
    nodes_por.append(0.05)
    nodes_pres.append(q_dyn * 0.5)

    # Radial rings of gore nodes
    for r_ring in range(1, n_radial + 1):
        r     = R * r_ring / n_radial
        area  = np.pi * (r**2 - (R*(r_ring-1)/n_radial)**2) / n_gores
        pres  = q_dyn * (1.0 - 0.3*(r/R)**2)   # Newtonian pressure profile
        for k in range(n_gores):
            theta = 2*np.pi*k/n_gores
            nodes_pos.append([r*np.cos(theta), r*np.sin(theta),
                               -0.3*R*(1-(r/R)**2)])   # bowl shape
            nodes_area.append(area)
            nodes_por.append(0.012 + 0.002*np.random.randn())
            nodes_pres.append(pres)

    # Skirt nodes (perimeter)
    for k in range(n_gores):
        theta = 2*np.pi*k/n_gores
        nodes_pos.append([R*np.cos(theta), R*np.sin(theta), -0.1*R])
        nodes_area.append(0.2)
        nodes_por.append(0.02)
        nodes_pres.append(q_dyn * 0.2)

    nodes_pos  = np.array(nodes_pos)
    N          = len(nodes_pos)

    # Edges: radial seams + circumferential
    src_list, dst_list = [], []

    # Apex to first ring
    for k in range(n_gores):
        src_list.append(0); dst_list.append(1 + k)

    # Ring to ring
    for r_ring in range(1, n_radial):
        base_in  = 1 + (r_ring-1)*n_gores
        base_out = 1 + r_ring*n_gores
        for k in range(n_gores):
            src_list.append(base_in + k); dst_list.append(base_out + k)
            # Circumferential seam
            kn = (k+1) % n_gores
            src_list.append(base_in + k); dst_list.append(base_in + kn)

    # Last ring to skirt
    base_last  = 1 + (n_radial-1)*n_gores
    base_skirt = 1 + n_radial*n_gores
    for k in range(n_gores):
        src_list.append(base_last + k); dst_list.append(base_skirt + k)

    src = np.array(src_list, dtype=int)
    dst = np.array(dst_list, dtype=int)

    # Edge properties
    d_   = np.linalg.norm(nodes_pos[dst] - nodes_pos[src], axis=1)
    stre = np.full(len(src), 5000.0)   # 5000 N/m seam strength

    # Make undirected (add reverse edges)
    src_u = np.concatenate([src, dst])
    dst_u = np.concatenate([dst, src])
    d_u   = np.concatenate([d_, d_])
    stre_u= np.concatenate([stre, stre])

    return CanopyGraph(
        node_pos      = nodes_pos,
        node_area     = np.array(nodes_area),
        node_porosity = np.array(nodes_por),
        node_pressure = np.array(nodes_pres),
        edge_src      = src_u,
        edge_dst      = dst_u,
        edge_length   = d_u,
        edge_strength = stre_u,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TORCH GNN
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH:
    class _MessageNet(nn.Module):
        def __init__(self, node_dim, edge_dim, hidden):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2*node_dim + edge_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, node_dim),
            )
        def forward(self, h_src, h_dst, e_feat):
            return self.net(torch.cat([h_src, h_dst, e_feat], dim=-1))

    class _UpdateNet(nn.Module):
        def __init__(self, node_dim, hidden):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2*node_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, node_dim),
            )
        def forward(self, h, agg):
            return h + self.net(torch.cat([h, agg], dim=-1))  # residual

    class CanopyGNN(nn.Module):
        """MPNN for canopy stress prediction."""
        def __init__(self, node_feat: int = 7, edge_feat: int = 5,
                     hidden: int = 64, n_layers: int = 4, out_dim: int = 4):
            super().__init__()
            self.lift  = nn.Linear(node_feat, hidden)
            self.msg_nets = nn.ModuleList(
                [_MessageNet(hidden, edge_feat, hidden) for _ in range(n_layers)])
            self.upd_nets = nn.ModuleList(
                [_UpdateNet(hidden, hidden) for _ in range(n_layers)])
            self.out = nn.Sequential(
                nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, out_dim))

        def forward(self, h_node, edge_src, edge_dst, edge_feat):
            h = torch.relu(self.lift(h_node))
            for msg_net, upd_net in zip(self.msg_nets, self.upd_nets):
                msgs = msg_net(h[edge_src], h[edge_dst], edge_feat)
                # Aggregate (mean over incoming edges per node)
                agg  = torch.zeros_like(h)
                count= torch.zeros(h.shape[0], 1)
                agg.index_add_(0, edge_dst, msgs)
                count.index_add_(0, edge_dst, torch.ones(len(edge_dst), 1))
                agg  = agg / (count + 1e-9)
                h    = upd_net(h, agg)
            return self.out(h)


# ══════════════════════════════════════════════════════════════════════════════
# NUMPY FALLBACK: Laplacian smoothing for stress approximation
# ══════════════════════════════════════════════════════════════════════════════

def _numpy_stress_estimate(graph: CanopyGraph) -> dict:
    """
    Laplacian-based stress estimate (no GNN).
    Stress ≈ pressure × area / connectivity
    """
    N = graph.N
    pressure = graph.node_pressure
    area     = graph.node_area

    # Adjacency
    degree = np.zeros(N)
    np.add.at(degree, graph.edge_src, 1)

    # Approximate displacement via iterative Laplacian (5 iterations)
    u = pressure * area / max(pressure.max(), 1e-6) * 0.01  # init

    L = np.zeros((N, N))
    for s, d in zip(graph.edge_src, graph.edge_dst):
        L[d, s] -= 1
    np.fill_diagonal(L, degree)

    for _ in range(10):
        force = pressure * area
        u_new = u.copy()
        for i in range(N):
            if degree[i] > 0:
                neighbours = graph.edge_dst[graph.edge_src == i]
                if len(neighbours):
                    u_new[i] = (force[i] + u[neighbours].sum()) / (degree[i] + 1)
        u = u_new

    # von Mises stress approximation
    strain_energy = 0.5 * np.abs(L @ u) * graph.node_area
    vm_stress = strain_energy / (graph.node_area + 1e-9)

    disp = np.column_stack([u * 0.01, u * 0.01, u])  # x,y,z displacement

    return {
        "displacement_m": disp,
        "von_mises_Pa":   vm_stress,
        "max_stress_Pa":  float(vm_stress.max()),
        "max_disp_m":     float(np.linalg.norm(disp, axis=1).max()),
        "backend": "numpy",
    }


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED API
# ══════════════════════════════════════════════════════════════════════════════

class CanopyGNNPredictor:
    """
    Predicts canopy stress and deformation from aerodynamic loading.
    Uses torch GNN if available, numpy approximation otherwise.
    """
    def __init__(self, hidden: int = 64, n_layers: int = 4):
        self._backend = "torch" if _TORCH else "numpy"
        if _TORCH:
            self.model = CanopyGNN(node_feat=7, edge_feat=5,
                                   hidden=hidden, n_layers=n_layers, out_dim=4)
            self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self._trained = False

    def train(self, graphs: list[CanopyGraph], targets: list[np.ndarray],
              n_epochs: int = 200, verbose: bool = True) -> list[float]:
        """Train on labelled (graph, target) pairs. targets: list of (N,4) arrays."""
        losses = []
        if not _TORCH:
            self._trained = True
            return [0.0]

        import torch
        for ep in range(1, n_epochs+1):
            ep_loss = 0.0
            for g, y in zip(graphs, targets):
                self.opt.zero_grad()
                h  = torch.tensor(g.node_features, dtype=torch.float32)
                es = torch.tensor(g.edge_src, dtype=torch.long)
                ed = torch.tensor(g.edge_dst, dtype=torch.long)
                ef = torch.tensor(g.edge_features, dtype=torch.float32)
                yt = torch.tensor(y, dtype=torch.float32)
                pred = self.model(h, es, ed, ef)
                loss = ((pred - yt)**2).mean()
                loss.backward(); self.opt.step()
                ep_loss += float(loss)
            losses.append(ep_loss / max(len(graphs), 1))
            if verbose and ep % max(1, n_epochs//4) == 0:
                print(f"  [GNN] ep {ep}/{n_epochs}  loss={losses[-1]:.4e}")

        self._trained = True
        return losses

    def predict(self, graph: CanopyGraph) -> dict:
        """Predict displacement and stress for a canopy graph."""
        if not _TORCH or not self._trained:
            return _numpy_stress_estimate(graph)

        import torch
        self.model.eval()
        h  = torch.tensor(graph.node_features, dtype=torch.float32)
        es = torch.tensor(graph.edge_src, dtype=torch.long)
        ed = torch.tensor(graph.edge_dst, dtype=torch.long)
        ef = torch.tensor(graph.edge_features, dtype=torch.float32)
        with torch.no_grad():
            pred = self.model(h, es, ed, ef).numpy()

        return {
            "displacement_m": pred[:, :3],
            "von_mises_Pa":   np.abs(pred[:, 3]),
            "max_stress_Pa":  float(np.abs(pred[:, 3]).max()),
            "max_disp_m":     float(np.linalg.norm(pred[:, :3], axis=1).max()),
            "backend": "torch",
        }

    def safety_factor(self, result: dict, graph: CanopyGraph) -> np.ndarray:
        """Per-node SF = seam_strength / von_mises_stress."""
        avg_strength = np.zeros(graph.N)
        count = np.zeros(graph.N)
        np.add.at(avg_strength, graph.edge_src, graph.edge_strength)
        np.add.at(count, graph.edge_src, 1)
        avg_strength /= np.maximum(count, 1)
        return avg_strength / np.maximum(result["von_mises_Pa"], 1.0)


def run_gnn(n_gores: int = 12, n_radial: int = 6,
            q_dyn: float = 150.0, verbose: bool = True) -> dict:
    """Run canopy GNN for a given dynamic pressure."""
    import matplotlib; matplotlib.use("Agg")

    graph = generate_canopy_graph(canopy_radius_m=5.0, n_gores=n_gores,
                                   n_radial=n_radial, q_dyn=q_dyn)
    predictor = CanopyGNNPredictor(hidden=32, n_layers=3)

    # Generate synthetic training data (physics-based labels)
    graphs_train = [generate_canopy_graph(n_gores=n_gores, n_radial=n_radial,
                                           q_dyn=q*q_dyn) for q in np.linspace(0.5,2,8)]
    targets_train = []
    for g in graphs_train:
        res = _numpy_stress_estimate(g)
        targets_train.append(np.column_stack([res["displacement_m"],
                                               res["von_mises_Pa"][:,None]]))

    predictor.train(graphs_train, targets_train, n_epochs=100, verbose=verbose)
    result = predictor.predict(graph)
    sf = predictor.safety_factor(result, graph)

    if verbose:
        print(f"\n[GNN] N={graph.N} nodes  M={graph.M} edges  q_dyn={q_dyn}Pa")
        print(f"  Max stress:  {result['max_stress_Pa']:.2f} Pa")
        print(f"  Max disp:    {result['max_disp_m']*1000:.3f} mm")
        print(f"  Min SF:      {sf.min():.3f}")
        print(f"  Backend:     {result['backend']}")

    return {"graph": graph, "result": result, "sf": sf, "predictor": predictor}
