"""
src/tps_multiobjective.py — Multi-Objective TPS Optimisation (NSGA-II)
=======================================================================
Pareto front over (mass [kg/m²], safety_factor, cost_index) for TPS design.

NSGA-II (Non-dominated Sorting Genetic Algorithm II, Deb 2002)
implemented in pure numpy/scipy — no external library required.

Objectives (all to be minimised — negate SF for maximisation):
  f1 = mass_kgm2              (minimise)
  f2 = -safety_factor          (minimise, i.e. maximise SF)
  f3 = cost_index              (minimise — proxy for material cost)

Design variables:
  x[0] = TPS thickness [m]     bounds: [0.005, 0.150]
  x[1] = material_index        continuous in [0, N_mat-1], rounded to int
  x[2] = nose_radius [m]       bounds: [0.5, 8.0]

Constraints:
  g1: recession < thickness        (TPS must survive)
  g2: T_peak < T_limit * 0.95      (10% structural margin)
  g3: SF > 1.0                     (minimum structural safety)

Output:
  Pareto front DataFrame + interactive matplotlib plot
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Callable


# ── Material cost index (relative, normalised to nylon=1.0) ──────────────────
MATERIAL_COSTS = {
    "nylon":   1.0,
    "kevlar":  3.5,
    "nomex":   2.8,
    "vectran": 5.2,
    "zylon":   7.0,
    "pica":    12.0,
    "avcoat":  18.0,
    "srp":     9.5,
}
MATERIAL_NAMES = list(MATERIAL_COSTS.keys())


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_design(thickness: float, material: str, nose_radius: float,
                     q_peak_MW: float, t_entry_s: float) -> dict:
    """Evaluate one TPS design. Returns objectives + constraint violations."""
    from src.ablation_model import AblationSolver, ABLATIVE_DB
    from src.thermal_model   import ThermalProtectionSystem, MATERIAL_DB

    mat_lower = material.lower()

    # Ablative materials
    if mat_lower in ABLATIVE_DB:
        try:
            solver = AblationSolver(mat_lower, thickness, n_nodes=8)
            t  = np.linspace(0, t_entry_s, 60)
            t_pk = t_entry_s * 0.30
            q0 = q_peak_MW * 1e6 * np.where(
                t <= t_pk, t/t_pk, (t_entry_s-t)/(t_entry_s-t_pk))
            q0 = np.clip(q0, 0, None)
            res = solver.solve(q0, t, verbose=False)
            mat_db = ABLATIVE_DB[mat_lower]
            rho    = mat_db.density_kgm3
            T_lim  = mat_db.T_ablate_K
            T_pk   = res["peak_T_K"]
            recession_m = res["total_recession_mm"] / 1000.0
            survived    = res["total_recession_mm"] < thickness * 1000.0
            sf = (thickness * 1000.0) / max(res["total_recession_mm"], 0.01)
        except Exception:
            rho=220; T_lim=3000; T_pk=9999; recession_m=thickness; survived=False; sf=0.1
    else:
        # Passive TPS
        try:
            tps = ThermalProtectionSystem(mat_lower, thickness, n_nodes=8)
            t   = np.linspace(0, t_entry_s, 60)
            q0  = q_peak_MW * 1e6 * np.ones(60) * 0.5
            tps.solve_1d_conduction(q0, t, T_initial_K=300)
            exc, T_pk = tps.check_material_limit()
            mat_p  = MATERIAL_DB[mat_lower]
            rho    = mat_p.density_kgm3
            T_lim  = mat_p.max_temperature_K
            recession_m = 0.0
            sf     = tps.safety_margin()
            survived = not exc
        except Exception:
            rho=1100; T_lim=700; T_pk=9999; recession_m=0; survived=False; sf=0.1

    mass = rho * thickness   # [kg/m²]
    cost = MATERIAL_COSTS.get(mat_lower, 5.0) * thickness * 1000  # cost_index

    # Constraint violations (positive = violated)
    g1 = recession_m - thickness * 0.999   # recession < thickness
    g2 = T_pk - T_lim * 0.95              # T_peak < 0.95 * T_limit
    g3 = 1.0 - sf                          # SF > 1.0

    feasible = (g1 <= 0) and (g2 <= 0) and (g3 <= 0)

    return {
        "thickness_m": thickness,
        "material":    material,
        "nose_radius": nose_radius,
        "mass_kgm2":   mass,
        "sf":          sf,
        "cost_index":  cost,
        "T_peak_K":    T_pk,
        "recession_mm":recession_m * 1000,
        "survived":    survived,
        "feasible":    feasible,
        # Objectives (minimise all)
        "f1": mass,          # minimise mass
        "f2": -sf,           # minimise -SF (maximise SF)
        "f3": cost,          # minimise cost
        # Constraint violations
        "g1": g1, "g2": g2, "g3": g3,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NSGA-II IMPLEMENTATION  (pure numpy)
# ══════════════════════════════════════════════════════════════════════════════

def _dominates(f_a: np.ndarray, f_b: np.ndarray) -> bool:
    """True if solution a dominates b: a ≤ b in all objectives, < in at least one."""
    return bool(np.all(f_a <= f_b) and np.any(f_a < f_b))


def _fast_non_dominated_sort(F: np.ndarray) -> list[list[int]]:
    """
    Fast non-dominated sorting (Deb 2002).
    Returns list of fronts, each front is list of solution indices.
    """
    n = len(F)
    S = [[] for _ in range(n)]   # S[i] = solutions dominated by i
    n_dom = np.zeros(n, int)      # domination count
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i == j: continue
            if _dominates(F[i], F[j]):
                S[i].append(j)
            elif _dominates(F[j], F[i]):
                n_dom[i] += 1
        if n_dom[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def _crowding_distance(F: np.ndarray, front: list[int]) -> np.ndarray:
    """Crowding distance for solutions in a front."""
    n  = len(front)
    cd = np.zeros(n)
    F_f = F[front]
    n_obj = F_f.shape[1]

    for m in range(n_obj):
        order = np.argsort(F_f[:, m])
        f_min, f_max = F_f[order[0], m], F_f[order[-1], m]
        spread = max(f_max - f_min, 1e-12)
        cd[order[0]] = cd[order[-1]] = np.inf
        for i in range(1, n-1):
            cd[order[i]] += (F_f[order[i+1], m] - F_f[order[i-1], m]) / spread

    return cd


def _constraint_violation(g_vals: np.ndarray) -> float:
    """Sum of constraint violations (0 = feasible)."""
    return float(np.sum(np.maximum(g_vals, 0)))


class NSGAII:
    """
    NSGA-II multi-objective optimiser.

    Minimises objectives f = [f1, f2, f3, ...] subject to g ≤ 0.
    """

    def __init__(self, n_pop: int = 60, n_gen: int = 50,
                 crossover_rate: float = 0.9, mutation_rate: float = 0.15,
                 seed: int = 42):
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.cr    = crossover_rate
        self.mr    = mutation_rate
        self.rng   = np.random.default_rng(seed)

    def run(self, evaluate: Callable, bounds: np.ndarray,
            verbose: bool = True) -> dict:
        """
        Run NSGA-II optimisation.

        Parameters
        ----------
        evaluate : function(x: ndarray) → (F: ndarray, G: ndarray)
                   F = objective vector (minimise), G = constraint violations (≤ 0)
        bounds   : (n_var, 2) array of [lo, hi]
        verbose  : print progress

        Returns
        -------
        dict: pareto_X, pareto_F, all_X, all_F, history
        """
        n_var = len(bounds)
        lo, hi = bounds[:, 0], bounds[:, 1]

        # ── Initial population ────────────────────────────────────────────────
        X = lo + (hi - lo) * self.rng.uniform(size=(self.n_pop, n_var))
        F_list, G_list = [], []
        for x in X:
            f, g = evaluate(x)
            F_list.append(f); G_list.append(g)
        F = np.array(F_list, dtype=float)
        G = np.array(G_list, dtype=float)
        CV = np.array([_constraint_violation(g) for g in G])

        history = []

        for gen in range(self.n_gen):
            # ── Tournament selection ──────────────────────────────────────────
            fronts = _fast_non_dominated_sort(F)
            cd     = np.zeros(self.n_pop)
            rank   = np.zeros(self.n_pop, int)
            for r, front in enumerate(fronts):
                cd_f = _crowding_distance(F, front)
                for k, idx in enumerate(front):
                    rank[idx] = r; cd[idx] = cd_f[k]

            def tournament(a, b):
                if CV[a] < CV[b]: return a
                if CV[b] < CV[a]: return b
                if rank[a] < rank[b]: return a
                if rank[b] < rank[a]: return b
                return a if cd[a] > cd[b] else b

            # ── Crossover + mutation (SBX + polynomial mutation) ──────────────
            X_off = np.empty_like(X)
            for i in range(0, self.n_pop, 2):
                p1 = tournament(*self.rng.integers(0, self.n_pop, 2))
                p2 = tournament(*self.rng.integers(0, self.n_pop, 2))
                if self.rng.random() < self.cr:
                    eta_c = 15.0
                    u = self.rng.uniform(size=n_var)
                    beta = np.where(u <= 0.5,
                                    (2*u)**(1/(eta_c+1)),
                                    (1/(2*(1-u)))**(1/(eta_c+1)))
                    c1 = 0.5*((1+beta)*X[p1] + (1-beta)*X[p2])
                    c2 = 0.5*((1-beta)*X[p1] + (1+beta)*X[p2])
                else:
                    c1, c2 = X[p1].copy(), X[p2].copy()

                # Polynomial mutation
                for c in [c1, c2]:
                    for j in range(n_var):
                        if self.rng.random() < self.mr:
                            eta_m = 20.0
                            u     = self.rng.random()
                            delta = ((2*u)**(1/(eta_m+1))-1) if u<0.5 else (1-(2*(1-u))**(1/(eta_m+1)))
                            c[j]  += delta * (hi[j]-lo[j])

                X_off[i]   = np.clip(c1, lo, hi)
                X_off[min(i+1,self.n_pop-1)] = np.clip(c2, lo, hi)

            # ── Evaluate offspring ────────────────────────────────────────────
            F_off, G_off, CV_off = [], [], []
            for x in X_off:
                f, g = evaluate(x)
                F_off.append(f); G_off.append(g)
                CV_off.append(_constraint_violation(g))

            # ── Combine parent + offspring ────────────────────────────────────
            X_all  = np.vstack([X,  X_off])
            F_all  = np.vstack([F,  F_off])
            CV_all = np.concatenate([CV, CV_off])

            # ── Select next generation ────────────────────────────────────────
            fronts_all = _fast_non_dominated_sort(F_all)
            selected   = []
            for front in fronts_all:
                if len(selected) + len(front) <= self.n_pop:
                    selected.extend(front)
                else:
                    needed = self.n_pop - len(selected)
                    cd_f   = _crowding_distance(F_all, front)
                    sorted_f = sorted(range(len(front)), key=lambda k: -cd_f[k])
                    selected.extend([front[k] for k in sorted_f[:needed]])
                    break

            X  = X_all[selected]
            F  = F_all[selected]
            CV = CV_all[selected]
            G  = [G_off[i-self.n_pop] if i >= self.n_pop else G_list[i] for i in selected]

            n_feasible = int((CV == 0).sum())
            f1_best    = float(F[CV == 0, 0].min()) if n_feasible > 0 else float("inf")
            history.append({"gen": gen+1, "n_feasible": n_feasible, "f1_best": f1_best})

            if verbose and (gen+1) % max(1, self.n_gen//5) == 0:
                print(f"  [NSGA-II] gen {gen+1:3d}/{self.n_gen}  "
                      f"feasible={n_feasible}/{self.n_pop}  "
                      f"mass_best={f1_best:.3f}kg/m²")

        # ── Extract Pareto front ──────────────────────────────────────────────
        feasible_mask = CV == 0
        if feasible_mask.any():
            F_feas = F[feasible_mask]; X_feas = X[feasible_mask]
            pf_idx = [i for i in _fast_non_dominated_sort(F_feas)[0]]
            pareto_F = F_feas[pf_idx]; pareto_X = X_feas[pf_idx]
        else:
            pareto_F = F; pareto_X = X

        return {
            "pareto_X":  pareto_X,
            "pareto_F":  pareto_F,
            "all_X":     X,
            "all_F":     F,
            "all_CV":    CV,
            "history":   history,
            "n_feasible":int(feasible_mask.sum()),
        }


# ══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_pareto_optimisation(
    q_peak_MW:  float = 15.0,
    t_entry_s:  float = 200.0,
    n_pop:      int   = 50,
    n_gen:      int   = 40,
    materials:  list  = None,
    verbose:    bool  = True,
) -> dict:
    """
    Run multi-objective TPS optimisation and return Pareto front.

    Design space
    ------------
    x[0] : thickness [m]          [0.005, 0.15]
    x[1] : material index         [0, len(materials)-1]
    x[2] : (reserved for nose_r)  [0.5, 8.0]
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    mats = materials or ["pica", "avcoat", "srp", "kevlar", "zylon"]
    n_m  = len(mats)

    bounds = np.array([
        [0.005, 0.12],    # thickness
        [0.0,   n_m-1],   # material index (continuous, rounded)
        [0.5,   6.0],     # nose radius
    ])

    def evaluate(x):
        th  = float(np.clip(x[0], bounds[0,0], bounds[0,1]))
        mat = mats[int(round(float(np.clip(x[1], 0, n_m-1))))]
        R_n = float(np.clip(x[2], bounds[2,0], bounds[2,1]))
        res = _evaluate_design(th, mat, R_n, q_peak_MW, t_entry_s)
        F   = np.array([res["f1"], res["f2"], res["f3"]])
        G   = np.array([res["g1"], res["g2"], res["g3"]])
        return F, G

    if verbose:
        print(f"\n[NSGA-II] TPS Pareto Optimisation  "
              f"q_peak={q_peak_MW}MW/m²  materials={mats}")

    nsga = NSGAII(n_pop=n_pop, n_gen=n_gen, seed=0)
    result = nsga.run(evaluate, bounds, verbose=verbose)

    # Decode designs
    records = []
    for x, f in zip(result["pareto_X"], result["pareto_F"]):
        th  = float(x[0])
        mat = mats[int(round(float(np.clip(x[1], 0, n_m-1))))]
        R_n = float(x[2])
        res = _evaluate_design(th, mat, R_n, q_peak_MW, t_entry_s)
        records.append(res)

    df_pareto = pd.DataFrame(records)
    result["pareto_df"] = df_pareto

    if verbose:
        print(f"\n  Pareto front: {len(df_pareto)} solutions")
        if len(df_pareto):
            print(f"  Mass range:   [{df_pareto['mass_kgm2'].min():.2f}, {df_pareto['mass_kgm2'].max():.2f}] kg/m²")
            print(f"  SF range:     [{df_pareto['sf'].min():.2f}, {df_pareto['sf'].max():.2f}]")
            print(f"  Cost range:   [{df_pareto['cost_index'].min():.1f}, {df_pareto['cost_index'].max():.1f}]")

    # Plot
    _plot_pareto(df_pareto, result, q_peak_MW)
    return result


def _plot_pareto(df: "pd.DataFrame", result: dict, q_peak: float):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    matplotlib.rcParams.update({
        "figure.facecolor":"#080c14","axes.facecolor":"#0d1526",
        "axes.edgecolor":"#2a3d6e","text.color":"#c8d8f0",
        "axes.labelcolor":"#c8d8f0","xtick.color":"#c8d8f0",
        "ytick.color":"#c8d8f0","grid.color":"#1a2744",
        "font.family":"monospace","font.size":9,
    })
    TX="#c8d8f0"; C1="#00d4ff"; C2="#ff6b35"; C3="#a8ff3e"; C4="#ffd700"

    fig = plt.figure(figsize=(20, 11), facecolor="#080c14")
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            hspace=0.48, wspace=0.36,
                            top=0.90, bottom=0.07, left=0.05, right=0.97)

    def gax(r, c):
        a = fig.add_subplot(gs[r, c])
        a.set_facecolor("#0d1526"); a.grid(True, alpha=0.28)
        a.tick_params(colors=TX); a.spines[:].set_color("#2a3d6e")
        return a

    if len(df) == 0:
        fig.text(0.5, 0.5, "No feasible solutions found — increase n_gen or n_pop",
                 ha="center", va="center", color=TX, fontsize=14)
    else:
        # Mass vs SF (primary Pareto)
        a = gax(0, 0)
        mat_colors = {m: plt.cm.plasma(i/max(len(df["material"].unique())-1,1))
                      for i, m in enumerate(df["material"].unique())}
        for mat in df["material"].unique():
            sub = df[df["material"]==mat]
            a.scatter(sub["mass_kgm2"], sub["sf"], s=60,
                      color=mat_colors[mat], label=mat, zorder=4, alpha=0.85)
        # Pareto front line
        pf = df.sort_values("mass_kgm2")
        a.plot(pf["mass_kgm2"], pf["sf"], color="#ffd700", lw=1.2, ls="--", alpha=0.7)
        a.axhline(1.5, color=C2, lw=1, ls=":", label="SF=1.5 min", alpha=0.7)
        a.set_xlabel("Mass [kg/m²]"); a.set_ylabel("Safety Factor")
        a.set_title("Pareto: Mass vs SF", fontweight="bold"); a.legend(fontsize=7)

        # Mass vs Cost
        a = gax(0, 1)
        sc1 = a.scatter(df["mass_kgm2"], df["cost_index"],
                         c=df["sf"], cmap="RdYlGn", s=60, alpha=0.85, vmin=0.5, vmax=4)
        fig.colorbar(sc1, ax=a, label="SF", pad=0.02).ax.tick_params(labelsize=7)
        a.set_xlabel("Mass [kg/m²]"); a.set_ylabel("Cost index")
        a.set_title("Pareto: Mass vs Cost (coloured by SF)", fontweight="bold")

        # SF vs T_peak
        a = gax(0, 2)
        sc2 = a.scatter(df["sf"], df["T_peak_K"],
                         c=df["mass_kgm2"], cmap="plasma", s=60, alpha=0.85)
        fig.colorbar(sc2, ax=a, label="mass [kg/m²]", pad=0.02).ax.tick_params(labelsize=7)
        a.axvline(1.5, color=C2, lw=1, ls=":", alpha=0.7)
        a.set_xlabel("Safety Factor"); a.set_ylabel("T_peak [K]")
        a.set_title("SF vs Peak Temperature", fontweight="bold")

        # Thickness vs material scatter
        a = gax(0, 3)
        from matplotlib.patches import Patch
        for mat in df["material"].unique():
            sub = df[df["material"]==mat]
            a.barh([mat], [sub["mass_kgm2"].mean()], color=mat_colors[mat], alpha=0.75)
        a.set_xlabel("Mean mass [kg/m²]"); a.set_title("Mean mass by material", fontweight="bold")

        # Convergence history
        a = gax(1, 0)
        hist = result["history"]
        gens = [h["gen"] for h in hist]
        feas = [h["n_feasible"] for h in hist]
        best = [h["f1_best"] for h in hist]
        a2   = a.twinx()
        a.plot(gens, feas, color=C1, lw=1.8, label="Feasible")
        a2.plot(gens, [b if b<1e8 else np.nan for b in best], color=C3, lw=1.5, ls="--", label="Best mass")
        a.set_xlabel("Generation"); a.set_ylabel("Feasible count", color=C1)
        a2.set_ylabel("Best mass [kg/m²]", color=C3)
        a.set_title("NSGA-II Convergence", fontweight="bold")
        a.tick_params(axis="y", colors=C1); a2.tick_params(axis="y", colors=C3)
        for sp in a2.spines.values(): sp.set_color("#2a3d6e")
        a2.set_facecolor("#0d1526")

        # 3-D Pareto surface
        ax3 = fig.add_subplot(gs[1, 1:3], projection="3d")
        ax3.set_facecolor("#0d1526")
        sc3 = ax3.scatter(df["mass_kgm2"], df["sf"], df["cost_index"],
                           c=df["sf"], cmap="plasma", s=50, alpha=0.85)
        ax3.set_xlabel("Mass\n[kg/m²]", fontsize=8)
        ax3.set_ylabel("SF", fontsize=8)
        ax3.set_zlabel("Cost", fontsize=8)
        ax3.set_title("3-D Pareto Surface", fontweight="bold")

        # Material distribution pie
        a = gax(1, 3)
        mat_counts = df["material"].value_counts()
        colors_pie = [mat_colors.get(m, "#888") for m in mat_counts.index]
        wedges, texts, autotexts = a.pie(mat_counts.values, labels=mat_counts.index,
                                           colors=colors_pie, autopct="%1.0f%%",
                                           startangle=90)
        for t in texts+autotexts: t.set_color(TX); t.set_fontsize(8)
        a.set_title("Material Distribution", fontweight="bold")

    fig.text(0.5, 0.955,
             f"Multi-Objective TPS Pareto Optimisation (NSGA-II)  |  "
             f"q_peak={q_peak:.1f}MW/m²  |  "
             f"{len(df)} Pareto solutions  |  "
             f"{result['n_feasible']} feasible",
             ha="center", fontsize=11, fontweight="bold", color=TX)

    sp = Path("outputs/tps_pareto.png")
    sp.parent.mkdir(exist_ok=True)
    fig.savefig(sp, facecolor=fig.get_facecolor(), dpi=150, bbox_inches="tight")
    print(f"  ✓ Pareto plot saved: {sp}")
    plt.close(fig)
