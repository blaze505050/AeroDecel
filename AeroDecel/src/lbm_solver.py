"""
src/lbm_solver.py — D2Q9 Lattice Boltzmann CFD Solver
=======================================================
Full D2Q9 LBM with:
  • BGK collision operator (single-relaxation-time)
  • Bounce-back boundary conditions (no-slip walls)
  • Zou-He velocity/pressure BCs for inlet/outlet
  • Immersed boundary mask for canopy cross-sections
  • Drag & lift coefficient extraction
  • Convergence monitoring
"""
from __future__ import annotations
import numpy as np


# ── D2Q9 lattice constants ────────────────────────────────────────────────────
W  = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
CX = np.array([0,   1,   0,  -1,   0,   1,   -1,   -1,    1 ], dtype=int)
CY = np.array([0,   0,   1,   0,  -1,   1,    1,   -1,   -1 ], dtype=int)
OPP= np.array([0,   3,   4,   1,   2,   7,    8,    5,    6 ], dtype=int)


def _feq(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Equilibrium distribution for all 9 velocities, shape (9, Ny, Nx)."""
    usq = ux**2 + uy**2
    feq = np.empty((9, *rho.shape))
    for i in range(9):
        cu = CX[i]*ux + CY[i]*uy
        feq[i] = W[i] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*usq)
    return feq


class LBMSolver:
    """
    2-D Lattice Boltzmann solver on a rectangular domain.

    Parameters
    ----------
    resolution  : (Ny, Nx) grid size
    reynolds    : target Reynolds number (used to set viscosity)
    u_lid       : characteristic velocity (lattice units, keep < 0.1 for stability)
    """

    def __init__(self, resolution: tuple[int, int], reynolds: float,
                 u_lid: float = 0.05):
        self.Ny, self.Nx = resolution
        self.Re     = reynolds
        self.u_lid  = u_lid

        # Kinematic viscosity ν = U·L/Re  (lattice units, L = Ny)
        self.nu     = u_lid * self.Ny / max(reynolds, 1.0)
        self.omega  = 1.0 / (3.0 * self.nu + 0.5)   # relaxation rate

        # Clip omega for numerical stability
        self.omega  = float(np.clip(self.omega, 0.50, 1.98))

        self.f: np.ndarray | None = None
        self.geometry: np.ndarray | None = None     # True = solid cell

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self, rho0: float = 1.0,
                   u0: tuple[float, float] = (0.0, 0.0)) -> None:
        """Uniform initial condition."""
        rho = np.full((self.Ny, self.Nx), rho0)
        ux  = np.full((self.Ny, self.Nx), u0[0])
        uy  = np.full((self.Ny, self.Nx), u0[1])
        self.f = _feq(rho, ux, uy)

    def set_geometry(self, mask: np.ndarray) -> None:
        """
        mask : bool array (Ny, Nx)  True = solid (no-flow) cell.
        """
        self.geometry = np.asarray(mask, dtype=bool)

    def geometry_from_canopy(self, canopy_x: np.ndarray,
                              canopy_y: np.ndarray) -> np.ndarray:
        """
        Rasterise a canopy cross-section outline (in physical coords)
        onto the LBM grid.  Canopy is centred in the domain.

        Returns boolean mask (solid = True).
        """
        from matplotlib.path import Path as MplPath
        # Normalise cross-section to grid
        xmin, xmax = canopy_x.min(), canopy_x.max()
        ymin, ymax = canopy_y.min(), canopy_y.max()

        i_arr = np.arange(self.Nx)
        j_arr = np.arange(self.Ny)
        II, JJ = np.meshgrid(i_arr, j_arr)

        # Map grid to canopy coordinate space
        cx_grid = (II / self.Nx) * (xmax - xmin) + xmin - (xmax+xmin)/2
        cy_grid = (JJ / self.Ny) * (ymax - ymin) + ymin - (ymax+ymin)/2

        pts  = np.column_stack([cx_grid.ravel(), cy_grid.ravel()])
        path = MplPath(np.column_stack([canopy_x, canopy_y]))
        mask = path.contains_points(pts).reshape(self.Ny, self.Nx)
        self.geometry = mask
        return mask

    # ── Core LBM steps ────────────────────────────────────────────────────────

    def _stream(self) -> None:
        for i in range(9):
            self.f[i] = np.roll(self.f[i], CX[i], axis=1)
            self.f[i] = np.roll(self.f[i], CY[i], axis=0)

    def _macroscopic(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho = self.f.sum(axis=0)
        ux  = (CX[:, None, None] * self.f).sum(axis=0) / np.maximum(rho, 1e-12)
        uy  = (CY[:, None, None] * self.f).sum(axis=0) / np.maximum(rho, 1e-12)
        return rho, ux, uy

    def _collide(self, rho, ux, uy) -> None:
        feq = _feq(rho, ux, uy)
        self.f += self.omega * (feq - self.f)

    def _bounce_back(self) -> None:
        """No-slip bounce-back at solid cells."""
        if self.geometry is None:
            return
        solid = self.geometry
        f_tmp = self.f.copy()
        for i in range(9):
            self.f[i][solid] = f_tmp[OPP[i]][solid]

    def _apply_lid_driven(self, u_top: float = None) -> None:
        """Lid-driven cavity: top wall moves at u_top."""
        u = u_top if u_top is not None else self.u_lid
        # Top row — Zou-He moving wall
        rho_top = (self.f[0, -1, :] + self.f[1, -1, :] + self.f[3, -1, :]
                   + 2*(self.f[2, -1, :] + self.f[6, -1, :] + self.f[5, -1, :])) / (1 - 0)
        self.f[4, -1, :] = self.f[2, -1, :] - (2/3) * rho_top * 0
        self.f[7, -1, :] = self.f[5, -1, :] + 0.5*(self.f[1, -1, :] - self.f[3, -1, :]) - 0.5*rho_top*u
        self.f[8, -1, :] = self.f[6, -1, :] - 0.5*(self.f[1, -1, :] - self.f[3, -1, :]) + 0.5*rho_top*u

    def _apply_channel_flow(self, u_in: float = None) -> None:
        """Pressure-driven channel: uniform inlet velocity, outlet zero-gradient."""
        u = u_in if u_in is not None else self.u_lid
        # Inlet (left): Zou-He fixed velocity
        rho_in = (self.f[0, :, 0] + self.f[2, :, 0] + self.f[4, :, 0]
                  + 2*(self.f[3, :, 0] + self.f[7, :, 0] + self.f[6, :, 0])) / (1 - u)
        self.f[1, :, 0] = self.f[3, :, 0] + (2/3)*rho_in*u
        self.f[5, :, 0] = self.f[7, :, 0] + (1/6)*rho_in*u
        self.f[8, :, 0] = self.f[6, :, 0] + (1/6)*rho_in*u
        # Outlet (right): copy from second-to-last column
        self.f[:, :, -1] = self.f[:, :, -2]

    # ── Drag / lift extraction ────────────────────────────────────────────────

    def compute_force_coefficients(self, rho_ref: float = 1.0,
                                   u_ref: float = None) -> dict:
        """
        Momentum exchange method for drag and lift on the immersed body.
        """
        if self.geometry is None or self.f is None:
            return {"Cd": 0.0, "Cl": 0.0}

        u = u_ref or self.u_lid
        solid = self.geometry
        Fx = Fy = 0.0

        for i in range(9):
            if CX[i] == 0 and CY[i] == 0:
                continue
            # Boundary cells adjacent to solid
            shifted = np.roll(np.roll(solid, -CY[i], axis=0), -CX[i], axis=1)
            boundary = shifted & ~solid
            Fx += CX[i] * (self.f[i][boundary].sum() + self.f[OPP[i]][boundary].sum())
            Fy += CY[i] * (self.f[i][boundary].sum() + self.f[OPP[i]][boundary].sum())

        L  = max(solid.sum() ** 0.5, 1.0)
        q  = 0.5 * rho_ref * u**2 * L
        return {"Cd": Fx / max(q, 1e-12), "Cl": Fy / max(q, 1e-12),
                "Fx": Fx, "Fy": Fy}

    # ── Main solver ───────────────────────────────────────────────────────────

    def solve(self, steps: int = 5000,
              flow_type: str = "channel",
              convergence_tol: float = 1e-6,
              verbose: bool = True) -> dict:
        """
        Run the LBM simulation.

        Parameters
        ----------
        steps          : maximum iterations
        flow_type      : "channel" | "lid_driven"
        convergence_tol: stop when max Δu < tol
        verbose        : print progress

        Returns
        -------
        dict with ux, uy, rho, vorticity, Cd, Cl, converged, step
        """
        if self.f is None:
            self.initialize()

        ux_prev = np.zeros((self.Ny, self.Nx))

        for step in range(1, steps + 1):
            self._stream()
            rho, ux, uy = self._macroscopic()
            self._collide(rho, ux, uy)
            self._bounce_back()

            if flow_type == "lid_driven":
                self._apply_lid_driven()
            else:
                self._apply_channel_flow()

            # Convergence check every 200 steps
            if step % 200 == 0:
                delta = np.max(np.abs(ux - ux_prev))
                ux_prev = ux.copy()
                if verbose:
                    print(f"\r  LBM step {step:5d}/{steps}  Δu={delta:.2e}", end="", flush=True)
                if delta < convergence_tol:
                    if verbose:
                        print(f"\n  Converged at step {step}  Δu={delta:.2e}")
                    break

        if verbose and step == steps:
            print(f"\n  Reached max steps ({steps})")

        # Vorticity: ωz = ∂uy/∂x - ∂ux/∂y
        vorticity = np.gradient(uy, axis=1) - np.gradient(ux, axis=0)
        forces    = self.compute_force_coefficients(rho_ref=rho.mean())

        return {
            "ux":        ux,
            "uy":        uy,
            "rho":       rho,
            "vorticity": vorticity,
            "Cd":        forces["Cd"],
            "Cl":        forces["Cl"],
            "converged": step < steps,
            "step":      step,
            "Re":        self.Re,
            "omega":     self.omega,
            "nu":        self.nu,
        }
