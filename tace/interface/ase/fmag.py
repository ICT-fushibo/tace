################################################################################
# Synergistic Spin–Lattice Optimization
# (Born–Oppenheimer type, PRB 111, 134412 style)
################################################################################

import numpy as np
from ase.io import read
from tace.interface.ase import TACEAseCalc


# ==============================================================================
# Math utilities
# ==============================================================================

def normalize(v):
    """
    Normalize a (N,3) vector field safely
    """
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return v / n


def project_to_tangent(G, S):
    """
    Project magnetic force to tangent plane of spin sphere
    """
    S_hat = normalize(S)
    return G - np.sum(G * S_hat, axis=1, keepdims=True) * S_hat


# ==============================================================================
# Spin relaxer (inner BO loop)
# ==============================================================================

class SpinRelaxer:
    """
    Relax spins at fixed atomic positions
    """

    def __init__(
        self,
        S,
        dt=0.1,
        gamma=0.8,
        spin_norm=1.0,
        max_iter=300,
        tol=5e-3,
    ):
        self.S = S.copy()
        self.vS = np.zeros_like(S)

        self.dt = dt
        self.gamma = gamma
        self.spin_norm = spin_norm

        self.max_iter = max_iter
        self.tol = tol

    def run(self, compute_energy_and_G, verbose=False):
        """
        Spin self-consistency loop
        """
        for it in range(self.max_iter):
            _, G = compute_energy_and_G()

            Gp = project_to_tangent(G, self.S)
            Gmax = np.linalg.norm(Gp, axis=1).max()

            # damped gradient flow on tangent space
            self.vS = self.gamma * self.vS + self.dt * Gp
            self.S += self.dt * self.vS

            # enforce fixed spin length
            self.S = normalize(self.S) * self.spin_norm

            if verbose and it % 20 == 0:
                print(f"    spin {it:4d} |G⊥|_max = {Gmax:.3e}")

            if Gmax < self.tol:
                break

        return self.S, Gmax


# ==============================================================================
# Position updater (outer BO loop)
# ==============================================================================

class PositionUpdater:
    """
    Update atomic positions with inertial gradient descent
    """

    def __init__(
        self,
        R,
        dt=0.05,
        gamma=0.8,
        max_step=0.2,
    ):
        self.R = R.copy()
        self.vR = np.zeros_like(R)

        self.dt = dt
        self.gamma = gamma
        self.max_step = max_step

    def step(self, F):
        self.vR = self.gamma * self.vR + self.dt * F
        dR = self.dt * self.vR

        step_norm = np.linalg.norm(dR)
        if step_norm > self.max_step:
            dR *= self.max_step / step_norm

        self.R += dR


# ==============================================================================
# Main driver
# ==============================================================================

def main():

    # --------------------------------------------------------------------------
    # Load system
    # --------------------------------------------------------------------------

    atoms = read("Fe-Fmag.xyz", index=0)

    atoms.calc = TACEAseCalc(
        model="last.ckpt",
        device="cuda",
        dtype="float32",
        level=0,
    )

    # initial degrees of freedom
    R = atoms.get_positions()
    S = atoms.arrays["force_mag"]   # initial noncollinear spins

    # --------------------------------------------------------------------------
    # Parameters
    # --------------------------------------------------------------------------

    max_ionic_steps = 200
    f_tol = 0.05
    g_tol = 0.00001

    # --------------------------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------------------------

    pos_updater = PositionUpdater(
        R,
        dt=0.05,
        gamma=0.8,
        max_step=0.2,
    )

    # --------------------------------------------------------------------------
    # BO spin–lattice loop
    # --------------------------------------------------------------------------

    for ionic_step in range(max_ionic_steps):

        # ===== spin self-consistency =====

        def compute_E_G():
            atoms.set_positions(R)
            atoms.info["initial_noncollinear_magmoms"] = S
            E = atoms.get_potential_energy()
            G = atoms.calc.results["noncollinear_magnetic_forces"]
            return E, G

        spin_relax = SpinRelaxer(
            S,
            dt=0.1,
            gamma=0.8,
            spin_norm=1.0,
            tol=g_tol,
        )

        S, Gmax = spin_relax.run(compute_E_G, verbose=False)

        # ===== forces with converged spins =====

        atoms.set_positions(R)
        atoms.info["initial_noncollinear_magmoms"] = S

        E = atoms.get_potential_energy()
        F = atoms.get_forces()

        Fmax = np.linalg.norm(F, axis=1).max()

        print(
            f"ionic {ionic_step:4d}  "
            f"E = {E: .8f}  "
            f"|F|max = {Fmax:.3f}  "
            f"|G⊥|_max = {Gmax:.3f}"
        )

        # ===== position update =====

        pos_updater.step(F)
        R = pos_updater.R.copy()

        # ===== convergence =====

        if Fmax < f_tol and Gmax < g_tol:
            print("✅ Spin–lattice converged")
            break


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    main()
