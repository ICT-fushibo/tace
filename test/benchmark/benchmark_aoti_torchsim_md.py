################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
"""Benchmark eager PyTorch and AOTI with a fixed 192-atom TorchSim MD system.

The ASE and TorchSim integrations share the same graph AOTI package. Compile it
once from the repository root on the second GPU:

    TACE_USE_OEQ=1 CUDA_VISIBLE_DEVICES=1 tace-export-eval \
        -m ~/.cache/tace/TACE-OAM-L.pt \
        -o ~/.cache/tace/TACE-OAM-L.pt2 \
        --backend aoti \
        --device cuda \
        --dtype float32 \
        --sample example/data/liquid-64.xyz \
        --sample-index 0 \
        --batch-size 1

Install the optional TorchSim dependency and run:

    pip install -e '.[torchsim]'
    python test/benchmark/benchmark_aoti_torchsim_md.py
"""

import os
import statistics
import time
from pathlib import Path
from typing import Sequence


# Modify benchmark settings here.
CUDA_VISIBLE_DEVICES = "1"
USE_OEQ = True
EAGER_MODEL = Path.home() / ".cache/tace/TACE-OAM-L.pt"
AOTI_MODEL = Path.home() / ".cache/tace/TACE-OAM-L.pt2"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STRUCTURE = REPOSITORY_ROOT / "example/data/liquid-64.xyz"
DEVICE = "cuda"
DTYPE = "float32"
FIDELITY_INDEX = 0
TEMPERATURE_K = 300.0
TIMESTEP_PS = 0.001
WARMUP_STEPS = 5
MD_STEPS = 50
REPEATS = 3
SEED = 42
ATOL = 1.0e-4
RTOL = 1.0e-4


os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
if USE_OEQ:
    os.environ["TACE_USE_OEQ"] = "1"

import numpy as np
import torch
import torch_sim as ts
from ase.io import read
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
)

from tace.interface.torchsim import TACETorchSimCalc


def synchronize() -> None:
    if torch.device(DEVICE).type == "cuda":
        torch.cuda.synchronize()


def prepare_atoms():
    atoms = read(STRUCTURE, index=0)
    rng = np.random.RandomState(SEED)
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=rng)
    Stationary(atoms)
    return atoms


def build_calculator(model_path: Path) -> TACETorchSimCalc:
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    return TACETorchSimCalc(
        model=str(model_path),
        device=DEVICE,
        dtype=DTYPE,
        fidelity_idx=FIDELITY_INDEX,
        compute_forces=True,
        compute_stress=True,
    )


def run_md(atoms, calculator: TACETorchSimCalc, steps: int):
    torch.manual_seed(SEED)
    return ts.integrate(
        system=[atoms.copy()],
        model=calculator,
        integrator=ts.Integrator.nvt_nose_hoover,
        n_steps=steps,
        temperature=TEMPERATURE_K,
        timestep=TIMESTEP_PS,
        autobatcher=False,
        pbar=False,
    )


def check_outputs(atoms, eager_calculator, aoti_calculator) -> tuple[float, float]:
    eager_state = run_md(atoms, eager_calculator, 1)
    aoti_state = run_md(atoms, aoti_calculator, 1)
    eager_energy = eager_state.energy.detach()
    aoti_energy = aoti_state.energy.detach()
    eager_forces = eager_state.forces.detach()
    aoti_forces = aoti_state.forces.detach()

    energy_error = float(torch.max(torch.abs(eager_energy - aoti_energy)).cpu())
    force_error = float(torch.max(torch.abs(eager_forces - aoti_forces)).cpu())
    if not torch.allclose(eager_energy, aoti_energy, atol=ATOL, rtol=RTOL):
        raise RuntimeError(
            f"Eager and AOTI energies differ; maximum error is {energy_error:.3e}"
        )
    if not torch.allclose(eager_forces, aoti_forces, atol=ATOL, rtol=RTOL):
        raise RuntimeError(
            f"Eager and AOTI forces differ; maximum error is {force_error:.3e}"
        )
    return energy_error, force_error


def benchmark(atoms, calculator: TACETorchSimCalc) -> Sequence[float]:
    if WARMUP_STEPS:
        run_md(atoms, calculator, WARMUP_STEPS)
    synchronize()

    elapsed = []
    for _ in range(REPEATS):
        synchronize()
        start = time.perf_counter()
        run_md(atoms, calculator, MD_STEPS)
        synchronize()
        elapsed.append(time.perf_counter() - start)
    return elapsed


def print_result(name: str, elapsed: Sequence[float]) -> None:
    median = statistics.median(elapsed)
    ms_per_step = median * 1.0e3 / MD_STEPS
    steps_per_second = MD_STEPS / median
    timestep_fs = TIMESTEP_PS * 1.0e3
    ns_per_day = MD_STEPS * timestep_fs * 86400.0 / (median * 1.0e6)
    print(
        f"{name:<10} {median:>12.4f} {ms_per_step:>14.3f} "
        f"{steps_per_second:>14.2f} {ns_per_day:>12.4f}"
    )


def main() -> None:
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    atoms = prepare_atoms()
    print("Loading eager model...")
    eager_calculator = build_calculator(EAGER_MODEL)
    print("Loading AOTI package...")
    aoti_calculator = build_calculator(AOTI_MODEL)

    energy_error, force_error = check_outputs(
        atoms,
        eager_calculator,
        aoti_calculator,
    )
    print(f"atoms          : {len(atoms)}")
    print(f"energy |error| : {energy_error:.3e} eV")
    print(f"force max error: {force_error:.3e} eV/Angstrom")
    print(f"warmup/steps   : {WARMUP_STEPS}/{MD_STEPS} x {REPEATS}")
    print()

    eager_elapsed = benchmark(atoms, eager_calculator)
    aoti_elapsed = benchmark(atoms, aoti_calculator)

    print(f"{'backend':<10} {'median (s)':>12} {'ms/step':>14} {'steps/s':>14} {'ns/day':>12}")
    print("-" * 66)
    print_result("eager", eager_elapsed)
    print_result("AOTI", aoti_elapsed)
    speedup = statistics.median(eager_elapsed) / statistics.median(aoti_elapsed)
    print()
    print(f"AOTI speedup: {speedup:.3f}x")


if __name__ == "__main__":
    main()
