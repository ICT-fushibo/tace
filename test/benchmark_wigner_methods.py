"""Benchmark TACE Wigner-D generation methods.

Edit the configuration values below, then run:

    python test/benchmark_wigner_methods.py
"""

from __future__ import annotations

import time

import torch

from tace.models.so2.so3 import WignerD, init_edge_rot_mat


# === Configuration ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
LMAX = 6
MMAX = 6

NUM_EDGES = 100_000
WARMUP_STEPS = 10
BENCHMARK_STEPS = 50

METHODS = ("cartesian", "efficient", "euler")
BENCHMARK_BACKWARD = False
USE_TORCH_COMPILE = False


def synchronize() -> None:
    if torch.device(DEVICE).type == "cuda":
        torch.cuda.synchronize()


def benchmark(
    fn,
    *,
    warmup_steps: int = WARMUP_STEPS,
    benchmark_steps: int = BENCHMARK_STEPS,
) -> tuple[float, object]:
    result = None
    for _ in range(warmup_steps):
        result = fn()
    synchronize()

    start = time.perf_counter()
    for _ in range(benchmark_steps):
        result = fn()
    synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / benchmark_steps
    return elapsed_ms, result


def benchmark_forward(
    module: WignerD,
    edge_vector: torch.Tensor,
    rotation: torch.Tensor,
) -> tuple[float, float]:
    rotation_fn = lambda: module._rotation_to_wigner_matrix(rotation, 0, LMAX)
    get_wigner_fn = lambda: module.get_wigner(edge_vector)

    if USE_TORCH_COMPILE:
        rotation_fn = torch.compile(rotation_fn, fullgraph=True)
        get_wigner_fn = torch.compile(get_wigner_fn, fullgraph=True)

    with torch.no_grad():
        rotation_ms, _ = benchmark(rotation_fn)
        total_ms, _ = benchmark(get_wigner_fn)
    return rotation_ms, total_ms


def benchmark_backward(
    module: WignerD,
    edge_vector: torch.Tensor,
) -> float:
    def step() -> torch.Tensor:
        x = edge_vector.detach().clone().requires_grad_(True)
        wigner, wigner_inv = module.get_wigner(x)
        loss = wigner.square().mean() + wigner_inv.square().mean()
        loss.backward()
        return x.grad

    if USE_TORCH_COMPILE:
        step = torch.compile(step, fullgraph=True)

    backward_ms, _ = benchmark(step)
    return backward_ms


def main() -> None:
    if MMAX > LMAX:
        raise ValueError(f"MMAX={MMAX} must not exceed LMAX={LMAX}")
    if torch.device(DEVICE).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is not available")

    torch.manual_seed(0)
    edge_vector = torch.randn(NUM_EDGES, 3, dtype=DTYPE, device=DEVICE)
    rotation = init_edge_rot_mat(edge_vector)

    print("Wigner-D benchmark")
    print(f"device            : {DEVICE}")
    print(f"dtype             : {DTYPE}")
    print(f"lmax / mmax       : {LMAX} / {MMAX}")
    print(f"num_edges         : {NUM_EDGES:,}")
    print(f"warmup / steps    : {WARMUP_STEPS} / {BENCHMARK_STEPS}")
    print(f"torch.compile      : {USE_TORCH_COMPILE}")
    print()
    print(
        f"{'method':<12}"
        f"{'R -> full D (ms)':>18}"
        f"{'edge -> packed D (ms)':>23}"
        f"{'edges/s':>16}"
        + (f"{'fwd+bwd (ms)':>18}" if BENCHMARK_BACKWARD else "")
    )
    print("-" * (87 if BENCHMARK_BACKWARD else 69))

    for method in METHODS:
        module = WignerD(
            lmax=LMAX,
            mmax=MMAX,
            wigner_type=method,
        ).to(device=DEVICE, dtype=DTYPE)

        rotation_ms, total_ms = benchmark_forward(module, edge_vector, rotation)
        edges_per_second = NUM_EDGES / (total_ms * 1.0e-3)
        line = (
            f"{method:<12}"
            f"{rotation_ms:>18.3f}"
            f"{total_ms:>23.3f}"
            f"{edges_per_second:>16,.0f}"
        )
        if BENCHMARK_BACKWARD:
            backward_ms = benchmark_backward(module, edge_vector)
            line += f"{backward_ms:>18.3f}"
        print(line)


if __name__ == "__main__":
    main()
