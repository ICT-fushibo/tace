import gc
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from tace.models.so2.so3 import WignerD


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

METHODS = ("ictd", "flash", "euler")
EDGE_COUNTS = (10, 100, 1_000, 10_000, 100_000)
LMAX_VALUES = (1, 2, 3, 4, 5, 6)

FIXED_LMAX = 4
FIXED_NUM_EDGES = 10_0000

WARMUP_STEPS = 10
BENCHMARK_STEPS = 100
OUT_DIR = Path(__file__).resolve().parent / "wigner_benchmark_results"


def synchronize() -> None:
    if torch.device(DEVICE).type == "cuda":
        torch.cuda.synchronize()


def clear_cache() -> None:
    gc.collect()
    if torch.device(DEVICE).type == "cuda":
        torch.cuda.empty_cache()


def random_rotation(batch: int) -> torch.Tensor:
    q = torch.randn(batch, 4, dtype=DTYPE, device=DEVICE)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        [
            torch.stack(
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                dim=-1,
            ),
            torch.stack(
                [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
                dim=-1,
            ),
            torch.stack(
                [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
                dim=-1,
            ),
        ],
        dim=-2,
    )


def benchmark(method: str, lmax: int, num_edges: int) -> float:
    clear_cache()
    torch.manual_seed(0)
    rotation = random_rotation(num_edges)
    module = WignerD(lmax=lmax, mmax=lmax, wigner_type=method).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            module._rotation_to_wigner_matrix(rotation, 0, lmax)
        synchronize()

        start = time.perf_counter()
        for _ in range(BENCHMARK_STEPS):
            module._rotation_to_wigner_matrix(rotation, 0, lmax)
        synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000.0 / BENCHMARK_STEPS
    del rotation, module
    clear_cache()
    return elapsed_ms


def collect_edge_scaling() -> dict[str, list[float]]:
    results = {method: [] for method in METHODS}
    for num_edges in EDGE_COUNTS:
        for method in METHODS:
            elapsed_ms = benchmark(method, FIXED_LMAX, num_edges)
            results[method].append(elapsed_ms)
            print(
                f"edge scaling | {method:<9} | "
                f"edges={num_edges:>7} | lmax={FIXED_LMAX} | {elapsed_ms:>9.3f} ms"
            )
    return results


def collect_lmax_scaling() -> dict[str, list[float]]:
    results = {method: [] for method in METHODS}
    for lmax in LMAX_VALUES:
        for method in METHODS:
            elapsed_ms = benchmark(method, lmax, FIXED_NUM_EDGES)
            results[method].append(elapsed_ms)
            print(
                f"lmax scaling | {method:<9} | "
                f"edges={FIXED_NUM_EDGES:>7} | lmax={lmax} | {elapsed_ms:>9.3f} ms"
            )
    return results


def plot(
    edge_results: dict[str, list[float]],
    lmax_results: dict[str, list[float]],
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "wigner_rotation_to_d.pdf"

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for method in METHODS:
        axes[0].plot(EDGE_COUNTS, edge_results[method], marker="o", label=method)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("number of rotations / edges")
    axes[0].set_ylabel("R -> Wigner-D time (ms)")
    axes[0].set_title(f"Edge scaling, lmax={FIXED_LMAX}")
    axes[0].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[0].legend()

    for method in METHODS:
        axes[1].plot(LMAX_VALUES, lmax_results[method], marker="o", label=method)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("lmax")
    axes[1].set_ylabel("R -> Wigner-D time (ms)")
    axes[1].set_title(f"lmax scaling, edges={FIXED_NUM_EDGES:,}")
    axes[1].grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[1].legend()

    fig.suptitle(f"Wigner-D generation from rotation matrix ({DEVICE}, {DTYPE})")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    print("Benchmarking R -> Wigner-D")
    print(f"device={DEVICE}, dtype={DTYPE}")
    print(f"methods={METHODS}")
    print()

    edge_results = collect_edge_scaling()
    lmax_results = collect_lmax_scaling()
    out_path = plot(edge_results, lmax_results)
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
