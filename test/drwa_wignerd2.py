import gc
import contextlib
import importlib.util
import math
import sys
import time
import types
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from tace.models.so2.so3 import WignerD


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

METHODS = ("ictd","flash", "euler", "dpa4")
LMAX_VALUES = range(2, 12)
FIXED_NUM_EDGES = 10000

WARMUP_STEPS = 10
BENCHMARK_STEPS = 100
OUT_DIR = Path(__file__).resolve().parent / "wigner_benchmark_results"
DEEPMD_KIT_DIR = Path.home() / "deepmd-kit"
USE_OPT = True
LBALE = {
    "ictd": "TACE-Direct",
    "flash": "TACE-Recursive",
    "euler": "eSCN-Euler",
    "dpa4": "DPA4-Cayley–Klein",
}

def synchronize() -> None:
    if torch.device(DEVICE).type == "cuda":
        torch.cuda.synchronize()


def clear_cache() -> None:
    gc.collect()
    if torch.device(DEVICE).type == "cuda":
        torch.cuda.empty_cache()


def random_quaternion(batch: int) -> torch.Tensor:
    q = torch.randn(batch, 4, dtype=DTYPE, device=DEVICE)
    return q / q.norm(dim=-1, keepdim=True)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
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


def load_dpa4_wigner_calculator():
    module_name = "deepmd.pt.model.descriptor.sezm_nn.wignerd"
    if module_name in sys.modules:
        return sys.modules[module_name].WignerDCalculator
    dpa4_wignerd_path = (
        DEEPMD_KIT_DIR
        / "deepmd"
        / "pt"
        / "model"
        / "descriptor"
        / "sezm_nn"
        / "wignerd.py"
    )
    if not dpa4_wignerd_path.exists():
        print(f"Skip DPA4: cannot find {dpa4_wignerd_path}")
        return None
    for name in (
        "deepmd",
        "deepmd.pt",
        "deepmd.pt.utils",
        "deepmd.utils",
        "deepmd.pt.model",
        "deepmd.pt.model.descriptor",
        "deepmd.pt.model.descriptor.sezm_nn",
    ):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package

    env = types.SimpleNamespace(
        DEVICE=torch.device(DEVICE),
        GLOBAL_PT_FLOAT_PRECISION=DTYPE,
    )
    sys.modules["deepmd.pt.utils"].env = env

    version = types.ModuleType("deepmd.utils.version")
    version.check_version_compatibility = lambda *args, **kwargs: None
    sys.modules["deepmd.utils.version"] = version

    utils = types.ModuleType("deepmd.pt.model.descriptor.sezm_nn.utils")
    utils.nvtx_range = lambda name: contextlib.nullcontext()
    sys.modules["deepmd.pt.model.descriptor.sezm_nn.utils"] = utils

    spec = importlib.util.spec_from_file_location(module_name, dpa4_wignerd_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.WignerDCalculator


def benchmark(method: str, lmax: int, num_edges: int) -> float:
    clear_cache()
    torch.manual_seed(0)
    quaternion = random_quaternion(num_edges)
    rotation = quaternion_to_rotation_matrix(quaternion)

    if method == "dpa4":
        calculator_cls = load_dpa4_wigner_calculator()
        if calculator_cls is None:
            return math.nan
        sys.modules["deepmd.pt.utils"].env.DEVICE = torch.device(DEVICE)
        sys.modules["deepmd.pt.utils"].env.GLOBAL_PT_FLOAT_PRECISION = DTYPE
        module = calculator_cls(lmax=lmax, dtype=DTYPE).to(device=DEVICE)
        fn = lambda: module(quaternion)[0]
    else:
        module = WignerD(lmax=lmax, mmax=lmax, wigner_type=method, use_opt_einsum_fx=USE_OPT).to(
            device=DEVICE,
            dtype=DTYPE,
        )
        fn = lambda: module._rotation_to_wigner_matrix(rotation, 0, lmax)

    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            fn()
        synchronize()

        start = time.perf_counter()
        for _ in range(BENCHMARK_STEPS):
            fn()
        synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000.0 / BENCHMARK_STEPS
    del quaternion, rotation, module
    clear_cache()
    return elapsed_ms


def collect_lmax_scaling() -> dict[str, list[float]]:
    results = {method: [] for method in METHODS}
    for lmax in LMAX_VALUES:
        for method in METHODS:
            if method == 'euler' and lmax > 11:
                continue
            if method == 'ictd' and lmax > 8:
                continue
            elapsed_ms = benchmark(method, lmax, FIXED_NUM_EDGES)
            results[method].append(elapsed_ms)
            elapsed_str = "skipped" if math.isnan(elapsed_ms) else f"{elapsed_ms:>9.3f} ms"
            print(
                f"lmax scaling | {method:<9} | "
                f"edges={FIXED_NUM_EDGES:>7} | lmax={lmax} | {elapsed_str}"
            )
    return results


def plot(
    lmax_results: dict[str, list[float]],
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "wigner_rotation_to_d.pdf"

    fig, ax = plt.subplots(figsize=(5.6, 4.2))

    for method in METHODS:
        x = []
        y = []
        for lmax, elapsed_ms in zip(LMAX_VALUES, lmax_results[method]):
            if not math.isnan(elapsed_ms):
                x.append(lmax)
                y.append(elapsed_ms)
        if y:
            ax.plot(x, y, marker="o", label=LBALE[method])
    ax.set_xticks(LMAX_VALUES)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell_{max}$")
    ax.set_ylabel("rotation matrix or quaternion -> Wigner-D time (ms)")
    ax.set_title(f"Number of Edges = {FIXED_NUM_EDGES:,}, {DEVICE}, {DTYPE}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()

    # fig.suptitle(f"Wigner-D generation ({DEVICE}, {DTYPE})")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    print("Benchmarking rotation representation -> Wigner-D")
    print(f"device={DEVICE}, dtype={DTYPE}")
    print(f"methods={METHODS}")
    print()

    lmax_results = collect_lmax_scaling()
    out_path = plot(lmax_results)
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
