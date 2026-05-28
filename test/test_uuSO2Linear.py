################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import argparse
import math

import torch

from tace.models._e3nn.fused import uuSO2Linear


def component_labels(lmax: int, mmax: int) -> list[str]:
    labels = [f"m0/l{l}" for l in range(lmax + 1)]
    for m in range(1, mmax + 1):
        for part in ("real", "imag"):
            for l in range(m, lmax + 1):
                labels.append(f"m{m}/{part}/l{l}")
    return labels


def check_variance(
    lmax: int,
    mmax: int,
    channels: int,
    batch: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
    tolerance: float,
    weight_type: str,
    path_norm: bool,
) -> None:
    torch.manual_seed(seed)
    module = uuSO2Linear(
        mmax=mmax,
        lmax=lmax,
        num_channel=channels,
        weight_type=weight_type,
        path_norm=path_norm,
    ).to(
        device=device
    )
    num_components = (lmax + 1) + sum(
        2 * (lmax + 1 - m) for m in range(1, mmax + 1)
    )

    x = torch.randn(batch, num_components, channels, device=device, dtype=dtype)
    weight = torch.randn(batch, module.weight_numel, device=device, dtype=dtype)
    out = module(x, weight)

    x_var = x.var(unbiased=False).item()
    w_var = weight.var(unbiased=False).item()
    out_var = out.var(unbiased=False).item()
    comp_var = out.var(dim=(0, 2), unbiased=False)
    channel_var = out.var(dim=(0, 1), unbiased=False)

    print(module)
    print(f"input variance  : {x_var:.6f}")
    print(f"weight variance : {w_var:.6f}")
    print(f"output variance : {out_var:.6f}")
    print(
        "component var   : "
        f"mean={comp_var.mean().item():.6f}, "
        f"min={comp_var.min().item():.6f}, "
        f"max={comp_var.max().item():.6f}, "
        f"std={comp_var.std(unbiased=False).item():.6f}"
    )
    print(
        "channel var     : "
        f"mean={channel_var.mean().item():.6f}, "
        f"min={channel_var.min().item():.6f}, "
        f"max={channel_var.max().item():.6f}, "
        f"std={channel_var.std(unbiased=False).item():.6f}"
    )

    labels = component_labels(lmax, mmax)
    print("\nper component variance:")
    for label, var in zip(labels, comp_var.tolist()):
        print(f"  {label:12s} {var:.6f}")

    rel_error = abs(out_var - 1.0)
    if not math.isfinite(out_var) or rel_error > tolerance:
        raise SystemExit(
            f"uuSO2Linear variance check failed: "
            f"|{out_var:.6f} - 1| = {rel_error:.6f} > {tolerance}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check uuSO2Linear variance preservation with unit-variance inputs."
    )
    parser.add_argument("--lmax", type=int, default=3)
    parser.add_argument("--mmax", type=int, default=2)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--batch", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument(
        "--weight-type",
        choices=("w1_w2", "w1_w1", "w1"),
        default="w1_w2",
    )
    parser.add_argument("--no-path-norm", action="store_true")
    args = parser.parse_args()

    check_variance(
        lmax=args.lmax,
        mmax=args.mmax,
        channels=args.channels,
        batch=args.batch,
        seed=args.seed,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        tolerance=args.tolerance,
        weight_type=args.weight_type,
        path_norm=not args.no_path_norm,
    )


if __name__ == "__main__":
    main()
