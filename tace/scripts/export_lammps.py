################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import argparse
from pathlib import Path

import torch

from tace.lightning import load_tace
from tace.utils._global import DTYPE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a TACE model for LAMMPS MLIAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Model path")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output path")
    parser.add_argument(
        "-l",
        "--fidelity_idx",
        type=int,
        default=None,
        help="Which fidelity to export",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default=None,
        help="Model dtype",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Load device")
    return parser.parse_args()


def _default_output_path(model_path: str) -> str:
    path = Path(model_path)
    return str(path.with_name(path.name + "-lammps_mliap.pt"))


def main():
    args = parse_args()
    model = load_tace(args.model, args.device, strict=True, use_ema=True)
    model_dtype = model.get_model_dtype()
    args_dtype = DTYPE[args.dtype] or model_dtype
    if args_dtype != model_dtype:
        print(
            "[Warning] Model dtype does not match args.dtype. "
            f"Forcing dtype from {model_dtype} to {args_dtype}"
        )
    torch.set_default_dtype(args_dtype)
    model.reset_fidelity_idx(args.fidelity_idx)
    model.eval().to(dtype=args_dtype, device=args.device)
    for param in model.parameters():
        param.requires_grad = False

    try:
        from tace.interface.lammps import TACELammpsCalc
    except (ImportError, NameError) as exc:
        raise RuntimeError(
            "LAMMPS MLIAP export requires the LAMMPS Python package. "
            "AOTI export for LAMMPS is not supported yet."
        ) from exc

    model.lmp = True
    lammps_model = TACELammpsCalc(model)
    output_path = args.output or _default_output_path(args.model)
    torch.save(lammps_model, output_path)
    print(f"[Done] LAMMPS MLIAP model saved to: {output_path}")


if __name__ == "__main__":
    main()
