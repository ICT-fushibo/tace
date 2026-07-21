################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import argparse
from pathlib import Path

import torch


from tace.lightning import load_tace, export_tace
from tace.utils._global import DTYPE


ALLOWED_BACKEND = ["state_dict", "aoti", "lammps"]


def _default_aoti_output_path(model_path: str) -> str:
    return str(Path(model_path).with_suffix(".pt2"))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Model path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output model path",
    )
    parser.add_argument(
        "-l", "--fidelity_idx",
        type=int,
        default=None,
        help="Which fidelity to use",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="Model dtype",
        choices=['float32', 'float64'],
        default=None,
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device for inference"
    )
    parser.add_argument(
        "--backend", 
        type=str, 
        default="state_dict",
        choices=ALLOWED_BACKEND, 
        help="Specify the backend to export"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_tace(args.model, args.device, strict=True, use_ema=True)
    model_dtype = model.get_model_dtype()
    args_dtype = DTYPE[args.dtype] or model_dtype
    if args_dtype != model_dtype:
        print(f"[Warning] Model dtype does not match args.dtype. Forcing dtype from {model_dtype} to {args_dtype}")
    torch.set_default_dtype(args_dtype)
    model.reset_fidelity_idx(args.fidelity_idx)
    model.to(dtype=args_dtype, device=args.device)

    if args.backend == "state_dict":
        export_tace(model, args.output or args.model + "-state.pt")
    elif args.backend == "aoti":
        from tace.models.compile import export_aotinductor

        output_path = args.output or _default_aoti_output_path(args.model)
        output_path = export_aotinductor(model, output_path)
        print(f"[Done] AOTInductor graph model saved to: {output_path}")
    elif args.backend == "lammps":
        from tace.interface.lammps import TACELammpsCalc
        model.lmp = True
        lammps_model = TACELammpsCalc(model)
        torch.save(lammps_model, args.output or args.model + "-lammps_mliap.pt")
    else:
        raise ValueError(f"Unsupported backend '{args.backend}'. One of {ALLOWED_BACKEND} is available.")

if __name__ == "__main__":
    main()
