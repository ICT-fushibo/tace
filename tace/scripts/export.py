################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import argparse

import torch


from ..lightning import load_tace
from ..utils._global import DTYPE


ALLOWED_BACKEND = ["state_dict", "whole_model", "lammps"]


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
        "-l", "--level",
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
        choices=["cpu", "cuda"], 
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
    model_dtype = model.readout_fn.cutoff.dtype
    args_dtype = DTYPE[args.dtype] or model_dtype
    if args_dtype != model_dtype:
        print(f"[Warning] Model dtype does not match args.dtype. Forcing dtype from {model_dtype} to {args_dtype}")
    torch.set_default_dtype(args_dtype)
    if args.level is not None:
        model.level = args.level
    model.to(dtype=args_dtype, device=args.device)

    if args.backend == "state_dict":
        torch.save(
            {
                "state_dict": model.state_dict(),
                "cfg": model.readout_fn.model_config,
                "target_property": model.readout_fn.target_property,
                "embedding_property": model.readout_fn.embedding_property,
                "statistics": model.readout_fn.statistics,
            }, 
            args.model + "-state.pt"
        )
    elif args.backend == "whole_model":
        torch.save(model, args.model + "-whole.pt") 
    elif args.backend == "lammps":
        from ..interface.lammps import TACELammpsCalc
        model.lmp = True
        lammps_model = TACELammpsCalc(model)
        torch.save(lammps_model, args.model + "-lammps_mliap.pt")
    else:
        raise ValueError(f"Unsupported backend '{args.backend}'. One of {ALLOWED_BACKEND} is available.")

if __name__ == "__main__":
    main()