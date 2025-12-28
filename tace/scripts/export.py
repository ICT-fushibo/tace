################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import argparse

import torch


from ..lightning import load_tace


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
        default="lammps",
        choices=["lammps", "torch"], 
        help="Specify the backend to export"
    )
    return parser.parse_args()

DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    None: None
}

def main():
    args = parse_args()
    model = load_tace(args.model, args.device, strict=True, use_ema=True)
    model_dtype = model.readout_fn.cutoff.dtype
    args_dtype = DTYPE[args.dtype] or model_dtype
    if args_dtype != model_dtype:
        print(f"[Warning] Model dtype does not match args.dtype. Forcing dtype from {model_dtype} to {args_dtype}")
    torch.set_default_dtype(args_dtype)
    model.to(dtype=args_dtype, device=args.device)
    if args.backend == "lammps":
        from ..interface.lammps import TACELammpsCalc
        model.lmp = True
        lammps_model = TACELammpsCalc(model)
        torch.save(lammps_model, args.model + "-lammps_mliap.pt")
    elif args.backend == "torch":
        torch.save(model, args.model + "-torch.pt") 
    else:
        raise ValueError(f"Unsupported backend '{args.backend}'. Currently only 'lammps' and 'torch' is available.")

if __name__ == "__main__":
    main()