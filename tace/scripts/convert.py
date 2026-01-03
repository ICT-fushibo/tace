################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import argparse

import torch

from ..lightning import load_tace
from ..lightning.lora import from_lora_to_merged_model
from ..utils._global import DTYPE

ALLOWED_TYPE = ["merged_lora"]

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
        "-t", "--type", 
        type=str, 
        default="merge_lora",
        choices=ALLOWED_TYPE, 
        help="Specify convert type"
    )
    parser.add_argument(
        "-d", "--debug", 
        type=int, 
        default=0,
        help="print some extra information for debug"
    )
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def main():
    args = parse_args()
    model = load_tace(args.model, 'cpu', strict=True, use_ema=True)
    if bool(args.debug):
        print(model)
    model_dtype = model.readout_fn.cutoff.dtype
    args_dtype = DTYPE[args.dtype] or model_dtype
    if args_dtype != model_dtype:
        print(f"[Warning] Model dtype does not match args.dtype. Forcing dtype from {model_dtype} to {args_dtype}")
    model.to(dtype=args_dtype)
    if args.type == "merge_lora":
        total_before = count_parameters(model)
        model = from_lora_to_merged_model(model)
        total_after = count_parameters(model)
        if bool(args.debug):
            print(model)
        print("The number of parameters: ")
        print(f"  Your LoRA:     {total_before - total_after}")
        print(f"  Before merged: {total_before}")
        print(f"  After merged:  {total_after}")
        model.to(dtype=args_dtype)
        torch.save(model, args.model + "-merged_lora.pt")
    else:
        raise ValueError(f"Unsupported convert type '{args.type}'. One of {ALLOWED_TYPE} is available.")

if __name__ == "__main__":
    main()