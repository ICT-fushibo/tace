################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import argparse
import torch

from tace.lightning import load_tace
from tace.lightning.select_model import select_model
from tace.utils._global import DTYPE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy the shared parameters from the source model to the target model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        nargs=2,
        required=True,
        metavar=("SRC_MODEL", "DST_MODEL"),
        help="Two model paths: SRC_MODEL DST_MODEL (SRC will overwrite DST shared parameters)",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        help="Model dtype",
        choices=["float32", "float64"],
        default=None,
    )

    return parser.parse_args()


def merge_state_dict(src_model, dst_model):

    src_sd = src_model.state_dict()
    dst_sd = dst_model.state_dict()

    merged_sd = dst_sd.copy()

    copied = 0
    skipped = 0

    for k, v in src_sd.items():

        if k in dst_sd:

            if dst_sd[k].shape == v.shape:
                merged_sd[k] = v
                copied += 1
            else:
                print(f"[Skip] shape mismatch: {k} {v.shape} -> {dst_sd[k].shape}")
                skipped += 1
                # k: str
                # if k.endswith('mask'):
                #     continue

                # if all(s <= d for s, d in zip(v.shape, dst_sd[k].shape)):
                #     new_tensor = dst_sd[k].clone()
                #     slices = tuple(slice(0, s) for s in v.shape)
                #     new_tensor[slices] = v
                #     merged_sd[k] = new_tensor
                #     print(f"[Partial] {k}: {v.shape} -> {dst_sd[k].shape}")
                #     copied += 1
                # else:
                #     print(f"[Skip] shape mismatch: {k} {v.shape} -> {dst_sd[k].shape}")
                #     skipped += 1

        else:
            print(f"{k} not in dst model")

    print(f"[Info] copied params : {copied}")
    print(f"[Info] skipped params: {skipped}")

    return merged_sd


def main():

    dst_path: str

    args = parse_args()

    src_path, dst_path = args.model

    src_model = load_tace(src_path, "cpu", strict=True, use_ema=True)
    # if dst_path.endswith("*.yaml"):
    #     dst_model = select_model
    # else:
    dst_model = load_tace(dst_path, "cpu", strict=True, use_ema=True)

    model_dtype = dst_model.readout_fn.cutoff.dtype
    args_dtype = DTYPE[args.dtype] or model_dtype

    if args_dtype != model_dtype:
        print(
            f"[Warning] Model dtype does not match args.dtype. "
            f"Forcing dtype from {model_dtype} to {args_dtype}"
        )

    torch.set_default_dtype(args_dtype)

    src_model.to(dtype=args_dtype, device="cpu")
    dst_model.to(dtype=args_dtype, device="cpu")

    merged_state = merge_state_dict(src_model, dst_model)

    dst_model.load_state_dict(merged_state, strict=True)

    save_path = dst_path + "-merged.pt"

    torch.save(
        {
            "state_dict": dst_model.state_dict(),
            "cfg": dst_model.readout_fn.model_config,
            "target_property": dst_model.readout_fn.target_property,
            "embedding_property": dst_model.readout_fn.embedding_property,
            "statistics": dst_model.readout_fn.statistics,
        },
        save_path,
    )

    print(f"[Done] merged model saved to: {save_path}")


if __name__ == "__main__":
    main()