################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import argparse
import torch

from tace.lightning import load_tace


def main():
    parser = argparse.ArgumentParser(
        description="Load a TACE model and optionally override target properties "
                    "at inference time."
    )

    parser.add_argument(
        "-m", "--model",
        required=True,
        type=str,
        help="Path to the model checkpoint."
    )

    parser.add_argument(
        "-t", "--target_property",
        nargs="+",
        required=False,
        default=None,
        type=str,
        help=(
            "Override the model target properties at inference time. "
            "This allows requesting additional outputs that were not explicitly "
            "used as training targets. For example, one may predict hessians even "
            "if the model was trained only on forces, or enable predictions of "
            "atomic_stresses, or properties under external fields...... "
            "If not provided, the original target_property stored in the checkpoint "
            "will be used."
        )
    )
    parser.add_argument(
        "-z", "--atomic_numbers",
        nargs="+",
        required=False,
        default=None,
        type=int,
        help=(
            "List of atomic numbers. "
            "When provided, the loaded model will be modified to support only "
            "the specified elements. This is useful for exporting a smaller, "
            "element-specific model for deployment or inference on systems "
            "containing a known subset of elements. "
        )
    )

    args = parser.parse_args()
    model = load_tace(
        args.model,
        device="cpu",
        strict=True,
        use_ema=True,
    )
    if args.target_property is not None:
        print("Override target_property with:")
        model.reset_target_property(args.target_property)
        for p in model.target_property:
            print(f"  - {p}")
    else:
        print("Using target_property stored in the checkpoint:")
        for p in model.target_property:
            print(f"  - {p}")

    if args.atomic_numbers is not None:
        raise NotImplementedError("Specifying atomic numbers is not supported yet.")


if __name__ == "__main__":
    main()
