################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import argparse
import logging
import sys

from ..foundations import tace_foundations


def download_one(registry, name: str):
    path = registry[name]
    logging.info(f"Model '{name}' is ready at: {path}")


def download_all(registry):
    logging.info("Downloading all available models...")
    for name in registry.list_models():
        path = registry[name]
        logging.info(f"✔ {name} -> {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained TACE models"
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Name of the model to download (download all if omitted)",
    )

    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List all available models and exit",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    registry = tace_foundations

    if args.list:
        print("Available pretrained models:")
        for name in registry.list_models():
            print(f"  - {name}")
        sys.exit(0)

    if args.model is None:
        download_all(registry)
    else:
        download_one(registry, args.model)


if __name__ == "__main__":
    main()
