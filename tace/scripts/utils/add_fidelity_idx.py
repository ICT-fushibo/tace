################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import argparse
import torch
from ase.io import read, write


def main():
    parser = argparse.ArgumentParser(description="Add fidelity idx")
    parser.add_argument(
        "-i", "--input",
        # nargs="+",
        type=str,
        required=True,
        help="Paths to ase.io readable files"
    )
    parser.add_argument(
        "-f", "--fidelity_idx",
        type=int,
        required=True,
        help="fidelity idx, start from 0",
    )
    args = parser.parse_args()
    atomsList = read(args.input, ':')
    output_file = f"{args.input}-f{args.fidelity_idx}.xyz"
    print(f"{args.input} -> {output_file} (fidelity {args.fidelity_idx})")
    for atoms in atomsList:
        atoms.info['fidelity_idx'] = args.fidelity_idx

    write(output_file, atomsList)
    
if __name__ == "__main__":
    main()
