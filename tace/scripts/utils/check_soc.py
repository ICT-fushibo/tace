################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Diagnose whether a magnetic TACE model exhibits SOC-sensitive behavior.

Every valid magnetic model should be invariant when positions
and magnetic moments are rotated together.  A non-SOC model has the additional
symmetry that all magnetic moments can be rotated while the atomic geometry is
held fixed.  An SOC model may change its energy under this spin-only rotation.
"""

import argparse

import ase.io
import numpy as np
import torch
from ase import Atoms
from e3nn import o3
from torch_geometric.loader import DataLoader

from tace.dataset.graph import from_atoms
from tace.dataset.quantity import KeySpecification
from tace.lightning import load_tace


def rotate_structure(
    atoms: Atoms,
    rotation: np.ndarray,
    magmom_key: str,
    *,
    rotate_geometry: bool,
) -> Atoms:
    """Rotate moments, and optionally the complete atomic geometry."""
    rotated = atoms.copy()
    rotated.arrays[magmom_key] = np.asarray(
        atoms.arrays[magmom_key] @ rotation.T,
        dtype=atoms.arrays[magmom_key].dtype,
    )
    if rotate_geometry:
        rotated.positions[:] = atoms.positions @ rotation.T
        rotated.set_cell(np.asarray(atoms.cell) @ rotation.T, scale_atoms=False)
    return rotated


def build_graphs(
    atoms_list: list[Atoms],
    model: torch.nn.Module,
    magmom_key: str,
):
    keyspec = KeySpecification.from_defaults()
    keyspec.arrays_keys["initial_noncollinear_magmoms"] = magmom_key
    max_neighbors = model.get_max_neighbors()
    graphs = []
    for atoms in atoms_list:
        graph = from_atoms(
            model.get_torch_element(),
            atoms,
            model.get_cutoff(),
            max_neighbors="inf" if max_neighbors is None else max_neighbors,
            keyspec=keyspec,
            target_property=[],
            embedding_property=model.get_embedding_property(),
            training=False,
            neighborlist_backend="matscipy",
        )
        # Magnetic interactions consume this field directly.  It is not
        # necessarily listed as a universal embedding in older checkpoints.
        graph["initial_noncollinear_magmoms"] = torch.as_tensor(
            np.asarray(atoms.arrays[magmom_key]),
            dtype=model.get_model_dtype(),
        )
        graphs.append(graph)
    return graphs


def predict_energies(
    graphs,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    energies = []
    loader = DataLoader(graphs, batch_size=16, shuffle=False)
    with torch.inference_mode():
        for batch in loader:
            output = model(batch.to(device))
            if "energy" not in output:
                raise RuntimeError("The TACE model does not provide an energy output.")
            energies.append(output["energy"].detach().cpu().reshape(-1))
    return torch.cat(energies).numpy()


def print_errors(name: str, errors: np.ndarray, errors_per_atom: np.ndarray):
    mae = np.mean(np.abs(errors)) * 1000.0
    rmse = np.sqrt(np.mean(errors**2)) * 1000.0
    mae_per_atom = np.mean(np.abs(errors_per_atom)) * 1000.0
    rmse_per_atom = np.sqrt(np.mean(errors_per_atom**2)) * 1000.0
    print(f"\n{name}")
    print(f"  MAE:           {mae:.8g} meV")
    print(f"  RMSE:          {rmse:.8g} meV")
    print(f"  MAE per atom:  {mae_per_atom:.8g} meV/atom")
    print(f"  RMSE per atom: {rmse_per_atom:.8g} meV/atom")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Classify the observable SOC behavior of a magnetic TACE model by "
            "comparing spin-only and joint rotations, following mMACE."
        )
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="TACE checkpoint or exported model (.ckpt, .pt, .pth, or .pt2).",
    )
    parser.add_argument(
        "-i",
        "--input",
        default="/home/xuzemin/mag-o2/Fe-Fmag.xyz",
        help="ASE-readable magnetic structures.",
    )
    parser.add_argument(
        "-n",
        "--num-structures",
        type=int,
        default=None,
        help="Number of leading structures to evaluate; default: all.",
    )
    parser.add_argument(
        "-k",
        "--magmom_key",
        default="spin",
        help="ASE per-atom array containing non-collinear magnetic moments.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_structures is not None and args.num_structures < 1:
        raise ValueError("--num-structures must be positive.")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    index = ":" if args.num_structures is None else f":{args.num_structures}"
    atoms_list = ase.io.read(args.input, index=index)
    if not atoms_list:
        raise RuntimeError(f"No structures were read from {args.input}.")

    model = load_tace(
        args.model,
        device=device,
        target_property=["energy"],
        dtype="float64",
    )
    model.eval()

    torch.manual_seed(42)
    rotations = o3.rand_matrix(12, dtype=torch.float64).numpy()
    spin_errors = []
    joint_errors = []
    spin_errors_per_atom = []
    joint_errors_per_atom = []

    for structure, atoms in enumerate(atoms_list):
        if args.magmom_key not in atoms.arrays:
            raise KeyError(f"Structure {structure} has no array {args.magmom_key!r}.")
        if atoms.arrays[args.magmom_key].shape != (len(atoms), 3):
            raise ValueError(
                f"Structure {structure}: {args.magmom_key!r} must have shape "
                f"({len(atoms)}, 3)."
            )
        variants = [atoms]
        for rotation in rotations:
            variants.append(
                rotate_structure(
                    atoms,
                    rotation,
                    args.magmom_key,
                    rotate_geometry=False,
                )
            )
            variants.append(
                rotate_structure(
                    atoms,
                    rotation,
                    args.magmom_key,
                    rotate_geometry=True,
                )
            )

        graphs = build_graphs(
            variants,
            model,
            args.magmom_key,
        )
        energies = predict_energies(graphs, model, device)
        spin_error = energies[1::2] - energies[0]
        joint_error = energies[2::2] - energies[0]
        spin_errors.append(spin_error)
        joint_errors.append(joint_error)
        spin_errors_per_atom.append(spin_error / len(atoms))
        joint_errors_per_atom.append(joint_error / len(atoms))

    spin_errors = np.concatenate(spin_errors)
    joint_errors = np.concatenate(joint_errors)
    spin_errors_per_atom = np.concatenate(spin_errors_per_atom)
    joint_errors_per_atom = np.concatenate(joint_errors_per_atom)

    print("All errors are multiplied by 1000 (eV -> meV).")
    print_errors("Spin-only rotation", spin_errors, spin_errors_per_atom)
    print_errors("Joint position-spin rotation", joint_errors, joint_errors_per_atom)


if __name__ == "__main__":
    main()
