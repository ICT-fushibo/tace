# TODO

import argparse
import numpy as np

import ase.io

import torch
torch.set_default_dtype(torch.float32)
from torch_geometric.loader import DataLoader

from tace.lightning import load_tace
from tace.dataset.graph import from_atoms
from tace.utils._global import DTYPE
from tace.utils.utils import num_params
from tace.dataset.quantity import KeySpecification, update_keyspec_from_kwargs, PROPERTY
from tace.dataset.read import check_keys



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--ema", type=int, default=1)
    parser.add_argument("--nl_backend", type=str, default="matscipy")
    parser.add_argument("--num_rot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    for k, v in PROPERTY.items():
        if v["enable_prediction"] or v["enable_embedding"]:
            parser.add_argument(f"--{k}_key", type=str, default=k)
    return parser.parse_args()



def random_rotation_matrix(device="cpu"):
    A = torch.randn(3, 3, device=device)
    Q, R = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def rotate_atoms(atoms, R):
    atoms_r = atoms.copy()
    pos = torch.tensor(atoms.get_positions(), dtype=torch.get_default_dtype())
    cell = torch.tensor(atoms.get_cell().array, dtype=torch.get_default_dtype())
    R = R.cpu()
    pos_r = pos @ R.T
    cell_r = cell @ R.T
    atoms_r.set_positions(pos_r.numpy())
    atoms_r.set_cell(cell_r.numpy(), scale_atoms=False)
    return atoms_r


def build_dataset(atoms_list, model, args, key_spec):
    max_neighbors = model.get_max_neighbors()
    cutoff = model.get_cutoff()
    element = model.get_torch_element()
    target_property = model.get_target_property()
    embedding_property = model.get_embedding_property()
    dataset = [
        from_atoms(
            element,
            atoms,
            cutoff,
            max_neighbors="inf" if max_neighbors is None else max_neighbors,
            keyspec=key_spec,
            target_property=target_property,
            embedding_property=embedding_property,
            neighborlist_backend=args.nl_backend,
        )
        for atoms in check_keys(
            atoms_list,
            target_property,
            key_spec,
            embedding_property,
            training=False,
        )
    ]

    return dataset


def predict(model, loader, device):
    forces_all = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        F = out["direct_forces"]
        natoms = torch.bincount(batch.batch)
        start = 0
        for n in natoms:
            end = start + n
            forces_all.append(F[start:end].detach().cpu())
            start = end
    return forces_all


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    atoms_list = ase.io.read(args.input, index=":")
    key_spec = KeySpecification()
    update_keyspec_from_kwargs(key_spec, vars(args))
    model = load_tace(args.model, args.device, strict=True, use_ema=args.ema)
    print("Params:", num_params(model))
    model.eval().to(args.device)
    dtype = DTYPE[args.dtype]
    torch.set_default_dtype(dtype)
    model.to(dtype)

    # Original prediction
    dataset = build_dataset(atoms_list, model, args, key_spec)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    F_orig = predict(model, loader, args.device)
    print("\nEquivariance Test ===")

    all_errors = []
    for i, atoms in enumerate(atoms_list):
        F = F_orig[i]
        print(f"\nStructure {i}")
        for r in range(args.num_rot):
            R = random_rotation_matrix(args.device)
            atoms_r = rotate_atoms(atoms, R)
            dataset_r = build_dataset([atoms_r], model, args, key_spec)
            loader_r = DataLoader(dataset_r, batch_size=1)
            F_r = predict(model, loader_r, args.device)[0].to(R.device)
            F_exp = F.to(R.device) @ R.T
            err = torch.max(torch.abs(F_r - F_exp)).item()
            rel = err / (torch.max(torch.abs(F_exp)).item() + 1e-12)
            ok = torch.allclose(F_r, F_exp, rtol=args.rtol, atol=args.atol)
            all_errors.append(err)
            print(f"rot {r:02d} | err={err:.3e} | rel={rel:.3e} | ok={ok}")

    all_errors = np.array(all_errors)

    print("\n==============================")
    print("FINAL SUMMARY")
    print("==============================")
    print("max abs error:", all_errors.max())
    print("mean abs error:", all_errors.mean())
    print("std abs error:", all_errors.std())

    print("\nPASS:", np.all(all_errors < args.atol))


if __name__ == "__main__":
    main()