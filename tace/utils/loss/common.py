################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict

import torch


def num_atoms_per_graph(label: Dict[str, torch.Tensor]) -> torch.Tensor:
    return label["ptr"][1:] - label["ptr"][:-1]


def polarization_error_per_atom(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    key: str = "polarization",
) -> torch.Tensor:
    lattice = label["lattice"]
    num_atoms = num_atoms_per_graph(label).reshape(-1, 1)
    error = pred[key] - label[key]
    error = torch.einsum("bi, bij -> bj", error, torch.linalg.inv(lattice))
    error = torch.remainder(error, 1.0)
    error = torch.where(error > 0.5, error - 1.0, error)
    error = torch.where(error < -0.5, error + 1.0, error)
    error = torch.einsum("bi, bij -> bj", error, lattice)
    return error / num_atoms


def voigt6_stress(stress: torch.Tensor) -> torch.Tensor:
    return stress.reshape(-1, 9)[:, [0, 4, 8, 5, 2, 1]]
