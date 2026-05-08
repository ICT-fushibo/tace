################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Callable, List


import torch
from torch import Tensor


LOSS_FN: Dict[str, Callable] = {}


def register_loss(func: Callable = None, *, key: str = None):
    def decorator(f: Callable):
        k = key if key is not None else f.__name__
        LOSS_FN[k] = f
        return f

    if func is None:
        return decorator
    else:
        return decorator(func)

@register_loss
def mse_energy(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = label.entropy * label.energy_weight
    return torch.mean(torch.square((label["energy"] - pred["energy"])) * total_weight)

@register_loss
def mse_energy_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = label.entropy * label.energy_weight
    return torch.mean(
        torch.square(
            (label["energy"] - pred["energy"]) / (label.ptr[1:] - label.ptr[:-1])
        )
        * total_weight
    )

@register_loss
def mse_forces(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    batch = label.batch
    total_weight = label.entropy[batch].unsqueeze(-1) * label.forces_weight[
        batch
    ].unsqueeze(-1)
    return torch.mean(torch.square(pred["forces"] - label["forces"]) * total_weight)

@register_loss
def mse_stress(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = label.entropy * label.stress_weight
    return torch.mean(
        torch.square(pred["stress"] - label["stress"])
        * total_weight.unsqueeze(-1).unsqueeze(-1)
    )

@register_loss
def mse_virials(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = label.entropy * label.virials_weight
    return torch.mean(
        torch.square((pred["virials"] - label["virials"]))
        * total_weight.unsqueeze(-1).unsqueeze(-1)
    )

@register_loss
def mse_virials_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = label.entropy * label.virials_weight
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).view(-1, 1, 1)
    return torch.mean(
        torch.square((pred["virials"] - label["virials"]) / num_atoms)
        * total_weight.unsqueeze(-1).unsqueeze(-1)
    )

@register_loss
def mse_direct_forces(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    batch = label.batch
    total_weight = label.entropy[batch].unsqueeze(-1) * label.direct_forces_weight[
        batch
    ].unsqueeze(-1)
    return torch.mean(torch.square(pred["direct_forces"] - label["direct_forces"]) * total_weight)

# torch.set_printoptions(sci_mode=False, precision=4)
@register_loss
def mse_direct_diagonal_hessian(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    key = 'direct_diagonal_hessian'
    batch = label['batch']
    total_weight = (label['entropy'] * label[key+'_weight'])[batch]
    # print(1, pred[key][0, :, :])
    # print(2, label[key][0, :, :])
    return torch.mean(
        torch.square((pred[key] - label[key]))
        * total_weight.unsqueeze(-1).unsqueeze(-1)
    )


@register_loss
def mse_direct_stress(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = label.entropy * label.direct_stress_weight
    return torch.mean(
        torch.square(pred["direct_stress"] - label["direct_stress"])
        * total_weight.unsqueeze(-1).unsqueeze(-1)
    )

@register_loss
def mse_direct_virials_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = label.entropy * label.direct_virials_weight
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).view(-1, 1, 1)
    return torch.mean(
        torch.square((pred["direct_virials"] - label["direct_virials"]) / num_atoms)
        * total_weight.unsqueeze(-1).unsqueeze(-1)
    )

@register_loss
def mse_direct_dipole(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (label.entropy * label.direct_dipole_weight).unsqueeze(-1)
    return torch.mean(torch.square((label["direct_dipole"] - pred["direct_dipole"])) * total_weight)


@register_loss
def mse_direct_dipole_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (label.entropy * label.direct_dipole_weight).unsqueeze(-1)
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).unsqueeze(-1)
    return torch.mean(
        torch.square((label["direct_dipole"] - pred["direct_dipole"]) / num_atoms) * total_weight
    )

@register_loss
def mse_conservative_dipole(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (label.entropy * label.conservative_dipole_weight).unsqueeze(-1)
    return torch.mean(
        torch.square(label["conservative_dipole"] - pred["conservative_dipole"]) * total_weight
    )

@register_loss
def mse_conservative_dipole_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (label.entropy * label.conservative_dipole_weight).unsqueeze(-1)
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).unsqueeze(-1)
    return torch.mean(
        torch.square((label["conservative_dipole"] - pred["conservative_dipole"]) / num_atoms) * total_weight
    )

@register_loss
def mse_direct_polarizability(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (
        (label.entropy * label.direct_polarizability_weight).unsqueeze(-1).unsqueeze(-1)
    )
    return torch.mean(
        torch.square(label["direct_polarizability"] - pred["direct_polarizability"])
        * total_weight
    )

@register_loss
def mse_direct_polarizability_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (
        (label.entropy * label.direct_polarizability_weight).unsqueeze(-1).unsqueeze(-1)
    )
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).unsqueeze(-1).unsqueeze(-1)
    return torch.mean(
        torch.square((label["direct_polarizability"] - pred["direct_polarizability"]) / num_atoms)
        * total_weight
    )

@register_loss
def mse_conservative_polarizability(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (
        (label.entropy * label.conservative_polarizability_weight).unsqueeze(-1).unsqueeze(-1)
    )
    return torch.mean(
        torch.square(label["conservative_polarizability"] - pred["conservative_polarizability"])
        * total_weight
    )

@register_loss
def mse_conservative_polarizability_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (
        (label.entropy * label.conservative_polarizability_weight).unsqueeze(-1).unsqueeze(-1)
    )
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).unsqueeze(-1).unsqueeze(-1)
    return torch.mean(
        torch.square((label["conservative_polarizability"] - pred["conservative_polarizability"]) / num_atoms)
        * total_weight
    )

@register_loss
def mse_born_effective_charges(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    batch = label.batch
    total_weight = (
        (label.entropy * label.born_effective_charges_weight)[batch]
        .unsqueeze(-1)
        .unsqueeze(-1)
    )

    return torch.mean(
        torch.square(label["born_effective_charges"] - pred["born_effective_charges"])
        * total_weight
    )

@register_loss
def mse_polarization_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    ptr = label["ptr"]
    lattice = label["lattice"]
    num_atoms = (ptr[1:] - ptr[:-1]).reshape(-1, 1)
    error = pred["polarization"] - label["polarization"]
    error = torch.einsum("bi, bij -> bj", error, torch.linalg.inv(lattice))
    error = torch.remainder(error, 1.0)
    error = torch.where(error > 0.5, error - 1.0, error)
    error = torch.where(error < -0.5, error + 1.0, error)
    error = torch.einsum("bi, bij -> bj", error, lattice)
    return torch.mean(torch.square(error / num_atoms))

@register_loss
def mse_charges(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.charges_weight)[batch]
    return torch.mean(torch.square(pred["charges"] - label["charges"]) * total_weight)

@register_loss
def mse_final_collinear_magmoms(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    key = "final_collinear_magmoms"
    batch = label.batch
    total_weight = (label.entropy * label.final_collinear_magmoms_weight)[batch]
    return torch.mean(torch.square(pred[key] - label[key]) * total_weight)

@register_loss
def mse_abs_final_collinear_magmoms(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    key = "abs_final_collinear_magmoms"
    batch = label.batch
    total_weight = (label.entropy * label.abs_final_collinear_magmoms_weight)[batch]
    return torch.mean(torch.square(pred[key] - label[key]) * total_weight)


@register_loss
def mse_final_noncollinear_magmoms(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    key = "final_noncollinear_magmoms"
    batch = label.batch
    total_weight = label.entropy[batch].unsqueeze(-1) * label.final_noncollinear_magmoms_weight[
        batch
    ].unsqueeze(-1)
    return torch.mean(torch.square(pred[key] - label[key]) * total_weight)

@register_loss
def mse_collinear_magnetic_forces(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    key = "collinear_magnetic_forces"
    batch = label.batch
    total_weight = (label.entropy * label.collinear_magnetic_forces_weight)[batch]
    return torch.mean(torch.square(pred[key] - label[key]) * total_weight)

@register_loss
def mse_noncollinear_magnetic_forces(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    key = "noncollinear_magnetic_forces"
    batch = label.batch
    total_weight = label.entropy[batch].unsqueeze(-1) * label.noncollinear_magnetic_forces_weight[
        batch
    ].unsqueeze(-1)
    return torch.mean(
        torch.square(pred[key] - label[key]) * total_weight
    )

@register_loss
def mse_total_collinear_magmom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    key = "total_collinear_magmom"
    total_weight = label.entropy * label.total_collinear_magmom_weight
    return torch.mean(torch.square((label[key] - pred[key])) * total_weight)


@register_loss
def mse_total_collinear_magmom_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    key = "total_collinear_magmom"
    total_weight = label.entropy * label.total_collinear_magmom_weight
    num_atoms = label.ptr[1:] - label.ptr[:-1]
    return torch.mean(
        torch.square(
            (label[key] - pred[key]) / num_atoms
        )
        * total_weight
    )

@register_loss
def mse_total_noncollinear_magmom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    key = "total_noncollinear_magmom"
    total_weight = (label.entropy * label.total_noncollinear_magmom_weight).unsqueeze(-1)
    return torch.mean(torch.square((label[key] - pred[key])) * total_weight)


@register_loss
def mse_total_noncollinear_magmom_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    key = "total_noncollinear_magmom"
    total_weight = (label.entropy * label.total_noncollinear_magmom_weight).unsqueeze(-1)
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).unsqueeze(-1)
    return torch.mean(
        torch.square((label[key] - pred[key]) / num_atoms) * total_weight
    )

