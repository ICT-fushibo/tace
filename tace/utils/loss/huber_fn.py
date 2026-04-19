################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict


import torch
from torch import Tensor


from .mse_fn import register_loss


@register_loss
def huber_energy(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = label.entropy * label.energy_weight
    key = "energy"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_energy_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = label.entropy * label.energy_weight
    num_atoms = label.ptr[1:] - label.ptr[:-1]
    key = "energy"
    return torch.nn.functional.huber_loss(
        total_weight * label[key] / num_atoms,
        total_weight * pred[key] / num_atoms,
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def real_huber_forces(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.forces_weight)[batch].unsqueeze(-1)
    key = "forces"
    return torch.nn.functional.huber_loss(
        total_weight * pred[key],
        total_weight * label[key],
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_forces(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    """Same logic as conditional_huber_forces,"""
    """from MACE https://github.com/ACEsuit/mace"""
    pred_forces = pred["forces"]
    label_forces = label["forces"]
    factors = huber_delta * torch.tensor(
        [1.0, 0.7, 0.4, 0.1], device=label_forces.device, dtype=label_forces.dtype
    )
    norm_forces = torch.norm(label_forces, dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    c4 = ~(c1 | c2 | c3)
    se = torch.zeros_like(pred_forces)
    se[c1] = torch.nn.functional.huber_loss(
        label_forces[c1], pred_forces[c1], reduction="none", delta=factors[0]
    )
    se[c2] = torch.nn.functional.huber_loss(
        label_forces[c2], pred_forces[c2], reduction="none", delta=factors[1]
    )
    se[c3] = torch.nn.functional.huber_loss(
        label_forces[c3], pred_forces[c3], reduction="none", delta=factors[2]
    )
    se[c4] = torch.nn.functional.huber_loss(
        label_forces[c4], pred_forces[c4], reduction="none", delta=factors[3]
    )
    return torch.mean(se)

@register_loss
def huber_stress(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (label.entropy * label.stress_weight).unsqueeze(-1).unsqueeze(-1)
    key = "stress"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_virials(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (label.entropy * label.virials_weight).unsqueeze(-1).unsqueeze(-1)
    key = "virials"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_virials_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (label.entropy * label.virials_weight).unsqueeze(-1).unsqueeze(-1)
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).view(-1, 1, 1)
    key = "virials"
    return torch.nn.functional.huber_loss(
        total_weight * label[key] / num_atoms,
        total_weight * pred[key] / num_atoms,
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_direct_forces(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    batch = label.batch
    total_weight = label.entropy[batch].unsqueeze(-1) * label.direct_forces_weight[
        batch
    ].unsqueeze(-1)
    key = "direct_forces"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_direct_stress(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (label.entropy * label.direct_stress_weight).unsqueeze(-1).unsqueeze(-1)
    key = "direct_stress"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_direct_virials_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (label.entropy * label.direct_virials_weight).unsqueeze(-1).unsqueeze(-1)
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).view(-1, 1, 1)
    key = "direct_virials"
    return torch.nn.functional.huber_loss(
        total_weight * label[key] / num_atoms,
        total_weight * pred[key] / num_atoms,
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_direct_diagonal_hessian(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    key = 'direct_diagonal_hessian'
    batch = label['batch']
    total_weight = (label['entropy'] * label[key+'_weight'])[batch].unsqueeze(-1).unsqueeze(-1)
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_direct_dipole(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (label.entropy * label.direct_dipole_weight).unsqueeze(-1)
    key = "direct_dipole"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_direct_dipole_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (label.entropy * label.direct_dipole_weight).unsqueeze(-1)
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).unsqueeze(-1)
    key = "direct_dipole"
    return torch.nn.functional.huber_loss(
        total_weight * label[key] / num_atoms,
        total_weight * pred[key] / num_atoms,
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_conservative_dipole(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (label.entropy * label.conservative_dipole_weight).unsqueeze(-1)
    key = "conservative_dipole"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_conservative_dipole_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (label.entropy * label.conservative_dipole_weight).unsqueeze(-1)
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).unsqueeze(-1)
    key = "conservative_dipole"
    return torch.nn.functional.huber_loss(
        total_weight * label[key] / num_atoms,
        total_weight * pred[key] / num_atoms,
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_direct_polarizability(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (
        (label.entropy * label.direct_polarizability_weight).unsqueeze(-1).unsqueeze(-1)
    )
    key = "direct_polarizability"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_direct_polarizability_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (
        (label.entropy * label.direct_polarizability_weight).unsqueeze(-1).unsqueeze(-1)
    )
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).unsqueeze(-1).unsqueeze(-1)
    key = "direct_polarizability"
    return torch.nn.functional.huber_loss(
        total_weight * label[key] / num_atoms,
        total_weight * pred[key] / num_atoms,
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_conservative_polarizability(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (
        (label.entropy * label.conservative_polarizability_weight).unsqueeze(-1).unsqueeze(-1)
    )
    key = "conservative_polarizability"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_conservative_polarizability_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = (
        (label.entropy * label.conservative_polarizability_weight).unsqueeze(-1).unsqueeze(-1)
    )
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).unsqueeze(-1).unsqueeze(-1)
    key = "conservative_polarizability"
    return torch.nn.functional.huber_loss(
        total_weight * label[key] / num_atoms,
        total_weight * pred[key] / num_atoms,
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_born_effective_charges(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    batch = label.batch
    total_weight = (
        (label.entropy * label.born_effective_charges_weight)[batch]
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    key = "born_effective_charges"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_polarization_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    lattice = label["lattice"]
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).unsqueeze(-1)
    key = "polarization"
    error = pred[key] - label[key]
    error = torch.einsum("bi, bij -> bj", error, torch.linalg.inv(lattice))
    error = torch.remainder(error, 1.0)
    error = torch.where(error > 0.5, error - 1.0, error)
    error = torch.where(error < -0.5, error + 1.0, error)
    error = torch.einsum("bi, bij -> bj", error, lattice)
    error_per_atom = error / num_atoms
    return torch.nn.functional.huber_loss(
        error_per_atom,
        torch.zeros_like(error_per_atom),
        delta=huber_delta,
        reduction="mean",
    )

@register_loss
def huber_charges(pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01) -> torch.Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.charges_weight)[batch]
    key = "charges"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_final_collinear_magmoms(pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01) -> torch.Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.final_collinear_magmoms_weight)[batch]
    key = "final_collinear_magmoms"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_abs_final_collinear_magmoms(pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01) -> torch.Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.abs_final_collinear_magmoms_weight)[batch]
    key = "abs_final_collinear_magmoms"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )


@register_loss
def huber_final_noncollinear_magmoms(pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01) -> torch.Tensor:
    batch = label.batch
    total_weight = label.entropy[batch].unsqueeze(-1) * label.final_collinear_magmoms_weight[
        batch
    ].unsqueeze(-1)
    key = "final_noncollinear_magmoms"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_collinear_magnetic_forces(pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01) -> torch.Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.collinear_magnetic_forces_weight)[batch]
    key = "collinear_magnetic_forces"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_noncollinear_magnetic_forces(pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01) -> torch.Tensor:
    batch = label.batch
    total_weight = label.entropy[batch].unsqueeze(-1) * label.noncollinear_magnetic_forces_weight[
        batch
    ].unsqueeze(-1)
    key = "noncollinear_magnetic_forces"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_total_collinear_magmom(pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01) -> torch.Tensor:
    total_weight = label.entropy * label.total_collinear_magmom_weight
    key = "total_collinear_magmom"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_total_collinear_magmom_per_atom(pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01) -> torch.Tensor:
    total_weight = label.entropy * label.total_collinear_magmom_weight
    num_atoms = label.ptr[1:] - label.ptr[:-1]
    key = "total_collinear_magmom"
    return torch.nn.functional.huber_loss(
        total_weight * label[key] / num_atoms,
        total_weight * pred[key] / num_atoms,
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_total_noncollinear_magmom(pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01) -> torch.Tensor:
    total_weight = (label.entropy * label.total_noncollinear_magmom_weight).unsqueeze(-1)
    key = "total_noncollinear_magmom"
    return torch.nn.functional.huber_loss(
        total_weight * label[key],
        total_weight * pred[key],
        reduction="mean",
        delta=huber_delta,
    )

@register_loss
def huber_total_noncollinear_magmom_per_atom(pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01) -> torch.Tensor:
    total_weight = (label.entropy * label.total_noncollinear_magmom_weight).unsqueeze(-1)
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).unsqueeze(-1)
    key = "total_noncollinear_magmom"
    return torch.nn.functional.huber_loss(
        total_weight * label[key] / num_atoms,
        total_weight * pred[key] / num_atoms,
        reduction="mean",
        delta=huber_delta,
    )
