################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict

import torch
from torch import Tensor

from .mse_fn import register_loss


@register_loss
def l2mae_energy(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = label.entropy * label.energy_weight
    return torch.mean(
        torch.linalg.vector_norm(label["energy"] - pred["energy"], ord=2, dim=-1)
        * total_weight
    )


@register_loss
def l2mae_energy_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = label.entropy * label.energy_weight
    num_atoms = label.ptr[1:] - label.ptr[:-1]
    return torch.mean(
        torch.linalg.vector_norm(
            (label["energy"] - pred["energy"]) / num_atoms, ord=2, dim=-1
        )
        * total_weight
    )


@register_loss
def l2mae_forces(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.forces_weight)[batch]
    return torch.mean(
        torch.linalg.vector_norm(pred["forces"] - label["forces"], ord=2, dim=-1)
        * total_weight
    )


@register_loss
def l2mae_stress(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = label.entropy * label.stress_weight
    return torch.mean(
        torch.linalg.vector_norm(
            pred["stress"] - label["stress"], ord=2, dim=(1, 2)
        )
        * total_weight
    )


@register_loss
def l2mae_virials(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = label.entropy * label.virials_weight
    return torch.mean(
        torch.linalg.vector_norm(
            pred["virials"] - label["virials"], ord=2, dim=(1, 2)
        )
        * total_weight
    )


@register_loss
def l2mae_virials_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = label.entropy * label.virials_weight
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).view(-1, 1, 1)
    return torch.mean(
        torch.linalg.vector_norm(
            (pred["virials"] - label["virials"]) / num_atoms, ord=2, dim=(1, 2)
        )
        * total_weight
    )


@register_loss
def l2mae_direct_forces(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.direct_forces_weight)[batch]
    return torch.mean(
        torch.linalg.vector_norm(pred["direct_forces"] - label["direct_forces"], ord=2, dim=-1)
        * total_weight
    )

# @register_loss # TODO
# def l2mae_direct_hessians(
#     pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
# ) -> torch.Tensor:
#     key = 'direct_hessians'
#     ptr = label['ptr']
#     error = pred[key] - label[key] # [-1]
#     error_list = []
#     offset = 0
#     for start, end in zip(ptr[:-1], ptr[1:]):
#         n = end - start
#         length = 9*n**2
#         this_error = error[offset:offset+length]
#         this_error = this_error.reshape(n, 3, n, 3).permute(0, 2, 1, 3).contiguous().view(-1, 3, 3)
#         offset += length
#         error_list.append(this_error)
#     error = torch.cat(error_list, dim=0)
#     print(error[0, :, :])
#     return torch.mean(
#         torch.linalg.vector_norm(error, ord=2, dim=(1, 2))
#         # * total_weight
#     )
@register_loss
def l2mae_direct_diagonal_hessian(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    key = 'direct_diagonal_hessian'
    batch = label['batch']
    total_weight = (label['entropy'] * label[key+'_weight'])[batch]
    return torch.mean(
        torch.linalg.vector_norm(
            pred[key] - label[key], ord=2, dim=(1, 2)
        )
        * total_weight.unsqueeze(-1).unsqueeze(-1)
    )

@register_loss
def l2mae_direct_stress(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = label.entropy * label.direct_stress_weight
    return torch.mean(
        torch.linalg.vector_norm(
            pred["direct_stress"] - label["direct_stress"], ord=2, dim=(1, 2)
        )
        * total_weight
    )


@register_loss
def l2mae_direct_virials_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = label.entropy * label.direct_virials_weight
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).view(-1, 1, 1)
    return torch.mean(
        torch.linalg.vector_norm(
            (pred["direct_virials"] - label["direct_virials"]) / num_atoms,
            ord=2,
            dim=(1, 2),
        )
        * total_weight
    )


@register_loss
def l2mae_direct_dipole(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.direct_dipole_weight)[batch]
    return torch.mean(
        torch.linalg.vector_norm(
            label["direct_dipole"] - pred["direct_dipole"], ord=2, dim=(1, 2),
        )
        * total_weight
    )


@register_loss
def l2mae_direct_dipole_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.direct_dipole_weight)[batch]
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).view(-1, 1)
    return torch.mean(
        torch.linalg.vector_norm(
            (label["direct_dipole"] - pred["direct_dipole"]) / num_atoms,
            ord=2,
            dim=-1,
        )
        * total_weight
    )


@register_loss
def l2mae_charges(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.charges_weight)[batch]
    return torch.mean(
        torch.linalg.vector_norm(
            pred["charges"] - label["charges"], ord=2, dim=-1
        )
        * total_weight
    )


@register_loss
def l2mae_total_collinear_magmom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    key = "total_collinear_magmom"
    total_weight = [label.entropy * label.total_collinear_magmom_weight]
    return torch.mean(
        torch.linalg.vector_norm(label[key] - pred[key], ord=2, dim=-1)
        * total_weight
    )


@register_loss
def l2mae_total_collinear_magmom_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    key = "total_collinear_magmom"
    total_weight = label.entropy * label.total_collinear_magmom_weight
    num_atoms = label.ptr[1:] - label.ptr[:-1]
    return torch.mean(
        torch.linalg.vector_norm(
            (label[key] - pred[key]) / num_atoms, ord=2, dim=-1
        )
        * total_weight
    )


@register_loss
def l2mae_total_noncollinear_magmom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    key = "total_noncollinear_magmom"
    total_weight = (label.entropy * label.total_noncollinear_magmom_weight)
    return torch.mean(
        torch.linalg.vector_norm(label[key] - pred[key], ord=2, dim=-1)
        * total_weight
    )


@register_loss
def l2mae_total_noncollinear_magmom_per_atom(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    key = "total_noncollinear_magmom"
    total_weight = (label.entropy * label.total_noncollinear_magmom_weight)
    num_atoms = (label.ptr[1:] - label.ptr[:-1]).unsqueeze(-1)
    return torch.mean(
        torch.linalg.vector_norm(
            (label[key] - pred[key]) / num_atoms, ord=2, dim=-1
        )
        * total_weight
    )