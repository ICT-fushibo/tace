################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict


import torch
from torch import Tensor


from .mse_fn import register_loss


@register_loss
def l2mae_forces(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    return torch.mean(
        torch.linalg.vector_norm(pred["forces"] - label["forces"], ord=2, dim=-1)
    )

@register_loss
def l2mae_stress(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    return torch.mean(
        torch.linalg.vector_norm(pred["stress"] - label["stress"], ord=2, dim=(1, 2))
    )

@register_loss
def l2mae_direct_forces(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    return torch.mean(
        torch.linalg.vector_norm(pred["direct_forces"] - label["direct_forces"], ord=2, dim=-1)
    )

@register_loss
def l2mae_direct_stress(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    return torch.mean(
        torch.linalg.vector_norm(pred["direct_stress"] - label["direct_stress"], ord=2, dim=(1, 2))
    )

@register_loss
def conditional_huber_forces(
    pred_forces: Tensor, label_forces: Tensor, huber_delta: float
) -> Tensor:
    "From MACE https://github.com/ACEsuit/mace"
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

@register_loss # TODO, check
def l2mae_hessians(
        pred: Dict[str, Tensor], 
        label: Dict[str, Tensor],
    ) -> torch.Tensor:

    true_hessian_flat = label["hessians"]
    num_atoms_per_graph = label["ptr"][1:] - label["ptr"][:-1]
    jacs_per_graph = pred["jacs_per_graph"]
    samples_per_graph = pred["samples_per_graph"]

    offset = 0
    losses = []

    for jac_pred, samples, n_g in zip(jacs_per_graph, samples_per_graph, num_atoms_per_graph):
        hess_size = n_g * 3 * n_g * 3
        hess_flat_g = true_hessian_flat[offset : offset + hess_size]
        hess_true = hess_flat_g.reshape(n_g, 3, n_g, 3)
        offset += hess_size

        atom_idx = samples[:, 0]
        xyz_idx = samples[:, 1]
        jac_true = hess_true[atom_idx, xyz_idx]
        diff = jac_pred - jac_true  # (k_g, n_g, 3)
        row_norm = torch.norm(diff, p=2, dim=-1)  # (k_g, n_g)
        loss_g = row_norm.sum(dim=1).mean(dim=0)
        if hess_true.abs().max().item() > 1e4:
            loss_g = loss_g * 1e-8
        losses.append(loss_g)

    return torch.stack(losses).mean()
