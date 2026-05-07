################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict


import torch
from torch import Tensor


from .mse_fn import register_loss


@register_loss # TODO, check
def hessian(
        pred: Dict[str, Tensor], 
        label: Dict[str, Tensor],
        huber_delta: float = 0.01,
    ) -> torch.Tensor:

    true_hessian_flat = label["hessian"]
    num_atoms_per_graph = label["ptr"][1:] - label["ptr"][:-1]
    jacs_per_graph = pred["jacs_per_graph"]
    samples_per_graph = pred["samples_per_graph"]

    if jacs_per_graph is None:
        return torch.tensor(0.0).to(label["ptr"].device)
    
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



NOISE_MUL = 0.05


@register_loss
def mse_dens_forces(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.forces_weight)[batch].unsqueeze(-1)
    noise_mask = label['noise_mask'].unsqueeze(-1)
    forces_error = (pred['forces'] - label['forces'])* (~noise_mask)
    noise_error = (pred['noise_vec'] - label['noise_vec'])* noise_mask  * NOISE_MUL
    return torch.mean(torch.square(forces_error + noise_error) * total_weight)


@register_loss
def mse_dens_direct_forces(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.direct_forces_weight)[batch].unsqueeze(-1)
    noise_mask = label['noise_mask'].unsqueeze(-1)
    forces_error = (pred['direct_forces'] - label['direct_forces'])* (~noise_mask)
    noise_error = (pred['noise_vec'] - label['noise_vec'])* noise_mask  * NOISE_MUL
    return torch.mean(torch.square(forces_error + noise_error) * total_weight)


@register_loss
def l2mae_dens_forces(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.forces_weight)[batch]
    noise_mask = label['noise_mask'].unsqueeze(-1)
    forces_error = (pred['forces'] - label['forces'])* (~noise_mask)
    noise_error = (pred['noise_vec'] - label['noise_vec'])* noise_mask  * NOISE_MUL 
    return torch.mean(
        torch.linalg.vector_norm(forces_error + noise_error, ord=2, dim=-1)
        * total_weight
    )

@register_loss
def l2mae_dens_direct_forces(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    batch = label.batch
    total_weight = (label.entropy * label.direct_forces_weight)[batch]
    noise_mask = label['noise_mask'].unsqueeze(-1)
    forces_error = (pred['direct_forces'] - label['direct_forces'])* (~noise_mask)
    noise_error = (pred['noise_vec'] - label['noise_vec'])* noise_mask * NOISE_MUL
    return torch.mean(
        torch.linalg.vector_norm(forces_error + noise_error, ord=2, dim=-1)
        * total_weight
    )


@register_loss
def mae_dens_direct_stress(
    pred: Dict[str, Tensor], label: Dict[str, Tensor], huber_delta: float = 0.01
) -> torch.Tensor:
    total_weight = label.entropy * label.stress_weight
    
    error = (pred["direct_stress"] - label["direct_stress"]) * (~label['dens_batch_mask']).unsqueeze(-1).unsqueeze(-1)
    
    return torch.mean(
        torch.abs(error)
        * total_weight.unsqueeze(-1).unsqueeze(-1)
    )
