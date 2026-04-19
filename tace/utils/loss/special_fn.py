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
