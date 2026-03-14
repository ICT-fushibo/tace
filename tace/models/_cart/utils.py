################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Tuple, Optional, NamedTuple


import torch


from ...utils.torch_scatter import scatter_sum


def add_dict_to_left(
    T1: Dict[int, torch.Tensor], T2: Dict[int, torch.Tensor]
) -> Dict[int, torch.Tensor]:

    for k in T2:
        if k in T1:
            T1[k] = T1[k] + T2[k]
        else:
            T1[k] = T2[k]
    return T1


def add_dict_to_right(
        T1: Dict[int, torch.Tensor], T2: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:

    for k in T1:
        if k in T2:
            T2[k] = T2[k] + T1[k]
        else:
            T2[k] = T1[k]
    return T2


def compute_fixed_charge_dipole(
    charges: torch.Tensor,
    positions: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    mu = positions * charges.unsqueeze(-1) * 4.8032047  # e·Å to Debye
    return scatter_sum(src=mu, index=batch.unsqueeze(-1), dim=0, dim_size=num_graphs)

def dict2flatten(max_r: int, t: Dict[int, torch.Tensor]):
    tmp = []
    B, C = t[0].shape[:2]
    for k in sorted(t.keys()):
        flat = t[k].reshape(B, C, -1) 
        tmp.append(flat)
    return torch.cat(tmp, dim=-1).reshape(B, -1)


def flatten2dict(max_r: int, t: torch.Tensor, C: int) -> Dict[int, torch.Tensor]:
    B = t.size(0)
    ndim = (3 ** (max_r + 1) - 1) // 2
    t = t.reshape(B, C, ndim)  
    outs = {}
    start_idx = 0
    for r in range(max_r+1):
        shape = (B, C,) + (3,) * r
        delta = 3**r
        outs[r] = t[:, :, start_idx:start_idx+delta].reshape(shape)
        start_idx += delta
    return outs


