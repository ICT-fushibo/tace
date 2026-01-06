################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Tuple, Optional, NamedTuple


import torch


from ...utils.torch_scatter import scatter_sum


def expand_dims_to(T: torch.Tensor, n_dim: int, dim: int = -1) -> torch.Tensor:
    '''jit-safe'''
    while T.ndim < n_dim:
        T = T.unsqueeze(dim)
    return T


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


def torch_full_3x3_to_voigt_6_stress(stress_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert stress tensor [batch, 3, 3] -> [batch, 6] in Voigt notation,
    matching ASE's full_3x3_to_voigt_6_stress.
    """
    s = stress_tensor
    s_voigt = torch.stack(
        [
            s[..., 0, 0],  # σ_xx
            s[..., 1, 1],  # σ_yy
            s[..., 2, 2],  # σ_zz
            0.5 * (s[..., 1, 2] + s[..., 2, 1]),  # σ_yz
            0.5 * (s[..., 0, 2] + s[..., 2, 0]),  # σ_xz
            0.5 * (s[..., 0, 1] + s[..., 1, 0]),  # σ_xy
        ],
        dim=-1,
    )
    return s_voigt

def select_corresponding_level(
        x: torch.Tensor, node_level: torch.Tensor, num_levels: int
    ) -> torch.Tensor:
    B = x.size(0)
    C_LEVELS = x.size(1)
    mask = torch.zeros(B, num_levels, C_LEVELS // num_levels, device=x.device, dtype=x.dtype)
    idx = torch.arange(B, device=x.device, dtype=torch.int64)
    mask[idx, node_level, :] = 1
    mask = mask.reshape((B, C_LEVELS))
    # return x * mask
    return x * expand_dims_to(mask, x.ndim, -1)


class Graph(NamedTuple):
    lmp: bool
    lmp_data: Optional[torch.Tensor]
    lmp_natoms: Tuple[int, int]
    num_graphs: int
    displacement: Optional[torch.Tensor]
    positions: torch.Tensor
    edge_vector: torch.Tensor
    edge_length: torch.Tensor
    lattice: torch.Tensor
    node_level: torch.Tensor
    num_atoms_arange: torch.Tensor


class LAMMPS_MP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        feats, data = args  # unpack
        ctx.vec_len = feats.shape[-1]
        ctx.data = data
        out = torch.empty_like(feats)
        data.forward_exchange(feats, out, ctx.vec_len)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad,) = grad_outputs  # unpack
        gout = torch.empty_like(grad)
        ctx.data.reverse_exchange(grad, gout, ctx.vec_len)
        return gout, None
    

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


