################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Tuple, Optional, NamedTuple


import torch


from cartnn.util import scatter_sum
from cartnn.o3 import expand_dims_to


def add_to_left(
    T1: Dict[int, torch.Tensor], T2: Dict[int, torch.Tensor]
) -> Dict[int, torch.Tensor]:

    for k in T2:
        if k in T1:
            T1[k] = T1[k] + T2[k]
        else:
            T1[k] = T2[k]
    return T1


def add_to_right(
        T1: Dict[int, torch.Tensor], T2: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:

    for k in T1:
        if k in T2:
            T2[k] = T2[k] + T1[k]
        else:
            T2[k] = T1[k]
    return T2


def satisfy(r_1:int, r_2: int, restriction, r_o: Optional[int] = None):
    r_1_r_2 = None; bool_1 = True
    r_o_r_1 = None; bool_2 = True

    if isinstance(restriction, str):
        r_1_r_2 = restriction
    elif isinstance(restriction, Dict):
        r_1_r_2 = restriction.get('r_1_r_2', None)
        r_o_r_1 = restriction.get('r_o_r_1', None)
    else:
        return True
    
    if r_1_r_2 == "<=":
        bool_1 = (r_1 <= r_2)
    elif r_1_r_2 == "==":
        bool_1 = (r_1 == r_2)
    else:
        bool_1 = True
    
    if r_o is not None:
        if r_o_r_1 == "<=":
            bool_2 = (r_o <= r_1)
        elif r_o_r_1 == "==":
            bool_2 = (r_o == r_1)
        else:
            bool_2 = True
    return bool_1 and bool_2


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


def vec_to_skew(v: torch.Tensor) -> torch.Tensor:
    """ TODO, maybe not (1,1,1) to (2,1,1), should use basis change
    v: (B, 3) tensor
    return: (B, 3, 3) skew-symmetric matrix
    """
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    zero = torch.zeros_like(x)

    row1 = torch.stack([zero, -z, y], dim=1)
    row2 = torch.stack([z, zero, -x], dim=1)
    row3 = torch.stack([-y, x, zero], dim=1)

    skew = torch.stack([row1, row2, row3], dim=1)
    return skew


def select_corresponding_level_for_scalar(
        x: torch.Tensor, node_level: torch.Tensor, num_levels: int
    ) -> torch.Tensor:
    '''
    For rank-0 tensor, 
    '''
    B = x.size(0)
    C_LEVELS = x.size(1)
    mask = torch.zeros(B, num_levels, C_LEVELS // num_levels, device=x.device, dtype=x.dtype)
    idx = torch.arange(B, device=x.device, dtype=torch.int64)
    mask[idx, node_level, :] = 1
    mask = mask.reshape((B, C_LEVELS))
    return x * mask

def select_corresponding_level_for_tensor(
        x: torch.Tensor, node_level: torch.Tensor, num_levels: int
    ) -> torch.Tensor:
    '''
    For rank>0 tensor, 
    '''
    B = x.size(0)
    C_LEVELS = x.size(1)

    mask = torch.zeros(B, num_levels, C_LEVELS // num_levels, device=x.device, dtype=x.dtype)
    idx = torch.arange(B, device=x.device, dtype=torch.int64)
    mask[idx, node_level, :] = 1
    mask = mask.reshape((B,C_LEVELS))
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


