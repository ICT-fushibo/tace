################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import itertools


import torch

def nonsym_tensor_to_sym(T: torch.Tensor) -> torch.Tensor:

    rank = T.ndim - 2
    perm_indices = list(range(-rank, 0))
    perms = list(itertools.permutations(perm_indices))

    sym_Ts = []
    for perm in perms:
        full_perm = list(range(T.ndim - rank)) + [T.ndim + i for i in perm]
        permuted_T = T.permute(full_perm)
        sym_Ts.append(permuted_T)

    sym_T = torch.stack(sym_Ts, dim=0).mean(dim=0)
    return sym_T


def delta_tensor(i: int, j: int, ndim: int, device=None, dtype=None) -> torch.Tensor:

    delta = torch.eye(3, device=device, dtype=dtype)
    for _ in range(ndim - 2):
        delta = delta.unsqueeze(0)
    perm = list(range(ndim))
    perm[i], perm[-2] = perm[-2], perm[i]
    perm[j], perm[-1] = perm[-1], perm[j]
    delta = delta.permute(*perm)
    return delta


def sym_tensor_to_traceless(T: torch.Tensor) -> torch.Tensor:
    """
    For r >= 4, numerical errors can be significant; full tracelessness requires using float64

    Compute the first-order, second-order, ..., up to floor(n/2) traces step by step,
    and subtract their corresponding contributions to obtain a fully symmetric traceless tensor
    """

    B, C = T.shape[:2]
    spatial_shape = T.shape[2:]

    T = T.view(B * C, *spatial_shape)

    ndim = T.ndim
    n = ndim - 1

    result = T.clone()
    base_combs = list(itertools.combinations(range(-n, 0), 2))
    for k in range(1, n // 2 + 1):
        denom = 1.0
        for j in range(2, k + 2):
            denom *= 3 + 2 * (n - j)
        coeff = ((-1) ** k) / denom
        corr = torch.zeros_like(T)
        for pairs in itertools.combinations(base_combs, k):
            idxs = [idx for pair in pairs for idx in pair]
            if len(set(idxs)) < 2 * k:
                continue
            delta = torch.ones_like(T)
            for i, j in pairs:
                delta = delta * delta_tensor(i, j, ndim, device=T.device, dtype=T.dtype)
            trace = torch.sum(T * delta, dim=tuple(idxs), keepdim=True)
            corr += delta * trace
        result = result + coeff * corr
    return result.view(B, C, *spatial_shape)
