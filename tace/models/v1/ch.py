################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


from typing import Dict
from itertools import combinations


import torch
from torch import Tensor


from cartnn import ICTD, expand_dims_to


def factorial(n: int) -> int:
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def double_factorial(n: int) -> int:
    result = 1
    for i in range(n, 0, -2):
        result *= i
    return result


def _norm(n: int) -> float:
    """(2n - 1)!! / n!"""
    num = double_factorial(2 * n - 1)
    den = factorial(n)
    return num / den


def delta_tensor(i: int, j: int, ndim: int, device=None, dtype=None) -> Tensor:
    delta = torch.eye(3, device=device, dtype=dtype)
    for _ in range(ndim - 2):
        delta = delta.unsqueeze(0)
    perm = list(range(ndim))
    perm[i], perm[-2] = perm[-2], perm[i]
    perm[j], perm[-1] = perm[-1], perm[j]
    delta = delta.permute(*perm)
    return delta


def symmetric_outer_product(v: Tensor, n: int, norm: bool = True) -> Tensor:
    out = torch.ones_like(v[..., 0])
    for _ in range(n):
        out = out[..., None] * expand_dims_to(v, out.ndim + 1, dim=v.ndim - 1)
    if norm:
        out = out * _norm(n)
    return out


def subtract_traces(T: Tensor, n: int) -> Tensor:
    result = T.clone()
    base_combs = list(combinations(range(-n, 0), 2))
    for k in range(1, n // 2 + 1):
        denom = 1.0
        for j in range(2, k + 2):
            denom *= 3 + 2 * (n - j)
        coeff = ((-1) ** k) / denom
        corr = torch.zeros_like(T)
        for pairs in combinations(base_combs, k):
            idxs = [idx for pair in pairs for idx in pair]
            if len(set(idxs)) < 2 * k:
                continue
            delta = torch.ones_like(T)
            for i, j in pairs:
                delta = delta * delta_tensor(
                    i, j, n + 1, device=T.device, dtype=T.dtype
                )
            trace = torch.sum(T * delta, dim=tuple(idxs), keepdim=True)
            corr += delta * trace
        result = result + coeff * corr
    return result

# numerical
def symmetric_traceless_outer_product(v: Tensor, n: int, norm: bool = True) -> Tensor:
    T = symmetric_outer_product(v, n, norm)
    return subtract_traces(T, n)


class LegacyCartesianHarmonics1(torch.nn.Module):
    def __init__(self, lmax: int, norm: bool = True, traceless: bool = True) -> None:
        super().__init__()
        self.lmax = lmax
        self.norm = norm
        self.traceless = traceless
        PS, DS, CS, SS = ICTD(lmax)
        self.register_buffer(f"Q", DS[0].to(torch.get_default_dtype()))
        del PS, DS, CS, SS

    def forward(self, v: Tensor) -> Tensor:
        T = symmetric_outer_product(v, self.lmax, self.norm)
        if self.traceless:
            B = T.size(0)
            if B == 0:
                return T 
            else:
                REST = T.size()[1:]
                T = T.reshape(B, -1)
                T = T @ self.Q
                T = T.reshape((B,) + REST)
                return T
        else:
            return T

    def __repr__(self):
        return f"{self.__class__.__name__}(r={self.lmax}, norm={self.norm})"
    

class LegacyCartesianHarmonics2(torch.nn.Module):
    def __init__(self, lmax: int, norm: bool = True, traceless: bool = True) -> None:
        super().__init__()
        self.lmax = lmax
        self.norm = norm
        self.traceless = traceless
        for l in range(self.lmax+1):
            PS, DS, CS, SS = ICTD(l)
            self.register_buffer(f"D{l}", DS[0].to(torch.get_default_dtype()))
            del PS, DS, CS, SS

    def forward(self, v: Tensor) -> Dict[int, Tensor]:
        T = torch.ones_like(v[..., 0])
        edge_attrs: Dict[int, Tensor] = {}
        edge_attrs[0] = T.unsqueeze(1)

        for l in range(1, self.lmax+1):
            T = T[..., None] * expand_dims_to(v, T.ndim + 1, dim=v.ndim - 1)
            edge_attrs[l] = T
                
        if self.traceless:
            for l in range(1, self.lmax+1):
                T = edge_attrs[l]
                B = T.size(0)
                if B != 0:
                    REST = T.size()[1:]
                    T = T.reshape(B, -1)
                    T = T @ self.D(l)
                    T = T.reshape((B,) + REST)
                edge_attrs[l] = T

        if self.norm:
            for l in range(1, self.lmax+1):
                edge_attrs[l] = edge_attrs[l] * _norm(l)
                
        for l in range(1, self.lmax+1):
            edge_attrs[l] = edge_attrs[l].unsqueeze(1)

        return edge_attrs

    def D(self, l: int):
        return dict(self.named_buffers())[f"D{l}"]
    
    def __repr__(self):
        return f"{self.__class__.__name__}(r={self.lmax}, norm={self.norm}, traceless={self.traceless})"
    
