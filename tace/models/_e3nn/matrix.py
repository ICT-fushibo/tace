################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math

import torch
from e3nn import o3


def e3nn_cg(L1, L2, L3):
    cg = torch.zeros(
        (L1 + 1) ** 2, (L2 + 1) ** 2, (L3 + 1) ** 2, dtype=torch.float64, device="cpu"
    )
    for l1 in range(L1 + 1):
        for l2 in range(L2 + 1):
            for l3 in range(abs(l1 - l2), min(l1 + l2, L3) + 1):
                cg[
                    l1**2:(l1+1)**2,
                    l2**2:(l2+1)**2,
                    l3**2:(l3+1)**2,
                ] = o3.wigner_3j(l1, l2, l3, torch.float64, "cpu") * math.sqrt(2*l3+1)
    return cg


class MatrixTensorProduct(torch.nn.Module):
    """
    Based on https://github.com/google-research/e3x/blob/main/e3x/nn/modules.py
    """
    def __init__(
        self,
        L1: int,
        L2: int,
        C: int,
        L3: int | None = None,
    ):
        super().__init__()

        self.L1 = L1
        self.L2 = L2
        self.L3 = max(L1, L2) if L3 is None else L3
        self.C = C

        self.maxL = max(self.L1, self.L2, self.L3)
        self.Lmat = math.ceil(self.maxL / 2)

        even_mask, odd_mask = self._build_mask(self.maxL)
        self.register_buffer("even_mask", even_mask, persistent=False)
        self.register_buffer("odd_mask", odd_mask, persistent=False)
        self.register_buffer("cg", self._build_cg().to(torch.get_default_dtype()))
        self.alpha = (1.0 / math.sqrt(2*self.Lmat) + 1) 

    def _build_cg(self) -> torch.Tensor:
        cg = e3nn_cg(
            self.Lmat,
            self.Lmat,
            self.maxL,   
        )
        i = self.Lmat ** 2
        j = (self.Lmat + 1) ** 2
        return cg[i:j, i:j, :]

    def _build_mask(self, L: int):
        degrees = torch.arange(L + 1)
        repeats = 2 * degrees + 1
        even = (degrees + 1) % 2
        odd = degrees % 2
        total = (L + 1) ** 2
        even_mask = torch.repeat_interleave(even, repeats)[:total]
        odd_mask = torch.repeat_interleave(odd, repeats)[:total]
        return even_mask.view(1, -1, 1), odd_mask.view(1, -1, 1)

    def _split_parity(self, x, L):
        l = (L + 1) ** 2
        return (
            x * self.even_mask[:, :l, :],
            x * self.odd_mask[:, :l, :],
        )

    def _to_matrix(self, x, L):
        return torch.einsum(
            "...nf,lmn->...lmf",
            x,
            self.cg[..., :L],
        )

    def _to_vector(self, x, L):
        return torch.einsum(
            "...lmf,lmn->...nf",
            x,
            self.cg[..., :L],
        )

    def _couple(self, x, y):
        return torch.einsum("...lmf,...mnf->...lnf", x, y) * self.alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor):

        l1 = (self.L1 + 1) ** 2
        l2 = (self.L2 + 1) ** 2
        l3 = (self.L3 + 1) ** 2

        e1, o1 = self._split_parity(x, self.L1)
        e2, o2 = self._split_parity(y, self.L2)

        e1 = self._to_matrix(e1, l1)
        o1 = self._to_matrix(o1, l1)
        e2 = self._to_matrix(e2, l2)
        o2 = self._to_matrix(o2, l2)

        eee = self._couple(e1, e2)
        ooe = self._couple(o1, o2)
        eoo = self._couple(e1, o2)
        oeo = self._couple(o1, e2)

        eee = self._to_vector(eee, l3)
        ooe = self._to_vector(ooe, l3)
        eoo = self._to_vector(eoo, l3)
        oeo = self._to_vector(oeo, l3)

        e3 = eee + ooe
        o3 = eoo + oeo

        return e3 * self.even_mask[:, :l3, :] + o3 * self.odd_mask[:, :l3, :]