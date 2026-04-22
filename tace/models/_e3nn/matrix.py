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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.C})"
    

# class MatrixTensorProduct(torch.nn.Module):
#     """
#     Based on https://github.com/google-research/e3x/blob/main/e3x/nn/modules.py
#     """
#     def __init__(
#         self,
#         L1: int,
#         L2: int,
#         C: int,
#         L3: int | None = None,
#     ):
#         super().__init__()

#         self.L1 = L1
#         self.L2 = L2
#         self.L3 = max(L1, L2) if L3 is None else L3
#         self.C = C

#         self.maxL = max(self.L1, self.L2, self.L3)
#         self.Lmat = math.ceil(self.maxL / 2)

#         even_mask, odd_mask = self._build_mask(self.maxL)
#         self.register_buffer("even_mask", even_mask, persistent=False)
#         self.register_buffer("odd_mask", odd_mask, persistent=False)

#         self.register_buffer("cg", self._build_cg().to(torch.get_default_dtype()))

#         num_mat = 2 * self.Lmat + 1
#         self.var_in = (1.0 / math.sqrt(num_mat)) * (
#             num_mat / min(self.L1 + 1, self.L2 + 1)
#         )

#         var_out = torch.full((self.L3 + 1, 1), 2.0)
#         var_out[0] = 1.0
#         self.register_buffer("var_out", var_out, persistent=False)

#         self.kernel_e1 = torch.nn.Parameter(self._init_kernel(self.L1, self.var_in))
#         self.kernel_o1 = torch.nn.Parameter(self._init_kernel(self.L1, self.var_in))

#         self.kernel_e2 = torch.nn.Parameter(self._init_kernel(self.L2, self.var_in))
#         self.kernel_o2 = torch.nn.Parameter(self._init_kernel(self.L2, self.var_in))

#         self.kernel_eee = torch.nn.Parameter(self._init_kernel(self.L3, self.var_out))
#         self.kernel_ooe = torch.nn.Parameter(self._init_kernel(self.L3, self.var_out))
#         self.kernel_eoo = torch.nn.Parameter(self._init_kernel(self.L3, self.var_out))
#         self.kernel_oeo = torch.nn.Parameter(self._init_kernel(self.L3, self.var_out))


#     def _init_kernel(self, L, scale):
#         """(L+1, C)"""
#         w = torch.randn(L + 1, self.C)
#         if isinstance(scale, torch.Tensor):
#             w = w * scale
#         else:
#             w = w * scale
#         return w


#     def _build_cg(self) -> torch.Tensor:
#         from e3nn import o3

#         cg = torch.zeros(
#             (self.Lmat + 1) ** 2,
#             (self.Lmat + 1) ** 2,
#             (self.maxL + 1) ** 2,
#         )

#         for l1 in range(self.Lmat + 1):
#             for l2 in range(self.Lmat + 1):
#                 for l3 in range(abs(l1 - l2), min(l1 + l2, self.maxL) + 1):
#                     cg[
#                         l1**2:(l1 + 1) ** 2,
#                         l2**2:(l2 + 1) ** 2,
#                         l3**2:(l3 + 1) ** 2,
#                     ] = o3.wigner_3j(l1, l2, l3) * math.sqrt(2 * l3 + 1)

#         i = self.Lmat**2
#         j = (self.Lmat + 1) ** 2
#         return cg[i:j, i:j, :]

#     def _build_mask(self, L: int):
#         degrees = torch.arange(L + 1)
#         repeats = 2 * degrees + 1

#         even = (degrees + 1) % 2
#         odd = degrees % 2

#         total = (L + 1) ** 2

#         even_mask = torch.repeat_interleave(even, repeats)[:total]
#         odd_mask = torch.repeat_interleave(odd, repeats)[:total]

#         return even_mask.view(1, -1, 1), odd_mask.view(1, -1, 1)


#     def _expand_kernel(self, kernel, L):
#         repeats = torch.arange(
#             L + 1,
#             device=kernel.device,
#             dtype=torch.long
#         )
#         repeats = 2 * repeats + 1
#         return torch.repeat_interleave(kernel, repeats, dim=0)


#     def _split_parity(self, x, L):
#         l = (L + 1) ** 2
#         return (
#             x * self.even_mask[:, :l, :],
#             x * self.odd_mask[:, :l, :],
#         )

#     def _to_matrix(self, x, kernel, L):
#         return torch.einsum(
#             "...nf,nf,lmn->...lmf",
#             x,
#             kernel,
#             self.cg[..., :L],
#         )

#     def _to_vector(self, x, kernel, L):
#         return torch.einsum(
#             "...lmf,nf,lmn->...nf",
#             x,
#             kernel,
#             self.cg[..., :L],
#         )

#     def _couple(self, x, y):
#         return torch.einsum("...lmf,...mnf->...lnf", x, y)

#     def forward(self, x: torch.Tensor, y: torch.Tensor):

#         l1 = (self.L1 + 1) ** 2
#         l2 = (self.L2 + 1) ** 2
#         l3 = (self.L3 + 1) ** 2

#         k_e1 = self._expand_kernel(self.kernel_e1, self.L1)
#         k_o1 = self._expand_kernel(self.kernel_o1, self.L1)
#         k_e2 = self._expand_kernel(self.kernel_e2, self.L2)
#         k_o2 = self._expand_kernel(self.kernel_o2, self.L2)

#         k_eee = self._expand_kernel(self.kernel_eee, self.L3)
#         k_ooe = self._expand_kernel(self.kernel_ooe, self.L3)
#         k_eoo = self._expand_kernel(self.kernel_eoo, self.L3)
#         k_oeo = self._expand_kernel(self.kernel_oeo, self.L3)

#         e1, o1 = self._split_parity(x, self.L1)
#         e2, o2 = self._split_parity(y, self.L2)

#         e1 = self._to_matrix(e1, k_e1, l1)
#         o1 = self._to_matrix(o1, k_o1, l1)
#         e2 = self._to_matrix(e2, k_e2, l2)
#         o2 = self._to_matrix(o2, k_o2, l2)

#         eee = self._couple(e1, e2)
#         ooe = self._couple(o1, o2)
#         eoo = self._couple(e1, o2)
#         oeo = self._couple(o1, e2)

#         eee = self._to_vector(eee, k_eee, l3)
#         ooe = self._to_vector(ooe, k_ooe, l3)
#         eoo = self._to_vector(eoo, k_eoo, l3)
#         oeo = self._to_vector(oeo, k_oeo, l3)

#         e3 = eee + ooe
#         o3 = eoo + oeo

#         return e3 * self.even_mask[:, :l3, :] + o3 * self.odd_mask[:, :l3, :]
    

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__} ({self.C})"