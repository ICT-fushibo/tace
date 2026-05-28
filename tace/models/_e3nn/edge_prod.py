# TODO, refactor
################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Union


import torch


from ..so2 import LegacyuuSO2TensorProduct, uuSO2TensorProduct


class SO2EdgeProductBasis(torch.nn.Module):

    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channels: int,
        num_elements: int,
        m1m2: Union[str, None] = '<=',
        agnostic: bool = True,
    ):
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_components = lmax+1
        self.num_channel = num_channels
        self.agnostic = agnostic

        self.ace = LegacyuuSO2TensorProduct(
            mmax, 
            lmax,
            num_channels, 
            m1m2=m1m2, 
            internal_weights=agnostic,
        )
        self.weight_numel = self.ace.weight_numel

        if not agnostic:
            self.num_c1_weight = (mmax+1) * (lmax+1) * num_channels
            self.weight_numel += self.num_c1_weight
            self.source_coefs = torch.nn.Parameter(
                torch.randn(num_elements, self.weight_numel)
            )
            self.target_coefs = torch.nn.Parameter(
                torch.randn(num_elements, self.weight_numel)
            )
            self.source_coefs.data.mul_(1 / math.sqrt(2))
            self.target_coefs.data.mul_(1 / math.sqrt(2))

            expand_index = []
            offset = 0
            for m in range(mmax + 1):
                index = torch.arange((lmax + 1))
                index = index + offset
                expand_index.append(index)
                if m > 0:
                    expand_index.append(index)    # +- m
                offset = offset + len(index)
            expand_index = torch.cat(expand_index, dim=0)
            expand_index = expand_index.long()
            self.num_components = offset
            self.register_buffer('expand_index', expand_index, persistent=False)
   
    def forward(
            self, 
            x, 
            node_attes, 
            edge_index
        ) -> torch.Tensor:

        if self.agnostic:
            return x + self.ace(x, x)

        B = x.size(0)
        C = self.num_channel

        node_type = node_attes.argmax(dim=-1)
        src_type = node_type[edge_index[0]]
        dst_type = node_type[edge_index[1]]
        source_coefs = self.source_coefs[src_type]
        target_coefs = self.target_coefs[dst_type]
        
        # nu = 1
        w1 = (
            source_coefs[:, :self.num_c1_weight]
            + target_coefs[:, :self.num_c1_weight]
        )
        w1 = w1.view(B, -1, C)
        w1 = torch.index_select(w1, dim=1, index=self.expand_index)
        corr_feats1 = x * w1

        # nu = 2
        w2 = (
            source_coefs[:, self.num_c1_weight:]
            + target_coefs[:, self.num_c1_weight:]
        )
        corr_feats2 = self.ace(x, x, w2)

        return corr_feats1 + corr_feats2


    def extra_repr(self) -> str:
        p = {
            0: 'e',
            1: 'o',
        }
        irreps = []
        for m in range(self.mmax + 1):
            irreps.append(f"{self.num_channel*(self.lmax+1)}x{m}{p[m % 2]}")
        num_weights = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return (
            f"{self.__class__.__name__}"
            f"({'+'.join(irreps)} x {'+'.join(irreps)} -> "
            f"{'+'.join(irreps)} | "
            f"{num_weights} weights)"
        )


# class SO2EdgeProductBasis(torch.nn.Module):

#     def __init__(
#         self,
#         mmax: int,
#         lmax: int,
#         num_channels: int,
#         num_elements: int,
#         m1m2: Union[str, None] = '<=',
#         agnostic: bool = True,
#         tp_version: Union[str, None] = "v2",
#         use_triton: Union[bool, None] = False,
#         weight_type: str = "w1_w2",
#     ):
#         super().__init__()

#         self.mmax = mmax
#         self.lmax = lmax
#         self.num_components = lmax+1
#         self.num_channel = num_channels
#         self.agnostic = agnostic
#         self.tp_version = tp_version
#         self.use_triton = use_triton
#         self.weight_type = weight_type

#         self.ace = uuSO2TensorProduct(
#             mmax,
#             lmax,
#             num_channels,
#             m1m2=m1m2,
#             internal_weights=agnostic,
#             use_triton=use_triton,
#             weight_type=weight_type,
#         )

#         self.weight_numel = self.ace.weight_numel


#     def forward(
#             self, 
#             x, 
#         ) -> torch.Tensor:

#         return x + self.ace(x, x)
