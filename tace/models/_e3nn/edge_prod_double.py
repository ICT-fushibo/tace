# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Union


import torch


from ..so2 import SO2TensorProduct, SO2ComplexMul



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

        self.mul = SO2ComplexMul(self.mmax, self.lmax, self.num_channel, channel_wise=True)

        self.ace = SO2TensorProduct(
            mmax, 
            lmax,
            num_channels, 
            m1m2=m1m2, 
            internal_weights=agnostic
        )
        self.weight_numel = self.ace.weight_numel

        if agnostic:

            self.source_coefs = torch.nn.ParameterList()
            self.target_coefs = torch.nn.ParameterList()

            # nu = 2
            self.source_coefs.append(torch.nn.Parameter(torch.randn(self.ace.weight_numel)))
            self.target_coefs.append(torch.nn.Parameter(torch.randn(self.ace.weight_numel)))

            # self.source_coefs.data.mul_(1 / math.sqrt(2))
            # self.target_coefs.data.mul_(1 / math.sqrt(2))

        else:
            self.source_coefs = torch.nn.ParameterList()
            self.target_coefs = torch.nn.ParameterList()

            # nu = 1
            self.source_coefs.append(torch.nn.Parameter(torch.randn(num_elements, (mmax+1) * (lmax+1) * num_channels)))
            self.target_coefs.append(torch.nn.Parameter(torch.randn(num_elements, (mmax+1) * (lmax+1) * num_channels)))
          
            # nu = 2
            self.source_coefs.append(torch.nn.Parameter(torch.randn(num_elements, self.ace.weight_numel)))
            self.target_coefs.append(torch.nn.Parameter(torch.randn(num_elements, self.ace.weight_numel)))

            # self.source_coefs.data.mul_(1 / math.sqrt(2))
            # self.target_coefs.data.mul_(1 / math.sqrt(2))

    def forward(self, x, y, edge_index) -> torch.Tensor:

        if self.agnostic:
            return x + self.ace(x, x, torch.stack([self.source_coefs[0], self.target_coefs[0]], dim=0).unsqueeze(0))

        node_type = y.argmax(dim=-1)
        src_type = node_type[edge_index[0]]
        dst_type = node_type[edge_index[1]]


        nu1_coefs = torch.stack(
            [
                self.source_coefs[0][src_type].view(-1, (self.mmax+1)*(self.lmax+1), self.num_channel),
                self.target_coefs[0][dst_type].view(-1, (self.mmax+1)*(self.lmax+1), self.num_channel),
            ], 
            dim=1,
        )
        nu1_coefs = nu1_coefs.view(-1, 2*(self.mmax+1)*(self.lmax+1), self.num_channel)

        nu2_coefs = torch.stack(
            [
                self.source_coefs[1][src_type],
                self.target_coefs[1][dst_type],
            ], 
            dim=1,
        )

        return self.mul.complex_mul(nu1_coefs, x) + self.ace(x, x, nu2_coefs)


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
    