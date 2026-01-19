################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import string
from math import sqrt
from typing import Dict, List, Optional


import torch
from e3nn import o3
from .kernel import CuEquivarianceConfig, OpenEquivarianceConfig


class WrapElementLinear(torch.nn.Module): # TODO, wrapper
    
    def __init__(
        self,
        num_elements: int,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        bias: bool = False,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OpenEquivarianceConfig] = None,
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.linear = o3.Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            shared_weights = False,
            internal_weights = False,
        )        
        self.weight = torch.nn.Parameter(
            torch.randn(num_elements, self.linear.weight_numel)
        )
        # torch.nn.init.uniform_(self.weights, -sqrt(3), sqrt(3))
        if bias :
            self.bias = torch.nn.Parameter(torch.empty(num_elements, irreps_out.num_irreps))
            torch.nn.init.zeros_(self.bias)
        else:
            # self.register_parameter("bias", torch.Tensor())
            self.register_parameter("bias", None)

 
    def forward(self, t: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
        # a = Cc
        W = torch.einsum("bz, za -> ba", onehot, self.weight)

        if self.bias is not None:
            b = torch.einsum('bz, zC -> bC', onehot, self.bias)
        else:
            b = None
        return self.linear(t, W, b)

    # def __repr__(self):
    #     return (f"{self.__class__.__name__}(irreps_in={self.irreps_in}, "
    #                 f"irreps_out={self.irreps_out}, bias={self.bias is not None})")

