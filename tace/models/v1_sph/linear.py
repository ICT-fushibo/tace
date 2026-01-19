################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch
from e3nn import o3
from .acc import AccLinear


class AccElementLinear(torch.nn.Module):
    """Not allow bias for cueq"""
    def __init__(
        self,
        num_elements: int,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.linear = AccLinear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            shared_weights = False,
            internal_weights = False,
        )        
        self.weight = torch.nn.Parameter(
            torch.randn(num_elements, self.linear.weight_numel)
        )

    def forward(self, x: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
        # a = Cc
        W = torch.einsum("bz, za -> ba", onehot, self.weight)
        return self.linear(x, W)
