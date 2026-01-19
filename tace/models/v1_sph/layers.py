################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional


import torch
from e3nn import o3
from .acc import AccLinear


class LinearNodeEmbedding(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
    ):
        super().__init__()
        self.linear = AccLinear(
            irreps_in=irreps_in, 
            irreps_out=irreps_out, 
        )

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor: 
        return self.linear(node_feats)

class NonLinearNodeEmbedding(torch.nn.Module):
    pass
