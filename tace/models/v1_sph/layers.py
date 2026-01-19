################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional


import torch
from e3nn import o3
from .kernel import CuEquivarianceConfig, OpenEquivarianceConfig


class LinearNodeEmbedding(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OpenEquivarianceConfig] = None,
    ):
        super().__init__()
        self.linear = o3.Linear(
            irreps_in=irreps_in, 
            irreps_out=irreps_out, 
            # cueq_config=cueq_config,
            # oeq_config=oeq_config,
        )

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor: 
        return self.linear(node_feats)

class NonLinearNodeEmbedding(torch.nn.Module):
    pass
