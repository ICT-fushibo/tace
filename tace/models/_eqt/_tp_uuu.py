###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import List, Tuple


import torch
from e3nn import o3


from ..layout import LayoutTransform
from .equitorch.nn import TensorProduct


class e3nnEqtTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        num_channel: int,
        path: List[Tuple[int, int, int]],
    ):
        super().__init__()

        self.reshap1 = LayoutTransform(irreps_in1)
        self.reshap2 = LayoutTransform(irreps_in2)
        self.reshap3 = LayoutTransform(irreps_out)

        self.eqt_tp = TensorProduct(
            irreps_in1="+".join(str(ir) for _, ir in irreps_in1),
            irreps_in2="+".join(str(ir) for _, ir in irreps_in2),
            irreps_out="+".join(str(ir) for _, ir in irreps_out),
            channels_in1=num_channel,
            channels_in2=num_channel,
            channels_out=num_channel,
            internal_weights=True,
            feature_mode='uuu',
            path_norm=True,
            channel_norm=False,
            trainable=False,
            path=[(k, i, j) for (i, j, k, _, _) in path],
        )
        

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.reshap1(x)
        y = self.reshap2(y)
        return self.reshap3.inverse(self.eqt_tp(x, y))

