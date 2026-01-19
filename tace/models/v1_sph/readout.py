################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
# Description: This file contains linear readout function for tensor rank > 0
################################################################################

from typing import List, Optional, Callable


import torch
from e3nn import o3
from e3nn.nn import Activation


from tace.models.v1.act import ACT
from .acc import AccLinear


def mask_head(x: torch.Tensor, head: torch.Tensor, num_heads: int) -> torch.Tensor:
    mask = torch.zeros(x.shape[0], x.shape[1] // num_heads, num_heads, device=x.device)
    idx = torch.arange(mask.shape[0], device=x.device)
    mask[idx, :, head] = 1
    mask = mask.permute(0, 2, 1).reshape(x.shape)
    return x * mask


class LinearReadout(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
    ):
        super().__init__()
        self.linear = AccLinear(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(
        self,
        x: torch.Tensor,
        node_level: Optional[torch.Tensor] = None,  
    ) -> torch.Tensor:  
        return self.linear(x) 
    
class NonLinearReadout(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_hidden: o3.Irreps,
        irreps_out: o3.Irreps,
        act: str | None,
        num_levels: int = 1,
    ):
        super().__init__()
        self.irreps_hidden = irreps_hidden
        self.num_levels = num_levels
        self.linear_1 = AccLinear(
            irreps_in=irreps_in, 
            irreps_out=irreps_hidden, 
        )
        self.nonlinearity = Activation(
            irreps_in=self.irreps_hidden, acts=[ACT[act]()]
        )
        self.linear_2 = AccLinear(
            irreps_in=irreps_hidden, 
            irreps_out=irreps_out, 
        )

    def forward(
        self, x: torch.Tensor, node_level: Optional[torch.Tensor] = None
    ) -> torch.Tensor: 
        x = self.nonlinearity(self.linear_1(x))
        if self.num_levels > 1 and node_level is not None:
            x = mask_head(x, node_level, self.num_levels)
        return self.linear_2(x) 

                    
def build_scalar_readout(
    irreps_in: List[o3.Irreps],
    irreps_hidden: o3.Irreps,
    act: int,
    bias: bool,
    num_levels: int,
    use_multi_head: bool,
    num_layers: int,
    use_all_layer: bool,
):
    irreps_out = o3.Irreps([(num_levels, (0, 1))])
    
    readouts = torch.nn.ModuleList()
    for layer in range(num_layers):
        if layer != num_layers - 1:
            readouts.append(
                LinearReadout(
                    irreps_in=irreps_in[layer],
                    irreps_out=irreps_out,
                )
            )
        else:
            readouts.append(
                NonLinearReadout(
                    irreps_in=irreps_in[layer],
                    irreps_hidden=irreps_hidden,
                    irreps_out=irreps_out,
                    act=act,
                    num_levels=num_levels,
                )
            )
    if use_all_layer:
        return torch.nn.ModuleList(readouts)
    else:
        return torch.nn.ModuleList([readouts[-1]])

