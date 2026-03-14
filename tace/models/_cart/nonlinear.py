################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, List


import torch


from ..utils import expand_dims_to
from ..activation import ACTIVATION
from .linear import Linear


class NormNonlinear(torch.nn.Module):
    """
    Activate function for tensor inputs with shape [batch, channel] + [3,] * l
    """
    def __init__(self, ls: int, num_channel:int, act: torch.nn.Module, bias: bool) -> None:
        super().__init__()
        self.ls = ls
        self.num_channel = num_channel
        self.act = act
        self.weight = torch.nn.Parameter(torch.ones(len(ls), num_channel))
        self.bias = torch.nn.Parameter(torch.zeros(len(ls), num_channel))
        self.linear = Linear(ls, num_channel, num_channel, bias=bias)

    def forward(self, xs: Dict[int, torch.Tensor]) -> torch.Tensor:
        for idx, l in enumerate(self.ls):
            if l == 0:
                norm = xs[l]
            else:
                B = xs[l].size(0)
                C = xs[l].size(1)
                norm = torch.linalg.norm(xs[l].view(B, C, -1), dim=-1, ord=2)
            norm = norm * self.weight[idx].unsqueeze(0) + self.bias[idx].unsqueeze(0)
            xs[l] = expand_dims_to(self.act(norm), 2 + l) * xs[l]
        return xs

class GateNonlinear(torch.nn.Module):
    def __init__(self, ls: int, num_channel:int, act: torch.nn.Module) -> None:
        super().__init__()
        self.ls = ls
        self.num_channel = num_channel
        self.act = act

    def forward(self, xs: Dict[int, torch.Tensor], gate: List[torch.Tensor]) -> torch.Tensor:
        for idx, l in enumerate(self.ls):
            xs[l] = expand_dims_to(self.act(gate[idx]), 2 + l) * xs[l]
        return xs
    



