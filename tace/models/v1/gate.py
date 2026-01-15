################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
"""
NormGate are Modified from HotPP.
https://gitlab.com/bigd4/hotpp

MIT License

Copyright (c) 2024 wang laosi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Dict
import torch
import torch.nn.functional as F


from .act import ACT
from .utils import expand_dims_to

class NormGateBase(torch.nn.Module):
    """
    Activate function for tensor inputs with shape [batch, channel] + [3,] * l
    """
    r: int
    def __init__(
        self,
        r: int,
        in_dim: int,
    ) -> None:
        super().__init__()

        self.r = r

        if self.r == 0:
            self.weight = self.register_parameter("weight", None)
            self.bias = self.register_parameter("bias", None)
        else:
            self.weight = torch.nn.Parameter(torch.ones(in_dim, requires_grad=True))
            self.bias = torch.nn.Parameter(torch.zeros(in_dim, requires_grad=True))

    def forward(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if self.r == 0:
            return self.activate(t)
        else:
            return self.tensor_activate(t, r=self.r)

    def activate(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} must have 'activate'!"
        )

    def tensor_activate(self, t: torch.Tensor, r: int) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} must have 'tensor_activate'!"
        )

class TensorNormSilu(NormGateBase):
    """
    silu(x) = x * sigmoid(x), gate = F.sigmoid(norm)
    """
    def activate(self, t: torch.Tensor) -> torch.Tensor:
        return F.silu(t)

    def tensor_activate(self, t: torch.Tensor, r: int) -> torch.Tensor:
        B, C = t.size(0), t.size(1)
        t_flat = t.view(B, C, -1)
        norm = self.weight * t_flat.pow(2).sum(dim=-1) + self.bias
        gate = torch.sigmoid(norm)
        return expand_dims_to(gate, 2 + r) * t
    
class TensorNormTanh(NormGateBase):

    def activate(self, t: torch.Tensor) -> torch.Tensor:
        return torch.tanh(t)

    def tensor_activate(self, t: torch.Tensor, r: int) -> torch.Tensor:
        B, C = t.size(0), t.size(1)
        t_flat = t.view(B, C, -1)
        norm = self.weight * t_flat.pow(2).sum(dim=-1) + self.bias
        gate = torch.tanh(norm) / torch.where(norm == 0, torch.ones_like(norm), norm)
        return expand_dims_to(gate, 2 + r) * t

class TensorNormRelu(NormGateBase):

    def activate(self, t: torch.Tensor) -> torch.Tensor:
        return F.relu(t)
    
    def tensor_activate(self, t: torch.Tensor, r: int) -> torch.Tensor:
        B, C = t.size(0), t.size(1)
        t_flat = t.view(B, C, -1)
        norm = self.weight * t_flat.pow(2).sum(dim=-1) + self.bias
        gate = torch.heaviside(norm, torch.zeros_like(norm))
        return expand_dims_to(gate, 2 + r) * t

class TensorNormJilu(NormGateBase):
    """
    Similar to TensorSilu, but use tanh(norm) as gate so the factor could be negative
    """
    def activate(self, t: torch.Tensor) -> torch.Tensor:
        return F.silu(t)

    def tensor_activate(self, t: torch.Tensor, r: int) -> torch.Tensor:
        B, C = t.size(0), t.size(1)
        t_flat = t.view(B, C, -1)
        norm = self.weight * t_flat.pow(2).sum(dim=-1) + self.bias
        gate = F.tanh(norm)
        return expand_dims_to(gate, 2 + r) * t
    
class TensorNormIdentity(torch.nn.Module):

    r: int

    def __init__(
        self,
        r: int,
        in_dim: int,
    ) -> None:
        super().__init__()

        self.r = r

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return t

NormGate: Dict[str | None, type[torch.nn.Module]] = {
    "silu": TensorNormSilu,
    "jilu": TensorNormJilu,
    "tanh": TensorNormTanh,
    # "relu": TensorNormRelu,
    "None": TensorNormIdentity,
    "null": TensorNormIdentity,
    None: TensorNormIdentity,
}

class GatedGate(torch.nn.Module):
    """
    shape [batch, channel] + [3,] * l
    """
    r: int
    gate: str
    act: torch.nn.Module

    def __init__(
        self,
        r: int,
        gate: str,
    ) -> None:
        super().__init__()

        self.r = r
        self.gate = gate
        self.act = ACT[gate]()

    def forward(self, gate: torch.Tensor, gated) -> torch.Tensor:
        if self.r == 0:
            return self.activate(gated)
        else:
            return self.tensor_activate(gate, gated)

    def activate(self, gated: torch.Tensor) -> torch.Tensor:
       return self.act(gated)

    def tensor_activate(self, gate: torch.Tensor, gated: int) -> torch.Tensor:
        gate = expand_dims_to(self.act(gate), self.r+2)
        return gate * gated
    

