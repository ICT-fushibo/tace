################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from math import sqrt
from typing import List, Optional


import torch


ACTIVATION = {
    None: torch.nn.Identity,
    "none": torch.nn.Identity,
    "None": torch.nn.Identity,
    "null": torch.nn.Identity,
    "identity": torch.nn.Identity,
    "relu": torch.nn.ReLU,
    "leaky_relu": torch.nn.LeakyReLU,
    "prelu": torch.nn.PReLU,
    "elu": torch.nn.ELU,
    "selu": torch.nn.SELU,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,  # Swish
    "mish": torch.nn.Mish,
    "softplus": torch.nn.Softplus,
    "softsign": torch.nn.Softsign,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "hardtanh": torch.nn.Hardtanh,
    "hardswish": torch.nn.Hardswish,
    "hardsigmoid": torch.nn.Hardsigmoid,
    "tanhshrink": torch.nn.Tanhshrink,
}


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        alpha: float = 1.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.weight = torch.nn.Parameter(torch.empty((in_dim, out_dim)))
        torch.nn.init.uniform_(self.weight, -sqrt(3), sqrt(3))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.alpha
        if self.bias is None:
            return torch.mm(input, weight)
        else:
            return torch.addmm(self.bias, input, weight)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim} bias={ self.bias is not None})"
    
class MLP(torch.nn.Module):
    def __init__(
        self,
        channels: List[int],
        bias: bool = False,
        layer_norm: bool = False,
        act: Optional[str] | torch.nn.Module= "silu",
        forward_weight_init: bool = True,
    ):
        super().__init__()

        if len(channels) < 2:
            raise ValueError("MLP must have at least 2 layers")
        
        self.num_layers = len(channels) - 1
        self.dims = channels
        self.is_nonlinear = False
  
        mlp = []
        for layer, (h_in, h_out) in enumerate(zip(self.dims, self.dims[1:])):

            if forward_weight_init:
                norm_dim = h_in
                gain = 1.0 if act is None or (layer == 0) else sqrt(2)
            else:
                norm_dim = h_out
                gain = (
                    1.0 if act is None or (layer == self.num_layers - 1) else sqrt(2)
                )

            mlp.append(
                Linear(
                    in_dim=h_in,
                    out_dim=h_out,
                    alpha=gain / sqrt(norm_dim),
                    bias=bias,
                )
            )

            if layer_norm:
                if layer < len(self.dims) -2:
                    mlp.append(torch.nn.LayerNorm(h_out))
            del gain, norm_dim

            if (layer != self.num_layers - 1) and (act is not None):
                if isinstance(act, torch.nn.Module):
                    mlp.append(act)
                else:
                    mlp.append(ACTIVATION[act]())
                self.is_nonlinear = True

        self.mlp = torch.nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)


