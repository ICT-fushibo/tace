################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from math import sqrt
from typing import List, Union

import torch
import torch.nn.functional as F

from .linear import mlpLinear


class ScaledSiLU(torch.nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.scale_factor = 1.6791767923989418  # scale from e3nn Activation

    def forward(self, inputs):
        return F.silu(inputs, inplace=self.inplace) * self.scale_factor


class ScaledSigmoid(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1.8467055342154763  # scale from e3nn Activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * self.scale_factor


class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2


ACTIVATION = {
    None: torch.nn.Identity,
    "none": torch.nn.Identity,
    "None": torch.nn.Identity,
    "null": torch.nn.Identity,
    "identity": torch.nn.Identity,
    "relu": torch.nn.ReLU,
    "leaky_relu": torch.nn.LeakyReLU,
    "smooth_leaky_relu": SmoothLeakyReLU,
    "prelu": torch.nn.PReLU,
    "elu": torch.nn.ELU,
    "selu": torch.nn.SELU,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,  
    "scaled_silu": ScaledSiLU,  
    "mish": torch.nn.Mish,
    "softplus": torch.nn.Softplus,
    "softsign": torch.nn.Softsign,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "scaled_sigmoid": ScaledSigmoid,  
    "hardtanh": torch.nn.Hardtanh,
    "hardswish": torch.nn.Hardswish,
    "hardsigmoid": torch.nn.Hardsigmoid,
    "tanhshrink": torch.nn.Tanhshrink,
}


class MLP(torch.nn.Module):
    def __init__(
        self,
        channels: List[int],
        bias: bool = False,
        act: Union[str, torch.nn.Module, None] = "silu",
        forward_weight_init: bool = True,
        layer_norm: bool = False,
        rms_norm: bool = False,
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
                gain = 1.0 if act is None or (layer == self.num_layers - 1) else sqrt(2)

            linear_layer = mlpLinear(
                in_dim=h_in,
                out_dim=h_out,
                alpha=gain / sqrt(norm_dim),
                bias=bias,
            )

            mlp.append(linear_layer)

            if layer < len(self.dims) - 2:
                if layer_norm:
                    mlp.append(torch.nn.LayerNorm(h_out))
                elif rms_norm:
                    mlp.append(torch.nn.RMSNorm(h_out))

            del gain, norm_dim

            if (layer != self.num_layers - 1) and (act is not None):
                if isinstance(act, torch.nn.Module):
                    mlp.append(act)
                else:
                    mlp.append(ACTIVATION[act]())
                self.is_nonlinear = True

        self.mlp = torch.nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
