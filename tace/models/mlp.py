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
        self.scale_factor = 1.6791767923989418 # scale from e3nn Activation

    def forward(self, inputs):
        return F.silu(inputs, inplace=self.inplace) * self.scale_factor

    def extra_repr(self):
        str = f"scale_factor={self.scale_factor}"
        if self.inplace:
            str = str + ", inplace=True"
        return str
    
class ScaledSigmoid(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1.8467055342154763 # scale from e3nn Activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * self.scale_factor
    
    
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

class MLP(torch.nn.Module):
    def __init__(
        self,
        channels: List[int],
        bias: bool = False,
        act: Union[str, torch.nn.Module, None] = "silu",
        forward_weight_init: bool = True,
        layer_norm: bool = False,
        rms_norm: bool = False,
        # parametrization: Union[str, None] = None, # ["spectral_norm", "weight_norm", "orthogonal"]
    ):
        """Based on https://github.com/mir-group/nequip/blob/main/nequip/nn/mlp.py"""
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


            linear_layer = mlpLinear(
                in_dim=h_in,
                out_dim=h_out,
                alpha=gain / sqrt(norm_dim),
                bias=bias,
            )

            # if parametrization == "spectral_norm":
            #     torch.nn.utils.parametrizations.spectral_norm(
            #         linear_layer, "weight", dim=1
            #     )
            # elif parametrization == "weight_norm":
            #     torch.nn.utils.parametrizations.weight_norm(
            #         linear_layer, "weight", dim=1
            #     )
            # elif parametrization == "orthogonal":
            #     torch.nn.utils.parametrizations.orthogonal(linear_layer, "weight")
            # elif parametrization not in [None, "None", "null"]:
            #     raise ValueError(
            #         f"Unknown parametrization '{parametrization}'. "
            #         "Available options: None, 'weight_norm', 'orthogonal', 'spectral_norm'"
            #     )
            
            mlp.append(linear_layer)

            if layer < len(self.dims) -2:
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


class GLULayer(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        act: Union[str, torch.nn.Module] = "sigmoid",
        alpha: float = 1.0,
        norm1: Union[torch.nn.Module, None] = None,
        norm2: Union[torch.nn.Module, None] = None,
    ):
        super().__init__()

        linear_v = []
        linear_v.append(
            mlpLinear(
                in_dim=in_dim,
                out_dim=out_dim,
                alpha=alpha,
                bias=bias,
            )
        )
        if norm1 is not None:
            linear_v.append(norm1)
        self.linear_v = torch.nn.Sequential(*linear_v)

        linear_g = []
        linear_g.append(
            mlpLinear(
                in_dim=in_dim,
                out_dim=out_dim,
                alpha=alpha,
                bias=bias,
            )
        )
        if norm2 is not None:
            linear_g.append(norm2)
        self.linear_g = torch.nn.Sequential(*linear_g)

        if isinstance(act, torch.nn.Module):
            self.act = act
        else:
            self.act = ACTIVATION[act]()

    def forward(self, x):
        v = self.linear_v(x)
        g = self.act(self.linear_g(x))
        return v * g


class GLU(torch.nn.Module):
    def __init__(
        self,
        channels: List[int],
        bias: bool = False,
        act: Union[str, torch.nn.Module, None] = "sigmoid",
        forward_weight_init: bool = True,
        layer_norm: bool = False,
        rms_norm: bool = False,
    ):
        super().__init__()

        if len(channels) < 2:
            raise ValueError("GLUMLP must have at least 2 layers")

        self.num_layers = len(channels) - 1
        self.dims = channels

        layers = []

        for layer, (h_in, h_out) in enumerate(zip(self.dims, self.dims[1:])):

            if forward_weight_init:
                norm_dim = h_in
                gain = 1.0 if layer == 0 else sqrt(2)
            else:
                norm_dim = h_out
                gain = 1.0 if layer == self.num_layers - 1 else sqrt(2)


            if layer < len(self.dims) -2:
                if layer_norm:
                    norm1 = torch.nn.LayerNorm(h_out)
                    norm2 = torch.nn.LayerNorm(h_out)
                elif rms_norm:
                    norm1 = torch.nn.RMSNorm(h_out)
                    norm2 = torch.nn.RMSNorm(h_out)
            else:
                norm1 = None
                norm2 = None

            if layer == len(self.dims) -2:
                layers.append(
                    mlpLinear(
                        in_dim=h_in,
                        out_dim=h_out,
                        alpha=gain / sqrt(norm_dim),
                        bias=bias,
                    )
                )
            else:
                layers.append(
                    GLULayer(
                        in_dim=h_in,
                        out_dim=h_out,
                        alpha=gain / sqrt(norm_dim),
                        bias=bias,
                        act=ACTIVATION[act](),
                        norm1=norm1,
                        norm2=norm2,
                    )
                )

        self.glu = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.glu(x)
    
FFN = {
    'mlp': MLP,
    'glu': GLU,
}

