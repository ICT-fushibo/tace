################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from math import sqrt
from typing import List


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


class LinearLayer(torch.nn.Module):
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
        act: str | None | torch.nn.Module= "silu",
        forward_weight_init: bool = True,
        layer_norm: bool = False,
        rms_norm: bool = False,
        # parametrization: str | None = None, # ["spectral_norm", "weight_norm", "orthogonal"]
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


            linear_layer = LinearLayer(
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
        act: str | torch.nn.Module = "sigmoid",
        alpha: float = 1.0,
        norm1: torch.nn.Module | None = None,
        norm2: torch.nn.Module | None = None,
    ):
        super().__init__()

        linear_v = []
        linear_v.append(
            LinearLayer(
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
            LinearLayer(
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
        act: str | None | torch.nn.Module = "sigmoid",
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
                    LinearLayer(
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

