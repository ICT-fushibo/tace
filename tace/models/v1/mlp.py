################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from math import sqrt
from typing import Optional, List


import torch
from torch import nn


from .act import ACT


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: List[int] = [],
        act: Optional[str] = "silu",
        bias: bool = False,
        forward_weight_init: bool = True,
        enable_layer_norm: bool = False,
    ):
        '''The parameter initialization method uses the earlier version of Allegro'''
        super().__init__()
        self.bias = bias
        self.dims = [in_dim] + hidden_dim + [out_dim]
        self.num_layers = len(self.dims) - 1
        assert self.num_layers >= 1
        act = ACT[act]()
        self.is_nonlinear = False
        self.enable_layer_norm = enable_layer_norm

        # === build the MLP + weight init ===
        mlp = []
        for layer, (h_in, h_out) in enumerate(zip(self.dims, self.dims[1:])):

            # === weight initialization ===
            if forward_weight_init:
                norm_dim = h_in
                gain = 1.0 if act is None or (layer == 0) else sqrt(2)
            else:
                norm_dim = h_out
                gain = (
                    1.0 if act is None or (layer == self.num_layers - 1) else sqrt(2)
                )

            # === instantiate Linear ===
            linear_layer = LinearLayer(
                in_dim=h_in,
                out_dim=h_out,
                alpha=gain / sqrt(norm_dim),
                bias=bias,
            )
            mlp.append(linear_layer)

            # === optional LayerNorm ===
            if enable_layer_norm:
                if layer < len(self.dims) -2:
                    mlp.append(nn.LayerNorm(h_out))
            del gain, norm_dim

            # === act ===
            if (layer != self.num_layers - 1) and (act is not None):
                mlp.append(act)
                self.is_nonlinear = True

        self.mlp = torch.nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)


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
        self._bias = bias
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
        return f"{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim} bias={ self._bias})"


class LoRALinearLayer(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        alpha: float = 1.0,
        bias: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 8.0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.use_lora = True
        self.lora_alpha = lora_alpha
        self.lora_r = lora_r
        self.scaling = lora_alpha / lora_r

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.weight = torch.nn.Parameter(torch.empty((in_dim, out_dim)))
        torch.nn.init.uniform_(self.weight, -sqrt(3), sqrt(3))
        self.weight.requires_grad_(False)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_dim))
            self.bias.requires_grad_(False)
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(torch.zeros(in_dim, lora_r))
        self.lora_B = nn.Parameter(torch.zeros(lora_r, out_dim))
        nn.init.kaiming_uniform_(self.lora_A, a=sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.use_lora:
            delta_w = self.lora_A @ (self.lora_B * self.scaling)
            W = self.weight + delta_w
        else:
            W = self.weight

        W = W * self.alpha

        if self.bias is None:
            return torch.mm(input, W)
        else:
            return torch.addmm(self.bias, input, W)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  in_dim      = {self.in_dim},\n"
            f"  out_dim     = {self.out_dim},\n"
            f"  bias        = {self.bias is not None},\n"
            f"  alpha       = {self.alpha:.2f},\n"
            f"  lora_r      = {self.lora_r},\n"
            f"  lora_alpha  = {self.lora_alpha},\n"
            f")"
        )
