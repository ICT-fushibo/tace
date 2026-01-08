################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
# Description: This file contains linear readout function for tensor rank > 0
################################################################################

from math import sqrt
from typing import List, Optional


import torch


from .act import ACT
from .linear import Linear
from .mlp import LinearLayer
from .gate import NormGate
from .utils import select_corresponding_level


class ScalarReadOut(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: List[int] = [],
        act: Optional[str] = "silu",
        bias: bool = False,
        forward_weight_init: bool = True,
        enable_layer_norm: bool = False,
        num_levels: int = 1,
        use_multi_head: bool = False,
    ):
        super().__init__()
        self.bias = bias
        self.dims = [in_dim] + hidden_dim + [out_dim]
        self.num_layers = len(self.dims) - 1
        assert self.num_layers >= 1
        act = ACT[act]()
        self.is_nonlinear = False
        self.enable_layer_norm = enable_layer_norm
        self.num_levels = num_levels
        self.use_multi_head = (use_multi_head) and (num_levels > 1) 

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
                    mlp.append(torch.nn.LayerNorm(h_out))
            del gain, norm_dim

            # === act ===
            if (layer != self.num_layers - 1) and (act is not None):
                mlp.append(act)
                self.is_nonlinear = True

        self.last_readout = True if len(hidden_dim) > 0 else False

        if self.use_multi_head: 
            if self.last_readout:
                self.mlp_1 = torch.nn.Sequential(*mlp[:-1]) 
                self.mlp_2 = torch.nn.Sequential(mlp[-1])   
            else:
                self.mlp = torch.nn.Sequential(mlp[-1])    
        else:
            self.mlp = torch.nn.Sequential(*mlp)

    def forward(self, x, node_level=None):
        if self.use_multi_head:
            if self.last_readout:
                x = self.mlp_1(x)
                x = select_corresponding_level(x, node_level, self.num_levels)
                x = self.mlp_2(x)
            else:
                x = self.mlp(x)
            return x
        else:
            return self.mlp(x)

class TensorReadOut(torch.nn.Module):
    def __init__(
        self,
        l: int,
        in_dim: int,
        hidden_dim: List[int],
        out_dim: int,
        gate: str = "silu",
        num_levels: int = 1,
        use_multi_head: bool = False,
    ) -> None:
        super().__init__()

        self.l = l
        self.num_levels = num_levels
        self.use_multi_head = (use_multi_head) and (num_levels > 1)

        self.linears_1 = torch.nn.ModuleList()
        prev_dim = in_dim
        for h_dim in hidden_dim:
            self.linears_1.append(
                Linear(
                    in_dim=prev_dim,
                    out_dim=h_dim,
                    bias=False,
                    l=self.l,
                )
            )
            prev_dim = h_dim

        self.gates = torch.nn.ModuleList()
        for dim in hidden_dim:
            self.gates.append(NormGate[gate](self.l, dim))
            
        self.linear_2= Linear(
            in_dim=prev_dim,
            out_dim=out_dim,
            bias=False,
            l=self.l,
        )

        self.last_readout = True if len(hidden_dim) > 0 else False

    def forward(self, t: torch.Tensor, node_level: torch.Tensor) -> torch.Tensor:
        if self.use_multi_head:
            if self.last_readout:
                for idx, linear in enumerate(self.linears_1):
                    t = self.gates[idx](linear(t))
                    t = select_corresponding_level(t, node_level, self.num_levels)
                return self.linear_2(t)
            else:
                for idx, linear in enumerate(self.linears_1):
                    t = self.gates[idx](linear(t))
                return self.linear_2(t)     
        else:
            for idx, linear in enumerate(self.linears_1):
                t = self.gates[idx](linear(t))
            return self.linear_2(t)
                    
def build_scalar_readout(
    in_dim: int,
    hidden_dim: List[int],
    act: int,
    bias: bool,
    num_levels: int,
    use_multi_head: bool,
    num_layers: int,
    use_all_layer: bool,
):
    
    if use_multi_head:
        assert len(hidden_dim) <= 1

    if use_multi_head:
        out_dim = num_levels
    else:
        out_dim = 1

    readouts = []
    for idx in range(num_layers):

        if idx == (num_layers-1):
            if use_multi_head:
                hidden_dim_ = [d * num_levels for d in hidden_dim]
            else:
                hidden_dim_ = hidden_dim
            act_ = act
            forward_weight_init = False
        else:
            hidden_dim_ = []
            act_ = None
            forward_weight_init = True

        readouts.append(
            ScalarReadOut(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=hidden_dim_,
                act=act_,
                bias=bias,
                forward_weight_init=forward_weight_init,
                num_levels=num_levels,
                use_multi_head=use_multi_head,
            )
        )
    if use_all_layer:
        return torch.nn.ModuleList(readouts)
    else:
        return torch.nn.ModuleList([readouts[-1]])


def build_tensor_readout(
    l: int,
    in_dim: int,
    hidden_dim: int,
    gate: int,
    num_levels: int,
    use_multi_head: bool,
    num_layers: int,
    use_all_layer: bool,
):

    if use_multi_head:
        assert len(hidden_dim) <= 1

    if use_multi_head:
        out_dim = num_levels
    else:
        out_dim = 1

    readouts = []
    for idx in range(num_layers):
        if idx == num_layers - 1:
            if use_multi_head:
                hidden_dim_ = [d * num_levels for d in hidden_dim]
            else:
                hidden_dim_ = hidden_dim
            gate_ = gate
        else:
            hidden_dim_ = []
            gate_ = None


        readouts.append(
            TensorReadOut(
                l=l,
                in_dim=in_dim,
                hidden_dim=hidden_dim_,
                out_dim=out_dim,
                gate=gate_,
                num_levels=num_levels,
                use_multi_head=use_multi_head,
            )
        ) 
    if use_all_layer:
        return torch.nn.ModuleList(readouts)
    else:
        return torch.nn.ModuleList([readouts[-1]])

