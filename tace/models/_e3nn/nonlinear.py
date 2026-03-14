################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import torch
from e3nn import o3
from e3nn.o3._tensor_product._sub import ElementwiseTensorProduct
from e3nn.nn._activation import Activation
from e3nn.util.jit import compile_mode
from e3nn.nn._gate import _Sortcut


from ..mlp import MLP
from .linear import Linear
from ..layout import LayoutTransform


@compile_mode("script")
class GateNonlinear(torch.nn.Module):
    """delete irreps_scalars and act_scalars from e3nn's Gate"""
    def __init__(self, irreps_gates, act_gates, irreps_gated) -> None:
        super().__init__()

        irreps_gates = o3.Irreps(irreps_gates)
        irreps_gated = o3.Irreps(irreps_gated)

        if len(irreps_gates) > 0 and irreps_gates.lmax > 0:
            raise ValueError(f"Gate scalars must be scalars, instead got irreps_gates = {irreps_gates}")
        if irreps_gates.num_irreps != irreps_gated.num_irreps:
            raise ValueError(
                f"There are {irreps_gated.num_irreps} irreps in irreps_gated, but a different number "
                f"({irreps_gates.num_irreps}) of gate scalars in irreps_gates"
            )

        self.sc = _Sortcut(irreps_gates, irreps_gated)
        self.irreps_gates, self.irreps_gated = self.sc.irreps_outs
        self._irreps_in = self.sc.irreps_in

        self.act_gates = Activation(irreps_gates, act_gates)
        irreps_gates = self.act_gates.irreps_out

        self.mul = ElementwiseTensorProduct(irreps_gated, irreps_gates)
        irreps_gated = self.mul.irreps_out

        self._irreps_out = irreps_gated

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_out})"

    def forward(self, features):
        gates, gated = self.sc(features)

        gates = self.act_gates(gates)
        gated = self.mul(gated, gates)
        features = gated

        return features

    @property
    def irreps_in(self):
        return self._irreps_in

    @property
    def irreps_out(self):
        return self._irreps_out


@compile_mode("script")
class NormNonlinear(torch.nn.Module):
    def __init__(
        self,
        irreps: o3.Irreps,
        activation: torch.nn.Module,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.irreps_in = o3.Irreps(irreps)
        self.irreps_out = o3.Irreps(irreps)
        self.norm_fn = o3.Norm(self.irreps_in, squared=False)
        self.linear = Linear(
            self.norm_fn.irreps_out,
            self.norm_fn.irreps_out,
            bias=bias,
        )
        self.activation = activation
        self.scalar_multiplier = ElementwiseTensorProduct(
            irreps_in1=self.norm_fn.irreps_out,
            irreps_in2=self.irreps_in,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.norm_fn(x)
        norm = self.linear(norm)
        norm = self.activation(norm)
        return self.scalar_multiplier(norm, x)
    

@compile_mode("script")
class GridNonlinear(torch.nn.Module):
    def __init__(
        self,
        irreps: o3.Irreps,
        activation: torch.nn.Module,
        bias: bool = False,
    ):
        super().__init__()

        # Default truncation may not enough
        lmax = max(ir.l for _, ir in irreps)
        num_channel = irreps.num_irreps // len(irreps)
        self.truncation = lmax
        self.num_latitude = 2 * (self.truncation + 1)
        self.num_longitude = 2 * (self.truncation+ 1) + 1

        self.mlp = MLP(
            [num_channel] * 4,
            bias=bias,
            layer_norm=False,
            act=activation,
        )

        to_s2 = o3.ToS2Grid(
            self.truncation, 
            (self.num_latitude, self.num_longitude), 
            normalization="component",
        )
        from_s2 = o3.FromS2Grid(
            (self.num_latitude, self.num_longitude), 
            self.truncation, 
            normalization="component",
        )

        self.register_buffer(
            "to_grid", 
            torch.einsum(
                "mbi, am -> bai", to_s2.shb, to_s2.sha
            ).detach(),
            persistent=False,
        )
        self.register_buffer(
            "from_grid", 
            torch.einsum(
                "am, mbi -> bai", from_s2.sha, from_s2.shb
            ).detach(),
            persistent=False,
        )

        self.transform = LayoutTransform(irreps)

    def _to_grid(self, x: torch.Tensor) -> torch.Tensor:           
        return torch.einsum("bai, Bic -> Bbac", self.to_grid, x)

    def _from_grid(self, x: torch.Tensor) -> torch.Tensor:       
        return torch.einsum("bai, Bbac -> Bic", self.from_grid, x)
    
    def forward(self, x: torch.Tensor):
        x = self.transform(x)
        grid = self._to_grid(x)
        B, b, a, C = grid.shape
        freq = self._from_grid(self.mlp(grid.reshape(-1, C)).reshape(B, b, a, C))
        freq = self.transform.inverse(freq)
        return freq