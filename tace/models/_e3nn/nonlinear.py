################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from math import sqrt


import torch
from e3nn import o3
from e3nn.o3._tensor_product._sub import ElementwiseTensorProduct
from e3nn.nn import Activation
from e3nn.nn._gate import _Sortcut


from ..mlp import FFN
from ..layout import LayoutTransform
from .linear import Linear


class GatedLinearUnit(torch.nn.Module):
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

    def forward(self, features, gates: torch.Tensor | None = None):
        if gates is None:
            gates, gated = self.sc(features)
        else:
            gated = features

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


class MixtralExpertsGatedLinearUnit(torch.nn.Module):
    def __init__(
        self, 
        irreps_in: o3.Irreps, 
        irreps_out: o3.Irreps, 
        bias: bool,
        act_gates: list[torch.nn.Module],
        num_experts: int,
        num_shared_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()

        self.use_bias = bias
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k    
        irreps_gated = irreps_out
        irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in irreps_out)
        self.sc = _Sortcut(irreps_gates, irreps_gated)
        self.irreps_gates, self.irreps_gated = self.sc.irreps_outs
        self.act_gates = Activation(irreps_gates, act_gates)
        self.mul = ElementwiseTensorProduct(irreps_gated, self.act_gates.irreps_out)
        self.irreps_in = irreps_in
        self.irreps_out = self.mul.irreps_out
        self.num_channel = irreps_out.count("0e")

        self.router = Linear(
            f"{self.num_channel}x0e",
            f"{self.num_experts}x0e",
            bias=False,
        )
        self.linear = Linear(
            irreps_in=irreps_in,
            irreps_out=(irreps_gates + irreps_out).regroup(),
            bias=self.use_bias,
            internal_weights=False,
        )

        if num_shared_experts > 0:
            self.weight1 = torch.nn.Parameter(
                torch.empty(num_shared_experts, self.linear.weight_numel)
            )
            self.alpha = 1.0 / (sqrt(self.num_shared_experts))
        self.weight2 = torch.nn.Parameter(
            torch.empty(num_experts, self.linear.weight_numel)
        )
    
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "weight1"):
            torch.nn.init.normal_(self.weight1)
        torch.nn.init.normal_(self.weight2)

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:

        gate_logits = self.router(node_feats[:, :self.num_channel])
        gate_probs = torch.softmax(gate_logits, dim=-1)
        topk_probs, topk_idx = torch.topk(gate_probs, k=self.top_k, dim=-1) 
        gate_probs_sparse = torch.zeros_like(gate_probs)
        gate_probs_sparse.scatter_(
            -1, 
            topk_idx, 
            topk_probs / (topk_probs.sum(-1, keepdim=True))
        )

        if self.num_shared_experts > 0:
            weight1 = self.alpha * self.weight1.sum(dim=0, keepdim=True)
        else:
            weight1 = 0.0

        weight2 = torch.einsum("bz, zi -> bi", gate_probs_sparse, self.weight2)

        router_l2 = gate_probs_sparse.pow(2).sum(dim=-1, keepdim=True)  # [B, 1]
  
        if self.num_shared_experts > 0:
            scale = torch.sqrt(1.0 + router_l2 + 1e-8)
        else:
            scale = torch.sqrt(router_l2 + 1e-8)

        weight = (weight1 + weight2) / scale

        node_feats = self.linear(
            node_feats, 
            weight,
        )
      
        gates, gated = self.sc(node_feats)
        gates = self.act_gates(gates)

        return self.mul(gated, gates)
        
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"\trouter={self.router}, \n"
            f"\tlinear1={self.linear1}, \n"
            f"\tlinear2={self.linear2}, \n"
            f"\tnum_shared_experts={self.num_shared_experts}, \n"
            f"\tnum_experts={self.num_experts}, \n"
            f")"
        )


class NormLinearUnit(torch.nn.Module):
    def __init__(
        self,
        irreps: o3.Irreps,
        activation: torch.nn.Module,
    ) -> None:
        super().__init__()

        self.irreps_in = o3.Irreps(irreps)
        self.irreps_out = o3.Irreps(irreps)
        self.norm_fn = o3.Norm(self.irreps_in, squared=True)
        with torch.no_grad():
            self.weight = torch.nn.Parameter(
                torch.randn(self.irreps_in.num_irreps) 
                / torch.tensor([2*l+1 for l in self.irreps_in.ls])
            ) # TODO
        self.bias = torch.nn.Parameter(torch.zeros(self.irreps_in.num_irreps))
        self.activation = Activation(self.norm_fn.irreps_out.regroup(), [activation])
        self.scalar_multiplier = ElementwiseTensorProduct(
            irreps_in1=self.norm_fn.irreps_out,
            irreps_in2=self.irreps_in,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        norm = self.norm_fn(x) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        norm = self.activation(norm)
        if y is not None:
            return self.scalar_multiplier(norm, y)
        return self.scalar_multiplier(norm, x)


class GridMLPUnit(torch.nn.Module):
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

        self.mlp = FFN['mlp'](
            [num_channel] + [num_channel*2] + [num_channel],
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