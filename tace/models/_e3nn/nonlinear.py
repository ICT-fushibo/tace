################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Union


import torch
from e3nn import o3
from e3nn.o3._tensor_product._sub import ElementwiseTensorProduct
from e3nn.nn import Activation
from e3nn.nn._gate import _Sortcut


class O3Gate(torch.nn.Module):
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

    def forward(self, features, gates: Union[torch.Tensor, None] = None):
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


