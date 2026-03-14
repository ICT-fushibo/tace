################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import os
TACE_WEIGHT_INIT = os.environ.get('TACE_WEIGHT_INIT', 'randn')
from typing import Optional


import torch
from e3nn import o3


class Linear(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        bias: bool = True,
    ):
        super().__init__()

        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)

        self.linear = o3.Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            internal_weights=False,
            shared_weights=False,
        )

        self.weight = torch.nn.Parameter(
            torch.empty(self.linear.weight_numel)
        )

        self._0e_muls = []
        self._0e_slices = []
        self._bias_slices = []
        acc = 0
        bias_acc = 0
        for mul, ir in irreps_out:
            dim = mul * ir.dim
            if ir.l == 0 and ir.p == 1:
                self._0e_muls.append(mul)
                self._0e_slices.append(slice(acc, acc + dim))
                self._bias_slices.append(slice(bias_acc, bias_acc + dim))
                bias_acc += dim
            acc += dim

        if bias and bias_acc > 0:
            self.bias = torch.nn.Parameter(
                torch.empty(bias_acc)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.linear(x, self.weight)
        if self.bias is not None:
            for sl, bias_sl in zip(self._0e_slices, self._bias_slices):
                out[:, sl] += self.bias[bias_sl].unsqueeze(0)
        return out

    def __repr__(self):
        return repr(self.linear) + f"(bias={self.bias is not None})"

class ElementLinear(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        bias: bool = True,
        num_elements: int,
    ):
        super().__init__()

        self.num_elements = num_elements

        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)

        self.linear = o3.Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            internal_weights=False,
            shared_weights=False,
        )

        self.weight = torch.nn.Parameter(
            torch.empty(num_elements, self.linear.weight_numel)
        )

        self._0e_muls = []
        self._0e_slices = []
        self._bias_slices = []
        acc = 0
        bias_acc = 0
        for mul, ir in irreps_out:
            dim = mul * ir.dim
            if ir.l == 0 and ir.p == 1:
                self._0e_muls.append(mul)
                self._0e_slices.append(slice(acc, acc + dim))
                self._bias_slices.append(slice(bias_acc, bias_acc + dim))
                bias_acc += dim
            acc += dim

        if bias and bias_acc > 0:
            self.bias = torch.nn.Parameter(
                torch.empty(num_elements, bias_acc)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        weight = torch.einsum("bz,zi->bi", y, self.weight)
        out = self.linear(x, weight)
        if self.bias is not None:
            bias = torch.einsum("bz,zi->bi", y, self.bias)
            for sl, bias_sl in zip(self._0e_slices, self._bias_slices):
                out[:, sl] += bias[:, bias_sl]
        return out

    def __repr__(self):
        return repr(self.linear) + f"(bias={self.bias is not None})"