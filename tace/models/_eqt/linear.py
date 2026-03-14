################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional

import torch
import equitorch as eqt


from ..env import TACE_WEIGHT_INIT


class Linear(torch.nn.Module):
    def __init__(
        self,
        irreps_in: eqt.irreps.Irreps,
        irreps_out: eqt.irreps.Irreps,
        channels_in: int,
        channels_out: int,
        *,
        path_norm: bool = True,
        channel_norm: bool = True,
        bias: bool = True,
    ):
        super().__init__()
   
        self.channel_norm = channel_norm

        irreps_in = eqt.irreps.Irreps(irreps_in)
        irreps_out = eqt.irreps.Irreps(irreps_out)

        self.linear = eqt.nn.IrrepsLinear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            channels_in=channels_in,
            channels_out=channels_out,
            internal_weights=False,
            path_norm=path_norm,
            channel_norm=channel_norm,
        )

        self.weight = torch.nn.Parameter(
            torch.empty(self.linear.weight_shape)
        )

        self._0e_slices = []
        acc = 0
        for ir in irreps_out:
            if ir.l == 0 and ir.p == 1:
                self._0e_slices.append(slice(acc, acc + ir.dim))
            acc += ir.dim

        if bias and len(self._0e_slices) > 0:
            self.bias = torch.nn.Parameter(
                torch.empty(len(self._0e_slices), channels_out)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if TACE_WEIGHT_INIT == 'uniform':
            eqt.nn.initialize_linear(
                self.weight, channel_normed=self.channel_norm
            )
        else:
            torch.nn.init.normal_(self.weight)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.linear(x, self.weight)

        if self.bias is not None:
            for i, sl in enumerate(self._0e_slices):
                out[:, sl, :] += self.bias[i][None, None, :]

        return out

    def __repr__(self):
        return repr(self.linear) + f"(bias={self.bias is not None})"

class ElementLinear(torch.nn.Module):
    def __init__(
        self,
        irreps_in: eqt.irreps.Irreps,
        irreps_out: eqt.irreps.Irreps,
        channels_in: int,
        channels_out: int,
        *,
        path_norm: bool = True,
        channel_norm: bool = True,
        bias: bool = True,
        num_elements: int = -1,
    ):
        super().__init__()

        self.channel_norm = channel_norm
        self.num_elements = num_elements

        irreps_in = eqt.irreps.Irreps(irreps_in)
        irreps_out = eqt.irreps.Irreps(irreps_out)

        self.linear = eqt.nn.IrrepsLinear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            channels_in=channels_in,
            channels_out=channels_out,
            internal_weights=False,
            path_norm=path_norm,
            channel_norm=channel_norm,
        )

        self.weight = torch.nn.Parameter(
            torch.empty(num_elements, *self.linear.weight_shape)
        )

        self._0e_slices = []
        acc = 0
        for ir in irreps_out:
            if ir.l == 0 and ir.p == 1:
                self._0e_slices.append(slice(acc, acc + ir.dim))
            acc += ir.dim

        if bias and len(self._0e_slices) > 0:
            self.bias = torch.nn.Parameter(
                torch.empty(num_elements, len(self._0e_slices), channels_out)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if TACE_WEIGHT_INIT == 'uniform':
            eqt.nn.initialize_linear(
                self.weight, channel_normed=self.channel_norm
            )
        else:
            torch.nn.init.normal_(self.weight)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.linear(x, torch.einsum("bz, zpcC -> bpcC", y, self.weight))

        if self.bias is not None:
            bias = torch.einsum("bz, zkc -> bkc", y, self.bias)
            for i, sl in enumerate(self._0e_slices):
                out[:, sl, :] += bias[:, i][:, None, :]

        return out

    def __repr__(self):
        return repr(self.linear) + f"(bias={self.bias is not None})"