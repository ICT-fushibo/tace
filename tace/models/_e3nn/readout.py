################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import List, Union

import torch
from e3nn import o3
from e3nn.nn import Activation

from ..linear import e3nnLinear
from ..mlp import ACTIVATION
from .base import ReadOut
from .nonlinear import O3Gate


def mh_mask(
    x: torch.Tensor,
    node_fidelity: torch.Tensor,
    num_fidelities: int,
    l: int,
) -> torch.Tensor:
    B = x.size(0)
    fid_mul_ir = x.size(-1)
    ir = 2 * l + 1
    fid_mul = fid_mul_ir // ir
    mul = fid_mul // num_fidelities
    mask = torch.zeros(B, num_fidelities, mul, device=x.device, dtype=x.dtype)
    mask[torch.arange(B, device=x.device), node_fidelity, :] = 1
    mask = mask.reshape(B, fid_mul, 1)
    x = x.reshape(B, fid_mul, ir)
    return (x * mask).view(B, -1)


class ScalarReadOut(ReadOut):
    def _setup(self):

        if self.layer == self.num_layers - 1:
            self.linear2 = torch.nn.ModuleList()
            self.acts = torch.nn.ModuleList(
                Activation(irreps_in=hidden, acts=[ACTIVATION[self.scalar_act]()])
                for hidden in self.irreps_hidden
            )
            for irreps_in, irreps_out in zip(
                ([self.irreps_in] + self.irreps_hidden + [self.irreps_out])[:-1],
                ([self.irreps_in] + self.irreps_hidden + [self.irreps_out])[1:],
            ):
                self.linear2.append(
                    e3nnLinear(
                        irreps_in,
                        irreps_out,
                        bias=self.use_bias,
                    )
                )
            self.last_layer = True
        else:
            self.linear1 = torch.nn.ModuleList(
                [
                    e3nnLinear(
                        irreps_in=self.irreps_in,
                        irreps_out=self.irreps_out,
                        bias=self.use_bias,
                    )
                ]
            )
            self.last_layer = False

    def forward(
        self, x: torch.Tensor, node_fidelity: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        if not self.last_layer:
            return self.linear1[0](x)
        for idx, linear in enumerate(self.linear2[:-1]):
            x = self.acts[idx](linear(x))
            if self.num_fidelities > 1:
                x = mh_mask(x, node_fidelity, self.num_fidelities, self.l)
        return self.linear2[-1](x)


class TensorReadOut(ReadOut):
    def _setup(self):

        if self.layer == self.num_layers - 1:
            self.linear2 = torch.nn.ModuleList()
            self.acts = torch.nn.ModuleList()
            for irreps_gates, irreps_gated in zip(
                self.irreps_gates, self.irreps_hidden
            ):
                self.acts.append(
                    O3Gate(
                        irreps_gates=irreps_gates,
                        act_gates=[ACTIVATION[self.tensor_act]()],
                        irreps_gated=irreps_gated,
                    )
                )
            for idx, (irreps_in, irreps_out) in enumerate(
                zip(
                    ([self.irreps_in] + self.irreps_hidden + [self.irreps_out])[:-2],
                    ([self.irreps_in] + self.irreps_hidden + [self.irreps_out])[1:-1],
                )
            ):
                self.linear2.append(
                    e3nnLinear(
                        irreps_in=irreps_in,
                        irreps_out=self.acts[idx].irreps_in,
                        bias=self.use_bias,
                    )
                )
            self.linear2.append(
                e3nnLinear(
                    irreps_in=self.irreps_hidden[-1],
                    irreps_out=self.irreps_out,
                    bias=self.use_bias,
                )
            )
            self.last_layer = True
        else:
            self.linear1 = torch.nn.ModuleList(
                [
                    e3nnLinear(
                        irreps_in=self.irreps_in,
                        irreps_out=self.irreps_out,
                        bias=self.use_bias,
                    )
                ]
            )
            self.last_layer = False

    def forward(
        self, x: torch.Tensor, node_fidelity: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        if not self.last_layer:
            return self.linear1[0](x)
        for idx, linear in enumerate(self.linear2[:-1]):
            x = self.acts[idx](linear(x))
            if self.num_fidelities > 1:
                x = mh_mask(x, node_fidelity, self.num_fidelities, self.l)
        return self.linear2[-1](x)


def build_scalar_readout(
    num_layers: int,
    hidden_channel: List[int],
    bias: bool,
    num_fidelities: int,
    use_alllayer: bool,
    parity: bool,
    irreps_in: list[o3.Irreps],
    irreps_out: Union[str, o3.Irreps],
):
    readouts = torch.nn.ModuleList()
    for layer in range(num_layers):
        readouts.append(
            ScalarReadOut(
                layer=layer,
                num_layers=num_layers,
                hidden_channel=hidden_channel,
                bias=bias,
                num_fidelities=num_fidelities,
                parity=parity,
                irreps_in=irreps_in[layer],
                irreps_out=o3.Irreps(irreps_out),
            )
        )
    if use_alllayer:
        return torch.nn.ModuleList(readouts)
    else:
        return torch.nn.ModuleList([readouts[-1]])


def build_tensor_readout(
    num_layers: int,
    hidden_channel: int,
    bias: bool,
    num_fidelities: int,
    use_alllayer: bool,
    parity: bool,
    irreps_in: list[o3.Irreps],
    irreps_out: Union[str, o3.Irreps],
):
    readouts = torch.nn.ModuleList()
    for layer in range(num_layers):
        readouts.append(
            TensorReadOut(
                layer=layer,
                num_layers=num_layers,
                hidden_channel=hidden_channel,
                bias=bias,
                num_fidelities=num_fidelities,
                parity=parity,
                irreps_in=irreps_in[layer],
                irreps_out=o3.Irreps(irreps_out),
            )
        )
    if use_alllayer:
        return torch.nn.ModuleList(readouts)
    else:
        return torch.nn.ModuleList([readouts[-1]])
