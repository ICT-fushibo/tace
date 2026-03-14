################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import List, Optional


import torch


from ..mlp import ACTIVATION
from .base import ReadOut
from .linear import Linear
from .nonlinear import GateNonlinear


def mh_mask(
        x: torch.Tensor, 
        node_fidelity: torch.Tensor, 
        num_fidelities: int
    ) -> torch.Tensor:
    B = x.size(0)
    C_ALL = x.size(-1)
    C = C_ALL // num_fidelities
    mask = torch.zeros(B, num_fidelities, C, device=x.device, dtype=x.dtype)
    mask[torch.arange(B, device=x.device), node_fidelity, :] = 1
    mask = mask.reshape(B, 1, C_ALL)
    return x * mask


class ScalarReadOut(ReadOut):
    def _setup(self):

        if self.layer == self.num_layers-1: 
            self.linear1 = Linear(
                    irreps_in=self.irreps_in,
                    irreps_out="0e",
                    channels_in=self.num_channel,
                    channels_out=self.hidden * self.num_fidelities,
                    bias=self.bias,
                )
            self.act = ACTIVATION[self.scalar_act]() 
            self.linear2 = Linear(
                    irreps_in="0e",
                    irreps_out="0e",
                    channels_in=self.hidden * self.num_fidelities,
                    channels_out=1 * self.num_fidelities,
                    bias=self.bias,
                )
            self.last_layer = True
        else:
            self.linear1 = Linear(
                    irreps_in=self.irreps_in,
                    irreps_out="0e",
                    channels_in=self.num_channel,
                    channels_out=1 * self.num_fidelities,
                    bias=self.bias,
                )
            self.last_layer = False
        

    def forward(
        self, x: torch.Tensor, node_fidelity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not self.last_layer:
            return self.linear1(x)
        x = self.act(self.linear1(x))
        if self.num_fidelities > 1:
            x = mh_mask(x, node_fidelity, self.num_fidelities)
        return self.linear2(x)

class TensorReadOut(ReadOut):
    def _setup(self):

        p = 'e'

        if self.layer == self.num_layers-1: 
            self.linear1 = Linear(
                    irreps_in=self.irreps_in,
                    irreps_out=f"1x{self.l}{p} +1x0e",
                    channels_in=self.num_channel,
                    channels_out=self.hidden * self.num_fidelities,
                    bias=self.bias,
                )
            self.act = GateNonlinear(
                f"{self.l}{p}",
                activation=ACTIVATION[self.tensor_act](),
                irrep_wise=True,
            )
            self.linear2 = Linear(
                    irreps_in=f"{self.l}{p}",
                    irreps_out=f"{self.l}{p}",
                    channels_in=self.hidden,
                    channels_out=1 * self.num_fidelities,
                    bias=self.bias,
                )
            self.last_layer = True
        else:
            self.linear1 = Linear(
                    irreps_in=self.irreps_in,
                    irreps_out=f"{self.l}{p}",
                    channels_in=self.num_channel,
                    channels_out=1 * self.num_fidelities,
                    bias=self.bias,
                )
            self.last_layer = False
        
    def forward(
        self, x: torch.Tensor, node_fidelity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not self.last_layer:
            return self.linear1(x)
        x = self.act(self.linear1(x))
        if self.num_fidelities > 1:
            x = mh_mask(x, node_fidelity, self.num_fidelities)
        return self.linear2(x)
    
    
def build_scalar_readout(
    num_layers: int,
    Lmax: int,
    lmax: int,
    num_channel: int,
    hidden_channel: List[int], 
    target_weight: List[int],
    bias: bool,
    num_fidelities: int,
    use_alllayer: bool,
    l: int,
):
    readouts = torch.nn.ModuleList()
    for layer in range(num_layers):
        readouts.append(
            ScalarReadOut(
                layer=layer,
                num_layers=num_layers,
                Lmax=Lmax,
                lmax=lmax,
                num_channel=num_channel,
                hidden_channel=hidden_channel, 
                target_weight=target_weight,
                bias=bias,
                num_fidelities=num_fidelities,
                l=l,
            )
        )
    if use_alllayer:
        return torch.nn.ModuleList(readouts)
    else:
        return torch.nn.ModuleList([readouts[-1]])

def build_tensor_readout(
    num_layers: int,
    Lmax: int,
    lmax: int,
    num_channel: int,
    hidden_channel: int, 
    target_weight: List[int],
    bias: bool,
    num_fidelities: int,
    use_alllayer: bool,
    l: int,
):
    readouts = torch.nn.ModuleList()
    for layer in range(num_layers):
        readouts.append(
            TensorReadOut(
                layer=layer,
                num_layers=num_layers,
                Lmax=Lmax,
                lmax=lmax,
                num_channel=num_channel,
                hidden_channel=hidden_channel,
                target_weight=target_weight,
                bias=bias,
                num_fidelities=num_fidelities,
                l=l,
            )
        )
    if use_alllayer:
        return torch.nn.ModuleList(readouts)
    else:
        return torch.nn.ModuleList([readouts[-1]])