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
        num_fidelities: int,
        l: int,
    ) -> torch.Tensor:    
    B = x.size(0)
    fid_mul_ir = x.size(-1)
    ir = 2*l+1
    fid_mul = fid_mul_ir // ir
    mul = fid_mul // num_fidelities
    mask = torch.zeros(B, num_fidelities, mul, device=x.device, dtype=x.dtype)
    mask[torch.arange(B, device=x.device), node_fidelity, :] = 1
    mask = mask.reshape(B, fid_mul, 1)
    x = x.reshape(B, fid_mul, ir)
    return  (x * mask).view(B, -1)

class ScalarReadOut(ReadOut):
    def _setup(self):

        if self.layer == self.num_layers-1: 
            self.linear1 = Linear(
                    irreps_in=self.irreps_in,
                    irreps_out=self.irreps_hidden,
                    bias=self.bias,
                )
            self.act = ACTIVATION[self.scalar_act]() 
            self.linear2 = Linear(
                    self.irreps_hidden,
                    irreps_out=self.irreps_out,
                    bias=self.bias,
                )
            self.last_layer = True
        else:
            self.linear1 = Linear(
                    irreps_in=self.irreps_in,
                    irreps_out=self.irreps_out,
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
            x = mh_mask(x, node_fidelity, self.num_fidelities, self.l)
        return self.linear2(x)

class TensorReadOut(ReadOut):
    def _setup(self):

        p = 'e'

        if self.layer == self.num_layers-1: 
            self.linear1 = Linear(
                irreps_in=self.irreps_in,
                irreps_out=self.irreps_hidden + self.irreps_gates,
                bias=self.bias,
            )
            self.act = GateNonlinear(
                irreps_gates=self.irreps_gates,
                act_gates=[ACTIVATION[self.tensor_act]()] * len(self.irreps_gates),
                irreps_gated=self.irreps_hidden,
            )
            self.linear2 = Linear(
                irreps_in=self.irreps_hidden,
                irreps_out=self.irreps_out,
                bias=self.bias,
            )
            self.last_layer = True
        else:
            self.linear1 = Linear(
                irreps_in=self.irreps_in,
                irreps_out=self.irreps_hidden,
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
            x = mh_mask(x, node_fidelity, self.num_fidelities, self.l)
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