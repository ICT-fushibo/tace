################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
import torch
import torch.nn.functional as F

from ..layout import LayoutTransform
from .base import Residual
from .linear import Linear, ElementLinear

# torch.set_printoptions(precision=4, sci_mode=False)

class AgnosticResidual(Residual):

    def _setup(self):

        self.identity = Linear(
            self.irreps_in,
            self.irreps_out,
            bias=self.use_bias
        )    

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        
        return self.identity(node_feats)
    

class AwareResidual(Residual):

    def _setup(self):

        self.identity = ElementLinear(
            self.irreps_in,
            self.irreps_out,
            bias=self.use_bias,
            num_elements=self.num_elements,
        )    

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        
        return self.identity(node_feats, node_attrs)
    

class AttentionResidual(Residual):
    def _setup(self):

        assert self.layer > 0
        assert self.num_layers > 2

        self.alpha = 1 / math.sqrt(self.num_channel) # not use rms_norm for key
        self.query = torch.nn.Parameter(torch.zeros(1, self.num_channel)) # if need, can add element
        self.linear = torch.nn.ModuleList()
        self.reshape = LayoutTransform(self.irreps_out)

        for _ in range(self.window):
            self.linear.append(
                ElementLinear(
                    self.irreps_in,
                    self.irreps_out,
                    bias=self.use_bias,
                    num_elements=self.num_elements,
                )
            )

    def forward(self, prev_feats: list[torch.Tensor], node_attrs: torch.Tensor):
        prev_feats = prev_feats[-self.window:]
        key = torch.stack([feats[:, :self.num_channel] for feats in prev_feats], dim=0)
        logits = torch.einsum('c, lbc -> lb', self.query.squeeze(0), key) * self.alpha
        attn = F.softmax(logits, dim=0) 
        new_feats = []
        for idx in range(self.window):
            new_feats.append(
                self.reshape(self.linear[idx](prev_feats[idx], node_attrs))
            )
        value = torch.stack(new_feats, dim=0)

        return self.reshape.inverse(torch.einsum('lb, lbmc -> bmc', attn, value))

    
RESIDUAL = {
    'aware': AwareResidual,
    'agnostic': AgnosticResidual,
    'AttnRes': AttentionResidual,
}
