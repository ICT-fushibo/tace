################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math

import torch


from ..layout import LayoutTransform
from ..so2 import SO3Rotation, so2_expand_index
from ..mlp import SmoothLeakyReLU
from ..softmax import GraphSoftmax
from ..linear import torchLinear


class SO2Attention(torch.nn.Module):
    """
    SO2 Attention for e3nn model.
    Leverage information from all SO(3) l to compute attention scores, 
    fully exploiting the lossless expressive and extrapolative capability of CGTP. 
    By combining oeq and cueq, this method avoids edge-level SO2 operations 
    typically required in conventional SO2 attention mechanisms, while preserving strong representation power.
    """
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        edge_wise_hidden: int,
        so2_angular_basis: SO3Rotation,
        reshape_in: LayoutTransform,
        num_head: int,
        weights_shape: int, 
    ) -> None:
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.edge_wise_hidden = edge_wise_hidden or self.num_channel
        self.so2_angular_basis = so2_angular_basis
        self.reshape_in = reshape_in
        self.num_out_channel = self.edge_wise_hidden
        self.num_head = num_head
        self.weights_shape = weights_shape

        self.num_components, _ = so2_expand_index(self.mmax, self.lmax)
        self.weight_numel = (self.num_components * self.num_channel * 2)

        self.num_channel_per_head = self.edge_wise_hidden // self.num_head
        assert self.num_channel % self.num_head == 0

        self.linear_alpha = torchLinear(
            self.num_channel * (lmax+1) * 2,
            self.edge_wise_hidden * (lmax+1),    
        )
        std = 1.0 / math.sqrt(self.num_channel_per_head)
        self.graph_softmax = GraphSoftmax()
        self.alpha_act = SmoothLeakyReLU()

        self.real_alpha_norm = torch.nn.LayerNorm(self.num_channel_per_head)
        self.real_alpha_dot = torch.nn.Parameter(torch.randn(self.num_head, self.num_channel_per_head))
        torch.nn.init.uniform_(self.real_alpha_dot, -std, std)


    def forward(
            self, 
            x: torch.Tensor, 
            w: torch.Tensor, 
            edge_index: torch.Tensor,
            cutoff: torch.Tensor,
            conv_weights: torch.Tensor,
        ) -> torch.Tensor:

        num_nodes = x.size(0)
        num_edges = conv_weights.size(0)
        x = self.reshape_in(x)
        m_ij = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=-1)
        m_ij = self.so2_angular_basis.rotate(m_ij) # all so3 => so2 m=0
        m_ij = w * m_ij.view(num_edges, -1)
        real_alpha = self.linear_alpha(m_ij).view(num_edges, self.lmax+1, -1)
        real_alpha = torch.bmm(self.so2_angular_basis.wigner_inv.narrow(1, 0, 1), real_alpha) # from so2 m=0 to so3 m=0
        real_alpha = real_alpha.reshape(num_edges, self.num_head, self.num_channel_per_head)
        real_alpha = self.real_alpha_norm(real_alpha)
        real_alpha = self.alpha_act(real_alpha)
        real_alpha = torch.einsum('bik, ik -> bi', real_alpha, self.real_alpha_dot)
        real_alpha = self.graph_softmax(real_alpha, edge_index[1], num_nodes=num_nodes, exp_rescale=cutoff).unsqueeze(-1) # [edge, head, 1]

        offset = 0
        new_conv_weights = []
        for shape in self.weights_shape:
            this_w = conv_weights[:, offset:offset+self.num_channel]
            this_w = this_w.view(num_edges, self.num_head, -1) * real_alpha
            new_conv_weights.append(this_w.view(num_edges, -1))
            offset += self.num_channel

        return torch.cat(new_conv_weights, dim=-1)