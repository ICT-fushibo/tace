################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Union

import torch
from tace.utils.torch_scatter import scatter_sum
from e3nn import o3


from tace.utils.env import get_tace_use_oeq, get_tace_use_cue, get_tace_use_eqt
from ..layout import LayoutTransform
from ..so2 import (
    SO3Rotation, SO2Linear, SO2Gate, SO2Norm, SO2ComplexMul,
    so2_expand_index, so3_expand_index,
)
from .paths import generate_paths
from .edge_prod import SO2EdgeProductBasis


from ..mlp import ScaledSigmoid, SmoothLeakyReLU
from ..softmax import GraphSoftmax



class O3ScatterTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        l1l2: Union[str, None] = None,
        l2l3: Union[str, None] = None,
        l3l1: Union[str, None] = None,
    ) -> None:
        super().__init__()

        irreps_in1 = o3.Irreps(irreps_in1)
        irreps_in2 = o3.Irreps(irreps_in2)
        irreps_out = o3.Irreps(irreps_out)

        instructions, actual_irreps_out = generate_paths(
            irreps_out=irreps_out,
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            l1l2=l1l2,
            l2l3=l2l3,
            l3l1=l3l1,
            e3nn_mode='uvu',
        )

        self.tp = o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            actual_irreps_out,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = actual_irreps_out
        self.instructions = instructions
        self.weight_numel = self.tp.weight_numel
        self.use_oeq = get_tace_use_oeq() == '1'
        self.use_cue = get_tace_use_cue() == '1'

        if self.use_oeq:
            from .._oeq import e3nnOeqScatterTensorProduct
            self.fused_tp = e3nnOeqScatterTensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out,
                instructions=self.instructions,
            )
        elif self.use_cue:
            from .._cue import e3nnCueScatterTensorProduct
            self.fused_tp = e3nnCueScatterTensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out,
                l1l2=l1l2,
                l2l3=l2l3,
                l3l1=l3l1,
            )
        else:
            pass

    def forward(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
            w: torch.Tensor, 
            edge_index: torch.Tensor
        ) -> torch.Tensor:
        
        if hasattr(self, "fused_tp"):
            return self.fused_tp(x, y, w, edge_index)
        return scatter_sum(
            self.tp(x[edge_index[0]], y, w), 
            edge_index[1], 
            dim=0, 
            dim_size=x.size(0)
        )


class uuuTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        l1l2: Union[str, None] = None,
        l2l3: Union[str, None] = None,
        l3l1: Union[str, None] = None,
        trainable: bool = False,
    ) -> None:
        super().__init__()

        instructions, actual_irreps_out = generate_paths(
            irreps_out=irreps_out,
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            l1l2=l1l2,
            l2l3=l2l3,
            l3l1=l3l1,
            e3nn_mode='uuu',
            trainable=trainable,
        )

        self.tp = o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            actual_irreps_out,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = actual_irreps_out
        self.instructions = instructions
        self.weight_numel = self.tp.weight_numel
        self.use_eqt = get_tace_use_eqt() == '1'
        # self.use_oeq = TACE_USE_OEQ == '1'
        # self.use_cue = TACE_USE_CUE == '1'

        if self.use_eqt:
            from .._eqt import e3nnEqtTensorProduct
            self.fused_tp = e3nnEqtTensorProduct(
                irreps_in1=irreps_in1,
                irreps_in2=irreps_in2,
                irreps_out=actual_irreps_out,
                num_channel=irreps_in2.count("1o"),
                path=instructions,
            )
        else:
            pass

    def forward(
            self, x: torch.Tensor, y: torch.Tensor, ws: Union[torch.Tensor, None] = None
        ) -> torch.Tensor:
            if hasattr(self, "fused_tp"):
                return self.fused_tp(x, y, ws)
            return self.tp(x, y, ws)
    


# modSwiGLU-SO2
class SO2ScatterTensorProduct(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        num_elements: int,
        edge_wise_hidden: int,
        so2_angular_basis: SO3Rotation,
        reshape_in: LayoutTransform,
        reshape_out: LayoutTransform,

        separate_so2_radial: bool,
        num_head: int,
        use_so2_edge_ace: bool,
        use_graph_softmax: bool,



    ) -> None:
        super().__init__()

        edge_wise_hidden = 32

        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.num_elements = num_elements
        self.edge_wise_hidden = edge_wise_hidden or self.num_channel
        self.so2_angular_basis = so2_angular_basis
        self.reshape_in = reshape_in
        self.reshape_out = reshape_out
   
        self.num_out_channel = 64
        
        self.separate_so2_radial = separate_so2_radial
        self.use_so2_edge_ace = use_so2_edge_ace
        self.num_head = num_head
        self.use_graph_softmax = use_graph_softmax


        self.num_components, expand_index = so2_expand_index(self.mmax, self.lmax)
        # if self.separate_so2_radial:
        #     self.num_components = self.num_components * 2 - (lmax+1)
        #     self.complex_mul = SO2ComplexMul(mmax, lmax, num_channel * 2, channel_wise=False)
        self.weight_numel = (self.num_components * self.num_channel * 2)
        self.register_buffer('expand_index', expand_index, persistent=False)

   
        self.num_gates = sum(lmax+1 for _ in range(mmax+1))

        self.ace = SO2EdgeProductBasis(
            mmax, 
            lmax, 
            self.edge_wise_hidden,
            num_elements=num_elements,
            m1m2='<=',
            agnostic=True,
        )

        self.linear_up = SO2Linear(
            mmax,
            lmax,
            self.num_channel * 2,
            self.edge_wise_hidden,     
            num_components_out=[self.num_gates + lmax+1] + [lmax+1 for m in range(1, mmax+1)],
        )
        self.split_list = [self.num_gates, (lmax+1) + (lmax+1) * mmax * 2]

        # if self.use_so2_edge_ace:    
        #     self.ace = SO2EdgeProductBasis(
        #         mmax, 
        #         lmax, 
        #         self.edge_wise_hidden,
        #         num_elements=num_elements,
        #         m1m2='<=',
        #         agnostic=True,
        #     )

        self.nonlinearity = SO2Gate(
            mmax,
            lmax,
            self.edge_wise_hidden,   
            channel_wise=True,
        )
        self.linear_down = SO2Linear(
            mmax,
            lmax,
            self.edge_wise_hidden,     
            64,      
            num_components_in=[lmax+1] * (mmax+1),
        )


        if self.use_graph_softmax:
            self.num_channel_per_head = self.edge_wise_hidden // self.num_head
            assert self.num_channel % self.num_head == 0

            self.linear_alpha = SO2Linear(
                0,
                lmax,
                self.num_channel * 2,
                self.edge_wise_hidden,     
                num_components_out=[1],
            )
            std = 1.0 / math.sqrt(self.num_channel_per_head)
            self.graph_softmax = GraphSoftmax()
            self.alpha_act = SmoothLeakyReLU()

            self.real_alpha_norm = torch.nn.LayerNorm(self.num_channel_per_head)
            self.real_alpha_dot = torch.nn.Parameter(torch.randn(self.num_head, self.num_channel_per_head))
            torch.nn.init.uniform_(self.real_alpha_dot, -std, std)

            # self.imag_alpha_norm = torch.nn.LayerNorm(self.num_channel_per_head)
            # self.imag_alpha_dot = torch.nn.Parameter(torch.randn(self.num_head, self.num_channel_per_head))
            # torch.nn.init.uniform_(self.imag_alpha_dot, -std, std)


    def forward(
            self, 
            x: torch.Tensor, 
            node_attrs: torch.Tensor,
            w: torch.Tensor, 
            edge_index: torch.Tensor,
            cutoff: torch.Tensor,
        ) -> torch.Tensor:

        num_nodes = x.size(0)
        num_edges = w.size(0)
        x = self.reshape_in(x) # [B, M, C]
        m_ij = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=-1)
        m_ij = self.so2_angular_basis.rotate(m_ij)


        # if self.separate_so2_radial:
        #     w = w.view(num_edges, self.num_components, -1)
        #     m_ij = self.complex_mul(w, m_ij, scale=False)
        # else:
        #     w = w.view(num_edges, self.num_components, -1)
        #     w = torch.index_select(w, dim=1, index=self.expand_index)
        #     m_ij = w * m_ij


        w = w.view(num_edges, self.num_components, -1)
        w = torch.index_select(w, dim=1, index=self.expand_index)
        m_ij = w * m_ij

        if self.use_graph_softmax:
            real_alpha = self.linear_alpha(m_ij)

        m_ij = self.linear_up(m_ij) 
        gate = m_ij.narrow(1, 0, self.split_list[0])
        m_ij = m_ij.narrow(1, self.split_list[0], self.split_list[1])


        if self.use_so2_edge_ace:
            m_ij = self.ace(m_ij, node_attrs, edge_index)

        m_ij = self.nonlinearity(m_ij, gate) 
        m_ij = self.linear_down(m_ij)

        if self.use_graph_softmax:
            real_alpha = real_alpha.reshape(num_edges, self.num_head, self.num_channel_per_head)
            real_alpha = self.real_alpha_norm(real_alpha)
            real_alpha = self.alpha_act(real_alpha)
            real_alpha = torch.einsum('bik, ik -> bi', real_alpha, self.real_alpha_dot)
            real_alpha = self.graph_softmax(real_alpha, edge_index[1], num_nodes=num_nodes, exp_rescale=cutoff) # [edge, head]
            if cutoff is not None:
                real_alpha = real_alpha * cutoff
            real_alpha = real_alpha.view(num_edges, 1, self.num_head, 1)
            m_ij = m_ij.view(num_edges, -1, self.num_head, self.num_channel_per_head)
            m_ij = real_alpha * m_ij 
            m_ij = m_ij.view(num_edges, -1, self.edge_wise_hidden)
        else:
            if cutoff is not None:
                m_ij = m_ij * cutoff.unsqueeze(-1)
  
        m_ij = self.so2_angular_basis.rotate_inv(m_ij)
        
        return self.reshape_out.inverse(
            scatter_sum(
                m_ij, 
                edge_index[1], 
                dim=0, 
                dim_size=num_nodes,
            )
        )
    

from ..mlp import MLP

class SO2Attention(torch.nn.Module):
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

        self.num_components, expand_index = so2_expand_index(self.mmax, self.lmax)
        self.weight_numel = (self.num_components * self.num_channel * 2)
        self.register_buffer('expand_index', expand_index, persistent=False)

        self.num_channel_per_head = self.edge_wise_hidden // self.num_head
        assert self.num_channel % self.num_head == 0

        self.linear_alpha = SO2Linear(
            0,
            lmax,
            self.num_channel * 2,
            self.edge_wise_hidden,     
            num_components_out=[1],
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
        m_ij = self.so2_angular_basis.rotate(m_ij)
        w = w.view(num_edges, self.num_components, -1)
        w = torch.index_select(w, dim=1, index=self.expand_index)
        m_ij = w * m_ij
        real_alpha = self.linear_alpha(m_ij)
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