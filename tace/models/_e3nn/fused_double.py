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
    


class SO2ScatterTensorProduct(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        edge_wise_hidden: int,
        num_head: int,
        use_graph_softmax: bool,
        is_so2_layout: bool,
        use_both_Bi_Bj: bool,
        use_so2_edge_ace: bool,
        edge_nonlinear: Union[str, None],
        num_elements: int,
        so2_angular_basis: SO3Rotation,
        reshape_in: LayoutTransform,
        reshape_out: LayoutTransform,
        scatter: Union[str, None],
    ) -> None:
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.edge_wise_hidden = edge_wise_hidden or self.num_channel
        self.is_so2_layout = is_so2_layout
        self.edge_nonlinear = edge_nonlinear
        self.edge_nonlinear_type = self.edge_nonlinear.split('_')[-1]
        self.use_both_Bi_Bj = use_both_Bi_Bj
        self.use_so2_edge_ace = use_so2_edge_ace
        self.scatter = scatter
        self.num_out_channel = num_channel
        self.so2_angular_basis = so2_angular_basis
        self.reshape_in = reshape_in
        self.reshape_out = reshape_out
        self.num_head = num_head
        self.use_graph_softmax = use_graph_softmax

        Cin = num_channel * 2 if self.use_both_Bi_Bj else num_channel
        Cout = num_channel


        self.num_components, expand_index = so2_expand_index(self.mmax, self.lmax)
        self.num_components = self.num_components * 2 - (self.lmax + 1)
        self.weight_numel = self.num_components * Cin 
        self.register_buffer('expand_index', expand_index, persistent=False)

        self.num_gates = 0 
        for m in range(mmax + 1):
            self.num_gates += lmax + 1 - m

        up_num_components_out = [lmax+1] + [lmax+1-m for m in range(1, mmax+1)]
        down_num_components_in = [lmax+1] + [lmax+1-m for m in range(1, mmax+1)]

        nonlinear_cls = SO2Gate
        self.nonlinearity = nonlinear_cls(
            mmax,
            lmax,
            self.edge_wise_hidden,   
            channel_wise=False,
        )
        self.linear_gate = SO2Linear(
            0,
            lmax,
            Cin,
            self.edge_wise_hidden,     
            num_components_out=[self.num_gates * 2 - (self.lmax+1)],
        )
        self.linear_up = SO2Linear(
            mmax,
            lmax,
            Cin,
            self.edge_wise_hidden,     
            num_components_out=up_num_components_out,
        )
        self.linear_down = SO2Linear(
            mmax,
            lmax,
            self.edge_wise_hidden,     
            Cout,     
            num_components_in=down_num_components_in,
        )

        from ..softmax import GraphSoftmax
        from ..mlp import SmoothLeakyReLU

        self.linear_alpha = SO2Linear(
            0,
            lmax,
            Cin,
            self.num_channel,  
            num_components_out=[1],   
            # num_components_out=[2],
        )


        self.num_channel_per_head = self.num_channel // self.num_head
        assert self.num_channel % self.num_head == 0

        self.alpha_act = SmoothLeakyReLU()
        std = 1.0 / math.sqrt(self.num_channel_per_head)
        self.attn_softmax = GraphSoftmax()


        self.real_alpha_norm = torch.nn.LayerNorm(self.num_channel_per_head)
        self.real_alpha_dot = torch.nn.Parameter(torch.randn(self.num_head, self.num_channel_per_head))
        torch.nn.init.uniform_(self.real_alpha_dot, -std, std)


        # self.imag_alpha_norm = torch.nn.LayerNorm(self.num_channel_per_head)
        # self.imag_alpha_dot = torch.nn.Parameter(torch.randn(self.num_head, self.num_channel_per_head))
        # torch.nn.init.uniform_(self.imag_alpha_dot, -std, std)


        self.layout = SO2ComplexMul(
            mmax,
            lmax,
            Cin,
            channel_wise=False,
        )
        self.layout2 = SO2ComplexMul(
            mmax,
            lmax,
            self.num_channel,
            channel_wise=False,
        )

        self.layout3 = SO2ComplexMul(
            mmax,
            lmax,
            self.num_head,
            channel_wise=False,
        )

        self.sigmoid = ScaledSigmoid()

    def forward(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor,  # node_attrs here
            w: torch.Tensor, 
            edge_index: torch.Tensor,
            cutoff: torch.Tensor,
        ) -> torch.Tensor:

        num_nodes = x.size(0)
        num_edges = w.size(0)
        x = self.reshape_in(x) # [B, M, C]

        if self.use_both_Bi_Bj:
            x = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=-1)
        else:
            x = x[edge_index[0]]


        w = w.view(num_edges, self.num_components, -1)
        m_ij = self.so2_angular_basis.rotate(x)
        m_ij = self.layout.complex_mul(w, m_ij)
        alpha = self.linear_alpha(m_ij)


        gate = self.linear_gate(m_ij)
        gate = self.sigmoid(gate)
        m_ij = self.linear_up(m_ij)
 
        m_ij = self.layout2.complex_mul(gate, m_ij)


        m_ij = self.linear_down(m_ij)
        m_ij = m_ij.view(num_edges, -1, self.num_head, self.num_channel_per_head)
        M, H, C = m_ij.shape[1:]


        real_alpha = alpha

        # real_alpha, imag_alpha = torch.split(alpha, 1, dim=1)

        real_alpha = real_alpha.reshape(num_edges, self.num_head, self.num_channel_per_head)
        real_alpha = self.real_alpha_norm(real_alpha)
        real_alpha = self.alpha_act(real_alpha)
        real_alpha = torch.einsum('bik, ik -> bi', real_alpha, self.real_alpha_dot)
        real_alpha = self.attn_softmax(real_alpha, edge_index[1], num_nodes=num_nodes, exp_rescale=cutoff) # [edge, head]
        if cutoff is not None:
            real_alpha = real_alpha * cutoff
        real_alpha = real_alpha.view(num_edges, 1, self.num_head, 1)
        # real_alpha = real_alpha.expand(num_edges, M // 2, self.num_head, self.num_channel_per_head)

        # imag_alpha = imag_alpha.reshape(num_edges, self.num_head, self.num_channel_per_head)
        # imag_alpha = self.imag_alpha_norm(imag_alpha)
        # imag_alpha = self.alpha_act(imag_alpha)
        # imag_alpha = torch.einsum('bik, ik -> bi', imag_alpha, self.imag_alpha_dot)
        # imag_alpha = self.attn_softmax(imag_alpha, edge_index[1], num_nodes=num_nodes, exp_rescale=cutoff) # [edge, head]
        # if cutoff is not None:
        #     imag_alpha = imag_alpha * cutoff
        # imag_alpha = imag_alpha.view(num_edges, 1, self.num_head, 1)
        # imag_alpha = imag_alpha.expand(num_edges, M // 2, self.num_head, self.num_channel_per_head)


        alpha = real_alpha
        # alpha = torch.cat([real_alpha, imag_alpha], dim=1)


        # alpha = alpha.permute(0, 3, 1, 2).reshape(-1, M, H)
        # m_ij = m_ij.permute(0, 3, 1, 2).reshape(-1, M, H)

        # m_ij = self.layout3.complex_mul(alpha, m_ij)

        # m_ij = m_ij.view(num_edges, C, M, H).permute(0, 2, 3, 1)


        m_ij = alpha * m_ij

        m_ij = m_ij.reshape(num_edges, -1, self.num_channel)

        m_ij = self.so2_angular_basis.rotate_inv(m_ij)
        


        m_i = scatter_sum(
                m_ij, 
                edge_index[1], 
                dim=0, 
                dim_size=num_nodes,
        )

        return self.reshape_out.inverse(m_i)


# class SO2ScatterTensorProduct(torch.nn.Module):
#     def __init__(
#         self,
#         mmax: int,
#         lmax: int,
#         num_channel: int,
#         edge_wise_hidden: int,
#         num_head: int,
#         use_graph_softmax: bool,
#         is_so2_layout: bool,
#         use_both_Bi_Bj: bool,
#         use_so2_edge_ace: bool,
#         edge_nonlinear: Union[str, None],
#         num_elements: int,
#         so2_angular_basis: SO3Rotation,
#         reshape_in: LayoutTransform,
#         reshape_out: LayoutTransform,
#         scatter: Union[str, None],
#     ) -> None:
#         super().__init__()

#         self.mmax = mmax
#         self.lmax = lmax
#         self.num_channel = num_channel
#         self.edge_wise_hidden = edge_wise_hidden or self.num_channel
#         self.is_so2_layout = is_so2_layout
#         self.edge_nonlinear = edge_nonlinear
#         self.edge_nonlinear_type = self.edge_nonlinear.split('_')[-1]
#         self.use_both_Bi_Bj = use_both_Bi_Bj
#         self.use_so2_edge_ace = use_so2_edge_ace
#         self.scatter = scatter
#         self.num_out_channel = num_channel
#         self.so2_angular_basis = so2_angular_basis
#         self.reshape_in = reshape_in
#         self.reshape_out = reshape_out
#         self.num_head = num_head
#         self.use_graph_softmax = use_graph_softmax

#         Cin = num_channel * 2 if self.use_both_Bi_Bj else num_channel
#         Cout = num_channel

#         self.num_components, expand_index = so2_expand_index(self.mmax, self.lmax)
#         self.num_components = self.num_components * 2 - (self.lmax + 1)
#         self.weight_numel = self.num_components * Cin 
#         self.register_buffer('expand_index', expand_index, persistent=False)

#         self.num_gates = 0 
#         for m in range(mmax + 1):
#             self.num_gates += lmax + 1 

#         up_num_components_out = [lmax+1] + [lmax+1 for m in range(1, mmax+1)]
#         down_num_components_in = [lmax+1] + [lmax+1 for m in range(1, mmax+1)]

#         nonlinear_cls = SO2Gate
#         self.nonlinearity = nonlinear_cls(
#             mmax,
#             lmax,
#             self.edge_wise_hidden,   
#             channel_wise=True,
#         )
#         self.linear_gate = SO2Linear(
#             0,
#             lmax,
#             Cin,
#             self.edge_wise_hidden,     
#             num_components_out=[self.num_gates * 2 - (self.lmax+1)],
#         )
#         self.linear_up = SO2Linear(
#             mmax,
#             lmax,
#             Cin,
#             self.edge_wise_hidden,     
#             num_components_out=up_num_components_out,
#         )
#         self.linear_down = SO2Linear(
#             mmax,
#             lmax,
#             self.edge_wise_hidden,     
#             Cout,     
#             num_components_in=down_num_components_in,
#         )

#         self.ace = SO2EdgeProductBasis(
#             mmax,
#             lmax,
#             num_channel,
#             num_elements,
#             m1m2="<=",
#             agnostic=True
#         )

#         self.linear_alpha = SO2Linear(
#             0,
#             lmax,
#             Cin,
#             self.num_channel,  
#             num_components_out=[1],   
#         )


#         self.num_channel_per_head = self.num_channel // self.num_head
#         assert self.num_channel % self.num_head == 0

#         self.alpha_act = SmoothLeakyReLU()
#         std = 1.0 / math.sqrt(self.num_channel_per_head)
#         self.attn_softmax = GraphSoftmax()

#         self.real_alpha_norm = torch.nn.LayerNorm(self.num_channel_per_head)
#         self.real_alpha_dot = torch.nn.Parameter(torch.randn(self.num_head, self.num_channel_per_head))
#         torch.nn.init.uniform_(self.real_alpha_dot, -std, std)


#         self.layout = SO2ComplexMul(
#             mmax,
#             lmax,
#             Cin,
#             channel_wise=False,
#         )
#         self.layout2 = SO2ComplexMul(
#             mmax,
#             lmax,
#             self.num_channel,
#             channel_wise=True,
#         )

#         self.layout3 = SO2ComplexMul(
#             mmax,
#             lmax,
#             self.num_head,
#             channel_wise=True,
#         )

#         self.sigmoid = ScaledSigmoid()

#     def forward(
#             self, 
#             x: torch.Tensor, 
#             y: torch.Tensor,  # node_attrs here
#             w: torch.Tensor, 
#             edge_index: torch.Tensor,
#             cutoff: torch.Tensor,
#         ) -> torch.Tensor:

#         num_nodes = x.size(0)
#         num_edges = w.size(0)
#         x = self.reshape_in(x) # [B, M, C]

#         if self.use_both_Bi_Bj:
#             x = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=-1)
#         else:
#             x = x[edge_index[0]]


#         w = w.view(num_edges, self.num_components, -1)
#         m_ij = self.so2_angular_basis.rotate(x)



#         m_ij = self.layout.complex_mul(w, m_ij)


#         alpha = self.linear_alpha(m_ij)
#         gate = self.linear_gate(m_ij)
#         m_ij = self.linear_up(m_ij)


#         m_ij = self.ace(m_ij, y, edge_index)
 
#         m_ij = self.layout2.complex_mul(self.sigmoid(gate), m_ij)


#         m_ij = self.linear_down(m_ij)
#         m_ij = m_ij.view(num_edges, -1, self.num_head, self.num_channel_per_head)
#         M, H, C = m_ij.shape[1:]


#         # real_alpha, imag_alpha = torch.split(alpha, 1, dim=1)

#         real_alpha = alpha.reshape(num_edges, self.num_head, self.num_channel_per_head)
#         real_alpha = self.real_alpha_norm(real_alpha)
#         real_alpha = self.alpha_act(real_alpha)
#         real_alpha = torch.einsum('bik, ik -> bi', real_alpha, self.real_alpha_dot)
#         real_alpha = self.attn_softmax(real_alpha, edge_index[1], num_nodes=num_nodes, exp_rescale=cutoff) # [edge, head]
#         if cutoff is not None:
#             real_alpha = real_alpha * cutoff
#         real_alpha = real_alpha.view(num_edges, 1, self.num_head, 1)

#         # real_alpha = real_alpha.expand(num_edges, M // 2, self.num_head, self.num_channel_per_head)

#         # imag_alpha = imag_alpha.reshape(num_edges, self.num_head, self.num_channel_per_head)
#         # imag_alpha = self.imag_alpha_norm(imag_alpha)
#         # imag_alpha = self.alpha_act(imag_alpha)
#         # imag_alpha = torch.einsum('bik, ik -> bi', imag_alpha, self.imag_alpha_dot)
#         # imag_alpha = self.attn_softmax(imag_alpha, edge_index[1], num_nodes=num_nodes, exp_rescale=cutoff) # [edge, head]
#         # # if cutoff is not None:
#         # #     imag_alpha = imag_alpha * cutoff
#         # imag_alpha = imag_alpha.view(num_edges, 1, self.num_head, 1)
#         # imag_alpha = imag_alpha.expand(num_edges, M // 2, self.num_head, self.num_channel_per_head)


#         # alpha = torch.cat([real_alpha, imag_alpha], dim=1)


#         # alpha = alpha.permute(0, 3, 1, 2).reshape(-1, M, H)
#         # m_ij = m_ij.permute(0, 3, 1, 2).reshape(-1, M, H)

#         # m_ij = self.layout3.complex_mul(alpha, m_ij)

#         # m_ij = m_ij.view(num_edges, C, M, H).permute(0, 2, 3, 1)

#         m_ij = real_alpha * m_ij
#         m_ij = m_ij.reshape(num_edges, -1, self.num_channel)

#         if cutoff is not None:
#             m_ij = m_ij * cutoff.unsqueeze(-1)




#         m_ij = self.so2_angular_basis.rotate_inv(m_ij)
        


#         m_i = scatter_sum(
#                 m_ij, 
#                 edge_index[1], 
#                 dim=0, 
#                 dim_size=num_nodes,
#         )

#         return self.reshape_out.inverse(m_i)