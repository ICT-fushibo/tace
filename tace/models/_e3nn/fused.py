################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import List


import torch
from tace.utils.torch_scatter import scatter_sum
from e3nn import o3


from ..layout import LayoutTransform
from ..env import TACE_USE_OEQ, TACE_USE_CUE, TACE_USE_EQT
from .paths import generate_paths
from ..so2 import (
    SO3Rotation, SO2Linear, SO2Gate, SO2TensorProduct,
    so2_expand_index, so3_expand_index,
)
from ..softmax import GraphSoftmax
try:
    from .._oeq import e3nnOeqScatterTensorProduct
except Exception:
    pass
try:
    from .._cue import e3nnCueScatterTensorProduct
except Exception:
    pass
try:
    from .._eqt import e3nnEqtTensorProduct
except Exception:
    pass


class O3ScatterTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        l1l2: str | None = None,
        l2l3: str | None = None,
        l3l1: str | None = None,
        ictp_ictc_like: bool = True,
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
            ictp_ictc_like=ictp_ictc_like,
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
        self.use_oeq = TACE_USE_OEQ == '1'
        self.use_cue = TACE_USE_CUE == '1'
        # assert not (self.use_oeq & self.use_cue)

        if self.use_oeq:
            self.fused_tp = e3nnOeqScatterTensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out,
                instructions=self.instructions,
            )
        elif self.use_cue:
            self.fused_tp = e3nnCueScatterTensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out,
                l1l2=l1l2,
                l2l3=l2l3,
                l3l1=l3l1,
                ictp_ictc_like=ictp_ictc_like,
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


class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope


    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2


    def extra_repr(self):
        return 'negative_slope={}'.format(self.alpha)

class SO2EdgeProductBasis(torch.nn.Module):

    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channels: int,
        num_elements: int,
        m1m2: str | None = '<=',
        internal_weights: bool = False,
    ):
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_components = lmax+1
        self.num_channel = num_channels

        self.ace = SO2TensorProduct(
            mmax, 
            lmax,
            num_channels, 
            m1m2=m1m2, 
            internal_weights=internal_weights
        )
        self.weight_numel = self.ace.weight_numel

        self.num_c1_weight = (mmax+1) * (lmax+1) * num_channels

        self.weight_numel += self.num_c1_weight

        self.source_coefs = torch.nn.Parameter(
            torch.randn(num_elements, self.weight_numel)
        )
        self.target_coefs = torch.nn.Parameter(
            torch.randn(num_elements, self.weight_numel)
        )
        self.source_coefs.data.mul_(1 / math.sqrt(2))
        self.target_coefs.data.mul_(1 / math.sqrt(2))


        expand_index = []
        offset = 0
        for m in range(mmax + 1):
            index = torch.arange((lmax + 1))
            index = index + offset
            expand_index.append(index)
            if m > 0:
                expand_index.append(index)    # +- m
            offset = offset + len(index)
        expand_index = torch.cat(expand_index, dim=0)
        expand_index = expand_index.long()
        self.num_components = offset
        self.register_buffer('expand_index', expand_index, persistent=False)

    def forward(self, x, y, edge_index) -> torch.Tensor:

        B = x.size(0)
        C = self.num_channel

        node_type = y.argmax(dim=-1)
        src_type = node_type[edge_index[0]]
        dst_type = node_type[edge_index[1]]
        ws = (
            self.source_coefs[src_type]
            + self.target_coefs[dst_type]
        )
        w1 = ws.narrow(1, 0, self.num_c1_weight)
        w1 = w1.view(B, -1, C)
        w1 = torch.index_select(w1, dim=1, index=self.expand_index)
        corr_feats1 = x * w1
        w2 = ws.narrow(1, self.num_c1_weight, self.weight_numel - self.num_c1_weight)
        corr_feats2 = self.ace(x, x, w2)

        return corr_feats1 + corr_feats2
   

    def extra_repr(self) -> str:
        p = {
            0: 'e',
            1: 'o',
        }
        irreps = []
        for m in range(self.mmax + 1):
            irreps.append(f"{self.num_channel*(self.lmax+1)}x{m}{p[m % 2]}")
        num_weights = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return (
            f"{self.__class__.__name__}"
            f"({'+'.join(irreps)} x {'+'.join(irreps)} -> "
            f"{'+'.join(irreps)} | "
            f"{num_weights} weights)"
        )


class SO2ScatterTensorProduct(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        num_hidden_channel: int,
        num_head: int | None,
        num_channel_per_head: int,
        is_scalar_tp: bool,
        is_so2_layout: bool,
        use_so2_edge_ace: bool,
        edge_nonlinear: str | None,
        num_elements: int,
        so2_angular_basis: SO3Rotation,
        reshape_in: LayoutTransform,
        reshape_out: LayoutTransform,
    ) -> None:
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.num_hidden_channel = num_hidden_channel or self.num_channel
        self.is_so2_layout = is_so2_layout
        self.is_scalar_tp = is_scalar_tp
        self.edge_nonlinear = edge_nonlinear
        self.use_so2_edge_ace = use_so2_edge_ace

        self.so2_angular_basis = so2_angular_basis
        self.reshape_in = reshape_in
        self.reshape_out = reshape_out

        # Transformer
        self.num_head = num_head or 1
        self.num_channel_per_head = num_channel_per_head or num_channel
        assert self.num_hidden_channel % self.num_head == 0
        if self.num_head > 1:
            self.use_transformer = True
        else:
            self.use_transformer = False

        Cin = num_channel if not self.use_transformer else num_channel * 2
        Cout = num_channel
        self.num_out_channel = Cout

        
        if self.is_so2_layout and not self.is_scalar_tp:
            self.num_components, expand_index = so2_expand_index(self.mmax, self.lmax)
            self.weight_numel = self.num_components * Cin
            self.register_buffer('expand_index', expand_index, persistent=False)
        else:
            self.num_components, expand_index = so3_expand_index(self.mmax, self.lmax)
            self.weight_numel = self.num_components * Cin
            self.register_buffer('expand_index', expand_index, persistent=False)

        self.num_gates = 0
        for m in range(mmax + 1):
            if self.use_so2_edge_ace:
                self.num_gates += lmax + 1
            else:
                self.num_gates += lmax + 1 -m

        if self.is_scalar_tp:
            pass
        else:
            assert edge_nonlinear is not None, "We force to use SO2 edge nonlinear in TACE"

            if self.use_transformer: # TODO
                self.linear_alpha = SO2Linear(
                    0, 
                    lmax, 
                    Cin, 
                    num_head * num_channel_per_head,
                    num_components_out=[1]
                )
                self.alpha_norm = torch.nn.LayerNorm(self.num_channel_per_head)
                self.alpha_act = SmoothLeakyReLU()
                self.alpha_dot = torch.nn.Parameter(torch.randn(self.num_head, self.num_channel_per_head))
                std = 1.0 / math.sqrt(self.num_channel_per_head)
                torch.nn.init.uniform_(self.alpha_dot, -std, std)
                self.attn_softmax = GraphSoftmax()

            if self.use_so2_edge_ace:
                self.ace = SO2EdgeProductBasis(
                    mmax, 
                    lmax, 
                    self.num_hidden_channel,
                    num_elements=num_elements,
                )
                self.linear_up = SO2Linear(
                    mmax,
                    lmax,
                    Cin,
                    self.num_hidden_channel,     
                    num_components_in=None,
                    num_components_out=[self.num_gates + lmax+1] + [lmax+1] * (lmax),
                )
                self.nonlinearity = SO2Gate(
                    mmax,
                    lmax,
                    self.num_hidden_channel,     
                    channel_wise=True
                )
                self.linear_down = SO2Linear(
                    mmax,
                    lmax,
                    self.num_hidden_channel,     
                    Cout,     
                    num_components_in=[lmax+1] * (lmax+1),
                    num_components_out=None,
                )
            else:
                self.linear_up = SO2Linear(
                    mmax,
                    lmax,
                    Cin,
                    self.num_hidden_channel,    
                    num_components_in=None,
                    num_components_out=[self.num_gates + lmax+1] + [lmax+1-m for m in range(1, mmax+1)],
                )
                self.nonlinearity = SO2Gate(
                    mmax,
                    lmax,
                    self.num_hidden_channel,    
                    channel_wise=False
                )
                self.linear_down = SO2Linear(
                    mmax,
                    lmax,
                    self.num_hidden_channel,    
                    Cout,     
                )     


    def forward(
            self, 
            x: torch.Tensor, # [B, so_m, C]
            y: torch.Tensor,  # node_attrs here
            w: torch.Tensor, 
            edge_index: torch.Tensor,
            cutoff: torch.Tensor,
        ) -> torch.Tensor:

        num_nodes = x.size(0)
        num_edges = w.size(0)
        x = self.reshape_in(x) 

        if self.use_transformer:
            x = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=-1)
        else:
            x = x[edge_index[0]]

        if self.is_scalar_tp:
            w = w.view(num_edges, self.num_components, -1)
            m_ij = torch.einsum(
                'bij, bjc -> bic', 
                    self.so2_angular_basis.wigner_inv.narrow(2, 0, (self.lmax + 1)),
                    x * w
            ) # first so3 tp, no nonlinearity is required here
        else:
            w = w.view(num_edges, self.num_components, -1)
            w = torch.index_select(w, dim=1, index=self.expand_index)

            if self.is_so2_layout:
                m_ij = self.so2_angular_basis.rotate(x)
                m_ij = m_ij * w
            else:
                m_ij =  x * w
                m_ij = self.so2_angular_basis.rotate(m_ij)

            if self.use_transformer:
                alpha = self.linear_alpha(m_ij)
            m_ij = self.linear_up(m_ij)

            gate = m_ij.narrow(1, 0, self.num_gates)
            m_ij = m_ij.narrow(
                1,
                self.num_gates,
                m_ij.size(1) - self.num_gates
            )
            if hasattr(self, 'ace'):
                m_ij = self.ace(m_ij, y, edge_index)
            m_ij = self.nonlinearity(m_ij, gate) 
            m_ij = self.linear_down(m_ij)
            
            if self.use_transformer:
                alpha = alpha.reshape(-1, self.num_head, self.num_channel_per_head)
                alpha = self.alpha_norm(alpha)
                alpha = self.alpha_act(alpha)
                alpha = torch.einsum('bik, ik -> bi', alpha, self.alpha_dot)
                alpha = self.attn_softmax(alpha, edge_index[1], num_nodes=num_nodes, exp_rescale=cutoff)
                if cutoff is not None:
                    alpha = alpha * cutoff
                alpha = alpha.view(alpha.size(0), 1, self.num_head, 1)
                attn = m_ij
                attn = attn.view(attn.size(0), attn.size(1), self.num_head, -1)
                attn = attn * alpha
                attn = attn.view(attn.size(0), attn.size(1), -1)
                m_ij = attn

            m_ij = self.so2_angular_basis.rotate_inv(m_ij)

        m_i = scatter_sum(
                m_ij, 
                edge_index[1], 
                dim=0, 
                dim_size=num_nodes,
        )
        return self.reshape_out.inverse(m_i)
    

class uuuTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        l1l2: str | None = None,
        l2l3: str | None = None,
        l3l1: str | None = None,
        l3s: List[int] | None = None,
        ictp_ictc_like: bool = True,
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
            l3s=l3s,
            ictp_ictc_like=ictp_ictc_like,
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
        self.trainable = trainable
        self.use_eqt = TACE_USE_EQT == '1'
        # self.use_oeq = TACE_USE_OEQ == '1'
        # self.use_cue = TACE_USE_CUE == '1'

        if self.use_eqt:
            self.fused_tp = e3nnEqtTensorProduct(
                irreps_in1=irreps_in1,
                irreps_in2=irreps_in2,
                irreps_out=actual_irreps_out,
                num_channel=irreps_in1.count("0e"),
                path=instructions,
                trainable=trainable,
            )
        else:
            pass

    def forward(
            self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None = None
        ) -> torch.Tensor:
            if hasattr(self, "fused_tp"):
                return self.fused_tp(x, y, w)
            return self.tp(x, y, w)

