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
    SO3Rotation, SO2Linear, SO2Gate,
    so2_expand_index, so3_expand_index,
)
from .paths import generate_paths
from .edge_prod import SO2EdgeProductBasis


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
                num_channel=irreps_in2.count("0e"),
                path=instructions,
            )
        else:
            pass

    def forward(
            self, x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            if hasattr(self, "fused_tp"):
                return self.fused_tp(x, y)
            return self.tp(x, y)


class SO2ScatterTensorProduct(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        num_hidden_channel: int,
        is_scalar_tp: bool,
        is_so2_layout: bool,
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
        self.num_hidden_channel = num_hidden_channel or self.num_channel
        self.is_so2_layout = is_so2_layout
        self.is_scalar_tp = is_scalar_tp
        self.edge_nonlinear = edge_nonlinear
        self.use_so2_edge_ace = use_so2_edge_ace
        self.scatter = scatter

        self.so2_angular_basis = so2_angular_basis
        self.reshape_in = reshape_in
        self.reshape_out = reshape_out
        

        Cin = num_channel if not self.use_so2_edge_ace else num_channel * 2
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

            if self.use_so2_edge_ace:
                self.ace = SO2EdgeProductBasis(
                    mmax, 
                    lmax, 
                    self.num_hidden_channel,
                    num_elements=num_elements,
                    m1m2='<='
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

        if self.use_so2_edge_ace:
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
            m_ij = self.so2_angular_basis.rotate_inv(m_ij)

        if cutoff is not None:
            m_ij = m_ij * cutoff.unsqueeze(-1)


        if self.scatter is None:
            return m_ij
        
        m_i = scatter_sum(
                m_ij, 
                edge_index[1], 
                dim=0, 
                dim_size=num_nodes,
        )
        return self.reshape_out.inverse(m_i)
    

