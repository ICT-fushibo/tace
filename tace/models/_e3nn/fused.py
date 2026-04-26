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
from ..so2 import SO3Rotation, SO2Linear, SO2Gate, so2_expand_index, so3_expand_index
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


class SO2ScatterTensorProduct(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        is_scalar_tp: bool,
        is_so2_layout: bool,
        edge_nonlinear: str | None,
        so2_angular_basis: SO3Rotation,
        reshape_in: LayoutTransform,
        reshape_out: LayoutTransform,
    ) -> None:
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.is_so2_layout = is_so2_layout
        self.is_scalar_tp = is_scalar_tp
        self.edge_nonlinear = edge_nonlinear

        self.so2_angular_basis = so2_angular_basis
        self.reshape_in = reshape_in
        self.reshape_out = reshape_out

        if self.is_so2_layout and not self.is_scalar_tp:
            self.num_components, expand_index = so2_expand_index(self.mmax, self.lmax)
            self.weight_numel = self.num_components * self.num_channel
            self.register_buffer('expand_index', expand_index, persistent=False)
        else:
            self.num_components, expand_index = so3_expand_index(self.mmax, self.lmax)
            self.weight_numel = self.num_components * self.num_channel
            self.register_buffer('expand_index', expand_index, persistent=False)

        # assert not is_scalar_tp, "To simplify the implementation, we enforce a constraint when using the SO(2) module: "
        # "the input node features must be equivariant. This implies that you should use node embedding with l > 0, or the "
        # "SO(2) interaction is applied from the second layer."

        if self.is_scalar_tp:
            pass
        else:
            num_gates = 0
            for m in range(mmax + 1):
                num_gates += lmax + 1 -m
            num_gates = num_gates * self.num_channel
            self.linear_up = SO2Linear(
                mmax,
                lmax,
                num_channel,
                num_channel,
                extra_m0_out_channels=num_gates,
            )
            self.nonlinearity = SO2Gate(
                mmax,
                lmax,
                num_channel,        
            )
            self.linear_down = SO2Linear(
                mmax,
                lmax,
                num_channel,
                num_channel,
            )

    def forward(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor,  # node_attrs here
            w: torch.Tensor, 
            edge_index: torch.Tensor
        ) -> torch.Tensor:

        num_nodes = x.size(0)
        num_edges = w.size(0)
        x = self.reshape_in(x) 

        if self.is_scalar_tp:
            w = w.view(num_edges, self.num_components, -1)
            m_ij = torch.einsum(
                'bij, bjc -> bic', 
                    self.so2_angular_basis.wigner_inv.narrow(2, 0, (self.lmax + 1)),
                    x[edge_index[0]] * w
            ) # first so3 tp, no nonlinearity is required here
        else:
            w = w.view(num_edges, self.num_components, -1)
            w = torch.index_select(w, dim=1, index=self.expand_index)
            if self.is_so2_layout:
                m_ij = self.so2_angular_basis.rotate(x[edge_index[0]])
                m_ij = m_ij * w
            else:
                m_ij =  x[edge_index[0]] * w
                m_ij = self.so2_angular_basis.rotate(m_ij)
  
            m_ij, gate = self.linear_up(m_ij) # m_ij: [edge, so2_m, C]
            m_ij = self.nonlinearity(m_ij, gate)      
            m_ij = self.linear_down(m_ij)
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

