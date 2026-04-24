################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import List


import torch
from tace.utils.torch_scatter import scatter_sum
from e3nn import o3


from ..layout import LayoutTransform
from ..env import TACE_USE_OEQ, TACE_USE_CUE, TACE_USE_EQT
from .paths import generate_paths
from .._so2 import SO3Rotation, SO2Linear
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
    from .._eqt.equitorch.nn import SO2TensorProduct
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

    
# class SO2ScatterTensorProduct(torch.nn.Module):
#     def __init__(
#         self,
#         irreps_in1: o3.Irreps,
#         irreps_in2: o3.Irreps,
#         irreps_out: o3.Irreps,
#         l1l2: str | None = None,
#         l2l3: str | None = None,
#         l3l1: str | None = None,
#         ictp_ictc_like: bool = True,
#         edge_nonlinear: str | None = None,
#     ) -> None:
#         super().__init__()

#         self.irreps_in1 = irreps_in1
#         self.irreps_in2 = irreps_in2
#         self.irreps_out = irreps_out

#         from ..so2_kernel import SO2TensorProduct

#         self.tp = SO2TensorProduct(
#             irreps_in="+".join(str(ir) for _, ir in self.irreps_in1), 
#             irreps_out="+".join(str(ir) for _, ir in self.irreps_out), 
#             channels_in=irreps_in1.count("0e"), 
#             channels_out=irreps_out.count("0e"), 
#             internal_weights=False,
#             # feature_mode='uu',
#             path_norm=True,
#             channel_norm=False, 
#             path=None,
#         )
#         self.weight_numel = self.tp.weight_numel

#         self.resahpe_in1 = LayoutTransform(self.irreps_in1)
#         self.resahpe_out = LayoutTransform(self.irreps_out)

#     def forward(
#             self, 
#             x: torch.Tensor, 
#             y: torch.Tensor, 
#             w: torch.Tensor, 
#             edge_index: torch.Tensor
#         ) -> torch.Tensor:

#         x = self.resahpe_in1(x)
#         num_nodes = x.size(0)
#         is_0e_only = x.size(1) == 1
#         x = x[edge_index[0]]
#         if not is_0e_only:
#             x = torch.einsum('bnm, bmc -> bnc', y, x)
#         out = self.tp(x, weight=w)
#         out = torch.einsum('bmn, bmc -> bnc', y, out)
#         out = scatter_sum(
#                 out, 
#                 edge_index[1], 
#                 dim=0, 
#                 dim_size=num_nodes,
#         )
#         out = self.resahpe_out.inverse(out)
#         return out


class SO2ScatterTensorProduct(torch.nn.Module):
    def __init__(
        self,
        angular_basis: SO3Rotation,
        reshape_in: LayoutTransform,
        reshape_out: LayoutTransform,
        mmax: int,
        lmax: int,
        num_channel_in: int,
        num_channel_out: int,
        num_channel_hidden: int | None = None,
        is_so2_layout: bool = False,
        edge_nonlinear: str | None = None,
    ) -> None:
        super().__init__()

        self.angular_basis = angular_basis
        self.reshape_in = reshape_in
        self.reshape_out = reshape_out
        self.edge_nonlinear = edge_nonlinear

        self.mmax = mmax
        self.lmax = lmax
        self.num_channel_in = num_channel_in
        self.num_channel_hidden = num_channel_hidden
        self.num_channel_out = num_channel_out
        self.is_so2_layout = is_so2_layout

        if self.is_so2_layout:
            self.weight_numel = 0
            for m in range(self.mmax + 1):
                self.weight_numel += (self.lmax + 1 - m)
            self.weight_numel *= self.num_channel_in
        else:
            self.weight_numel = self.num_channel_in * (self.lmax + 1)

        if self.num_channel_hidden is not None:
            self.linear1 = SO2Linear(
                num_channel_in,
                num_channel_hidden,
                lmax,
                mmax,
                extra_m0_out_channels=None,
            )
            self.linear2 = SO2Linear(
                num_channel_hidden,
                num_channel_out,
                lmax,
                mmax,
                extra_m0_out_channels=None,
            )
        else:
            self.linear1 = SO2Linear(
                num_channel_in,
                num_channel_out,
                lmax,
                mmax,
                extra_m0_out_channels=None,
            )

        if self.is_so2_layout: 
            expand_index = []
            offset = 0
            for m in range(self.mmax + 1):
                index = torch.arange((self.lmax + 1 - m))
                index = index + offset
                expand_index.append(index)
                if m > 0:
                    expand_index.append(index)    # +- m
                offset = offset + len(index)
            expand_index = torch.cat(expand_index, dim=0)
            expand_index = expand_index.long()
            self.register_buffer('expand_index', expand_index, persistent=False,)
            self.num_m_components = offset
            assert self.weight_numel % self.num_m_components == 0
        else:
            self.mmax = self.lmax
            assert self.lmax == self.mmax
            expand_index = torch.zeros([((self.lmax + 1) ** 2)]).long()
            start_idx = 0
            for l in range(self.lmax + 1):
                length = 2 * l + 1
                expand_index[start_idx : (start_idx + length)] = l
                start_idx = start_idx + length
            self.register_buffer('expand_index', expand_index, persistent=False,)
            assert self.weight_numel % (self.lmax + 1) == 0


    def forward(
            self, 
            x: torch.Tensor, 
            y: None, 
            w: torch.Tensor, 
            edge_index: torch.Tensor
        ) -> torch.Tensor:
        num_nodes = x.size(0)
        num_edges = w.size(0)
        if self.is_so2_layout:
            w = w.view(num_edges, self.num_m_components, -1)
        else:
            w = w.view(num_edges, (self.lmax + 1), -1)
        w = torch.index_select(w, dim=1, index=self.expand_index)
        x = self.reshape_in(x) 
        m_ij =  x[edge_index[0]] * w
        m_ij = self.angular_basis.rotate(m_ij)
        m_ij = self.linear1(m_ij)
        # edge nonlinear 
        if hasattr(self, "linear2"):
            m_ij = self.linear2(m_ij)
        m_ij = self.angular_basis.rotate_inv(m_ij)
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

