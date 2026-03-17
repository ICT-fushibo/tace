################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Tuple


import torch
import equitorch as eqt
from e3nn import o3
try:
    import openequivariance as oeq
except Exception:
    pass


from ..layout import LayoutTransform

def _get_str_irreps(irreps: eqt.irreps.Irreps):
    s = str(irreps)
    return s[s.find("(") + 1 : s.rfind(")")]


class eqtOeqTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: eqt.irreps.Irreps,
        irreps_in2: eqt.irreps.Irreps,
        irreps_out: eqt.irreps.Irreps,
        e3nn_irreps_out: o3.Irreps,
        num_channel: int,
        instructions: Tuple,
    ):
        super().__init__()

        self.irreps_in1 = eqt.irreps.check_irreps(irreps_in1)
        self.irreps_in2 = eqt.irreps.check_irreps(irreps_in2)
        self.irreps_out = eqt.irreps.check_irreps(irreps_out)

        # === e3nn to oeq ===
        e3nn_irreps_in1 = (o3.Irreps(_get_str_irreps(irreps_in1)) * num_channel).regroup()
        e3nn_irreps_in2 = o3.Irreps(_get_str_irreps(irreps_in2)).regroup()
        dtype = oeq.torch_to_oeq_dtype(torch.get_default_dtype())
        self.reshape1 = LayoutTransform(e3nn_irreps_in1)
        self.reshape2 = LayoutTransform(e3nn_irreps_out)
        tpp = oeq.TPProblem(
            e3nn_irreps_in1,
            e3nn_irreps_in2,
            e3nn_irreps_out,
            instructions,
            shared_weights=False,
            internal_weights=False,
            irrep_dtype=dtype,
            weight_dtype=dtype,
        )
        self.oeq_tp = oeq.TensorProductConv(
            tpp, 
            deterministic=False, 
            kahan=False, 
            torch_op=True, 
            use_opaque=False,
        )
        self.weight_numel = self.oeq_tp.weight_numel

    def forward(
            self, 
            node_feats: torch.Tensor,
            edge_attrs: torch.Tensor,
            conv_weights: torch.Tensor,
            edge_index: torch.Tensor,
        ) -> torch.Tensor:
        node_feats = self.reshape1.inverse(node_feats)
        message = self.oeq_tp(
            node_feats, edge_attrs, conv_weights, edge_index[1], edge_index[0]
        ) # target, source
        return self.reshape2(message)
    

class e3nnOeqTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: Tuple,
    ):
        super().__init__()

        dtype = oeq.torch_to_oeq_dtype(torch.get_default_dtype())
        tpp = oeq.TPProblem(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
            shared_weights=False,
            internal_weights=False,
            irrep_dtype=dtype,
            weight_dtype=dtype,
        )
        self.oeq_tp = oeq.TensorProductConv(
            tpp, 
            deterministic=False, 
            kahan=False, 
            torch_op=True, 
            use_opaque=False,
        )
        self.weight_numel = self.oeq_tp.weight_numel
        self.irreps_out = irreps_out

    def forward(
            self, 
            node_feats: torch.Tensor,
            edge_attrs: torch.Tensor,
            conv_weights: torch.Tensor,
            edge_index: torch.Tensor,
        ) -> torch.Tensor:
        return self.oeq_tp(
            node_feats, edge_attrs, conv_weights, edge_index[1], edge_index[0]
        ) # target, source