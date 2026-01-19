################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional


import torch
from e3nn import o3
try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet
except Exception:
    pass

try:
    import openequivariance as oeq
except Exception:
    pass


def cueq_irreps(irreps: o3.Irreps):
    return [(mul, ir.l) for mul, ir in irreps]


class FusedTensorProduct:
    """
    TACE needs to restrict l1, l2, l3 simultaneously,
    while cueq not support now, recommend using oeq.
    """
    def __new__(
        cls,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        instructions: Optional[list] = None,
        shared_weights: bool = False,
        internal_weights: bool = False,
        enable_e3nn: bool = True,
        enable_oeq: bool = False,
        enable_cueq: bool = False,
    ):
        if enable_oeq:
            return cls._build_oeq_module(
                irreps_in1,
                irreps_in2,
                irreps_out,
                instructions,
                shared_weights,
                internal_weights,
            )
        # if enable_cueq:
        #     return cls._build_cueq_module(
        #         irreps_in1,
        #         irreps_in2,
        #         irreps_out,
        #         instructions,
        #         shared_weights,
        #         internal_weights,
        #     )
        
        return cls._build_e3nn_module(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
            shared_weights,
            internal_weights,
        )

    @classmethod
    def _build_oeq_module(
        cls,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        shared_weights,
        internal_weights,
    ):
        dtype = oeq.torch_to_oeq_dtype(torch.get_default_dtype())
        tpp = oeq.TPProblem(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            irrep_dtype=dtype,
            weight_dtype=dtype,
        )
        return oeq.TensorProductConv(
            tpp, 
            deterministic=False, 
            kahan=False, 
            torch_op=True, 
            use_opaque=False,
        )

    # @classmethod
    # def _build_cueq_module(
    #     cls,
    #     irreps_in1,
    #     irreps_in2,
    #     irreps_out,
    #     instructions,
    #     shared_weights,
    #     internal_weights,
    #     group: str = "SO3",
        
    # ): 
    #     return cuet.ChannelWiseTensorProduct(
    #         cue.Irreps(group, cueq_irreps(irreps_in1)),
    #         cue.Irreps(group, cueq_irreps(irreps_in2)),
    #         cue.Irreps(group, cueq_irreps(irreps_out)),
    #         layout=cue.mul_ir,
    #         shared_weights=shared_weights,
    #         internal_weights=internal_weights,
    #         dtype=torch.get_default_dtype(),
    #         math_dtype=torch.get_default_dtype(),
    #     )

    @staticmethod
    def _build_e3nn_module(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        shared_weights,
        internal_weights,
    ):
        return o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            irreps_out,
            instructions=instructions,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )

class AccLinear:
    """No obvious acceleration in linear layer, use e3nn is enough"""
    def __new__(
        cls,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        shared_weights: bool = True,
        internal_weights: bool = True,
        enable_e3nn: bool = True,
        enable_cueq: bool = False,
    ):
        # if enable_cueq:
        #     return cls._build_cueq_module(
        #         irreps_in,
        #         irreps_out,
        #         shared_weights,
        #         internal_weights,
        #     )

        return cls._build_e3nn_module(
            irreps_in,
            irreps_out,
            shared_weights,
            internal_weights,
        )

    @staticmethod
    def _build_cueq_module(
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool,
        internal_weights: bool = True,
        group: str = "SO3",
    ):
        return cuet.Linear(
            cue.Irreps(group, cueq_irreps(irreps_in)),
            cue.Irreps(group, cueq_irreps(irreps_out)),
            layout=cue.mul_ir,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            method="naive",
        )

    @staticmethod
    def _build_e3nn_module(
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool,
        internal_weights: bool,
    ):
        return o3.Linear(
            irreps_in,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )

class AccElementLinear(torch.nn.Module):
    """Not allow bias for cueq"""
    def __init__(
        self,
        num_elements: int,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.linear = AccLinear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            shared_weights = False,
            internal_weights = False,
        )        
        self.weight = torch.nn.Parameter(
            torch.randn(num_elements, self.linear.weight_numel)
        )

    def forward(self, x: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
        # a = Cc
        W = torch.einsum("bz, za -> ba", onehot, self.weight)
        return self.linear(x, W)
