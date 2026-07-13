################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Union

import torch
from e3nn import o3

try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet
except Exception:
    pass

from .paths import generate_cueq_uuu_paths


class e3nnCueTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        l1l2: Union[str, None] = None,
        l2l3: Union[str, None] = None,
        l3l1: Union[str, None] = None,
        trainable: bool = False,
    ) -> None:
        super().__init__()

        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.target_irreps_out = o3.Irreps(irreps_out)
        self.trainable = trainable

        descriptor, actual_irreps_out = generate_cueq_uuu_paths(
            self.target_irreps_out,
            self.irreps_in1,
            self.irreps_in2,
            l1l2=l1l2,
            l2l3=l2l3,
            l3l1=l3l1,
        )
        self.irreps_out = actual_irreps_out
        self._cue_weight_numel = descriptor.inputs[0].dim
        self.weight_numel = self._cue_weight_numel if trainable else 0
        self.cueq_tp = cuet.SegmentedPolynomial(
            descriptor.flatten_coefficient_modes().squeeze_modes().polynomial,
            method="uniform_1d",
            math_dtype=torch.get_default_dtype(),
        )
        self.reshape1 = cuet.TransposeIrrepsLayout(
            self.irreps_in1,
            source=cue.mul_ir,
            target=cue.ir_mul,
        )
        self.reshape2 = cuet.TransposeIrrepsLayout(
            self.irreps_in2,
            source=cue.mul_ir,
            target=cue.ir_mul,
        )
        self.reshape3 = cuet.TransposeIrrepsLayout(
            self.irreps_out,
            source=cue.ir_mul,
            target=cue.mul_ir,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        ws: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if self.trainable:
            if ws is None:
                raise ValueError("cueq trainable uuu tensor product requires weights")
            weights = ws
        else:
            weights = x.new_ones(x.shape[0], self._cue_weight_numel)
        return self.reshape3(
            self.cueq_tp([weights, self.reshape1(x), self.reshape2(y)])[0]
        )
