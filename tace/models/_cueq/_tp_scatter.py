################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch
from e3nn import o3

try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet
    from cuequivariance.group_theory.experimental.e3nn import O3_e3nn
except Exception:
    pass


from .paths import generate_cueq_paths


class e3nnCueqTensorProduct(torch.nn.Module):
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

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2 
        self.irreps_out = irreps_out

        self.cueq_tp = cuet.SegmentedPolynomial(
            generate_cueq_paths(
                cue.Irreps(O3_e3nn, irreps_in1),
                cue.Irreps(O3_e3nn, irreps_in2),
                cue.Irreps(O3_e3nn, irreps_out),
                l1l2=l1l2,
                l2l3=l2l3,
                l3l1=l3l1,
                ictp_ictc_like=ictp_ictc_like,
            )
            .flatten_coefficient_modes()
            .squeeze_modes()
            .polynomial,
            method="fused_tp",
            math_dtype=torch.get_default_dtype(),
        )

        self.reshape1 = cuet.TransposeIrrepsLayout(
            irreps_in1, source=cue.mul_ir, target=cue.ir_mul
        )
        self.reshape2 = cuet.TransposeIrrepsLayout(
            irreps_out, source=cue.ir_mul, target=cue.mul_ir
        )

    def forward(self, x, y, edge_index, w):
        return self.reshape2(
            self.cueq_tp(
                [w, self.reshape1(x), y],
                {1: edge_index[0]},
                {0: x},
                {0: edge_index[1]},
            )[0]
        )