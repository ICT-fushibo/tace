################################################################################
# Authors: Zemin Xu 
# License: MIT, see LICENSE.md
################################################################################

from math import sqrt
from typing import Tuple


import torch
import opt_einsum_fx


from .paths import generate_path

PATH_ = 4
BATCH_ = 5
CHANNEL_ = 6

class InterEinsumTC(torch.nn.Module):
    '''Reducible tensor product and tensor contraction'''
    def __init__(self, comb: Tuple) -> None:
        super().__init__()
        l1, l2, _, k = comb
        self.comb = comb
        self.expr = generate_path(*comb, True)
        ctr = torch.fx.symbolic_trace(lambda T1, T2: torch.einsum(self.expr, T1, T2))
        self.ctr = (
            opt_einsum_fx.optimize_einsums_full(
                model=ctr,
                example_inputs=(
                    torch.randn([BATCH_] + [CHANNEL_] + [3] * l1),
                    torch.randn([BATCH_] + [CHANNEL_] + [3] * l2),
                ),
            )
        )
        self.normalizer = 1.0 / sqrt(3**k)

    def forward(self, T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
        return self.ctr(T1, T2) * self.normalizer

    def __repr__(self):
        return f"{self.__class__.__name__}(path={self.expr}, comb={self.comb})"


class ProdEinsumTC(torch.nn.Module):
    '''Reducible tensor product and tensor contraction'''
    def __init__(self, comb: Tuple) -> None:
        super().__init__()
        l1, l2, _, k = comb
        self.comb = comb
        expr = generate_path(*comb, True)
        inputs, output = expr.split("->")
        in1, in2 = [x.strip() for x in inputs.split(",")]
        in1 = "a" + in1
        output = "a" + output
        self.expr = in1 + "," + in2 + "->" + output
        ctr = torch.fx.symbolic_trace(lambda T1, T2: torch.einsum(self.expr, T1, T2))
        self.ctr = (
            opt_einsum_fx.optimize_einsums_full(
                model=ctr,
                example_inputs=(
                    torch.randn([PATH_] + [BATCH_] + [CHANNEL_] + [3] * l1),
                    torch.randn([BATCH_] + [CHANNEL_] + [3] * l2),
                ),
            )
        )
        self.normalizer = 1.0 / sqrt(3**k)

    def forward(self, T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
        return self.ctr(T1, T2) * self.normalizer

    def __repr__(self):
        return f"{self.__class__.__name__}(path={self.expr}, comb={self.comb})"

