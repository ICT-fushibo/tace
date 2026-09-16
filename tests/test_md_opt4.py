"""CPU contract checks for the TACE Opt4 route."""

import pytest
import torch
from torch import nn

from tace.md_stages import opt4
from tace.md_stages.opt4_fusion import _Uniform1DRejector


def test_opt4_rejects_other_route() -> None:
    with pytest.raises(ValueError, match="TACE Opt4 route"):
        opt4.run_md(type("Request", (), {"model": "dpa4", "stage": "opt4"})())


class _TP(nn.Module):
    def forward(self, x, y, weight):
        return x * y + weight


def test_rejector_output_validator_accepts_bounded_reordering() -> None:
    rows = torch.tensor([0, 0, 1, 1])
    boundary = _Uniform1DRejector(_TP(), rows, rows=2, max_terms=2)
    x = torch.full((2, 4), 0.1)
    y = torch.ones(4, 4)
    weight = torch.zeros(4, 4)
    src = torch.tensor([0, 0, 1, 1])
    expected = boundary(x, y, weight, src)

    boundary.validate_output(
        expected + 2.0e-6,
        expected,
        (x, y, weight, src),
        0,
    )
