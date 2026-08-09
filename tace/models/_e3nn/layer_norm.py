"""
Copy From equiformer_v3

MIT License
Copyright (c) 2026 The Atomic Architects

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from functools import partial

import torch

_LAYER_NORM = [
    "merge_layer_norm",
    "merge_rms_norm",
]


def get_normalization_layer(
    norm_type, ls, num_channels, eps=1e-5, affine=True, normalization="component"
):
    assert norm_type in _LAYER_NORM
    if norm_type == "merge_layer_norm":
        norm_class = EquivariantMergeLayerNorm
    elif norm_type == "merge_rms_norm":
        norm_class = partial(EquivariantMergeLayerNorm, centering=False)
    else:
        raise ValueError
    return norm_class(ls, num_channels, eps, affine, normalization)


class EquivariantMergeLayerNorm(torch.nn.Module):
    def __init__(
        self,
        ls,
        num_channels,
        eps=1e-5,
        affine=True,
        normalization="component",
        std_balance_degrees=True,
        centering=True,
    ):
        super().__init__()

        if isinstance(ls, int):
            self.ls = list(range(ls + 1))
        else:
            assert 0 in ls
            assert list(ls) == sorted(ls)
            self.ls = list(ls)

        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees
        self.centering = centering

        # total irreps dimension
        self.dim = sum([2 * l + 1 for l in self.ls])

        if self.affine:
            self.affine_weight = torch.nn.Parameter(
                torch.ones((len(self.ls), self.num_channels))
            )
            # build expand_index
            expand_index = torch.zeros(self.dim, dtype=torch.long)
            offset = 0
            for i, l in enumerate(self.ls):
                length = 2 * l + 1
                expand_index[offset : offset + length] = i
                offset += length
            self.register_buffer("expand_index", expand_index)
            if self.centering:
                self.affine_bias = torch.nn.Parameter(torch.zeros(self.num_channels))
            else:
                self.register_parameter("affine_bias", None)
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros(self.dim, 1)
            offset = 0
            for l in self.ls:
                length = 2 * l + 1
                balance_degree_weight[offset : offset + length, :] = 1.0 / length
                offset += length
            balance_degree_weight = balance_degree_weight / len(self.ls)
            balance_degree_weight = balance_degree_weight.permute(1, 0)
            self.register_buffer("balance_degree_weight", balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(ls={self.ls}, num_channels={self.num_channels}, "
            f"eps={self.eps}, std_balance_degrees={self.std_balance_degrees}, "
            f"centering={self.centering})"
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        # for L = 0
        if self.centering:
            scalars = inputs.narrow(1, 0, 1)
            scalars_mean = scalars.mean(dim=2, keepdim=True)  # [N, 1, 1]
            scalars = scalars - scalars_mean
            inputs = torch.cat(
                (scalars, inputs.narrow(1, 1, inputs.shape[1] - 1)), dim=1
            )

        # for L >= 0
        feature_norm = inputs.pow(2)
        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)
        if self.normalization == "norm":
            feature_norm = feature_norm.sum(dim=1, keepdim=True)
        elif self.normalization == "component":
            if self.std_balance_degrees:
                feature_norm = torch.einsum(
                    "ai, nic -> nac",
                    self.balance_degree_weight,
                    feature_norm,
                )
            else:
                feature_norm = feature_norm.mean(dim=1, keepdim=True)
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        if self.affine:
            weight = self.affine_weight.view(1, len(self.ls), self.num_channels)
            weight = torch.index_select(weight, dim=1, index=self.expand_index)
            feature_norm = feature_norm * weight
        outputs = inputs * feature_norm

        if self.affine and self.centering:
            outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + self.affine_bias.view(
                1, 1, self.num_channels
            )

        return outputs
