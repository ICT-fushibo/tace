################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
# TODO, other type norm

import torch
import torch.nn as nn


class ScalarLayerNorm(nn.Module):
    """
    LayerNorm for true scalar irreps (l = 0).

    This implementation is strictly consistent with TensorLayerNorm
    in the l = 0 case (2l + 1 = 1).
    
    Input shape:
        [batch, channel]
    """

    def __init__(
        self,
        lmin: int,
        lmax: int,
        num_channel: int,
        normalization: str = "component",
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        assert lmin == lmax == 0
        assert normalization in ["norm", "component"]

        self.lmin = lmin
        self.lmax = lmax
        self.num_channel = num_channel
        self.normalization = normalization
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(num_channel))
            if bias:
                self.bias = nn.Parameter(torch.zeros(num_channel))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"lmin={self.lmin}, lmax={self.lmax}, "
            f"num_channel={self.num_channel}, eps={self.eps})"
        )
    
    def forward(self, x: torch.Tensor):
        """
        x: [B, C]
        """
        x = x - x.mean(dim=1, keepdim=True)
        if self.normalization == "norm":
            norm = x.pow(2)
        else:
            norm = x.pow(2)
        var = norm.mean(dim=1, keepdim=True)
        inv_std = (var + self.eps).pow(-0.5)

        if self.elementwise_affine:
            x = x * inv_std * self.weight.view(1, -1)
        else:
            x = x * inv_std

        if self.elementwise_affine and self.bias is not None:
            x = x + self.bias.view(1, -1)

        return x


class TensorLayerNorm(nn.Module):
    """
    LayerNorm for scalar tensor blocks indexed by spherical harmonic degree l.

    Input shape:
        [batch, sum_{l=lmin}^{lmax} (2l+1), num_channel]
      = [batch, (lmax+1)^2 - lmin^2, num_channel]
    """

    def __init__(
        self,
        lmin: int,
        lmax: int,
        num_channel: int,
        normalization: str = "component",
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        assert 0 <= lmin <= lmax
        assert normalization in ["norm", "component"]

        self.lmin = lmin
        self.lmax = lmax
        self.num_channel = num_channel
        self.normalization = normalization
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(lmax - lmin + 1, num_channel)
            )
            if bias:
                self.bias = nn.Parameter(torch.zeros(num_channel))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"lmin={self.lmin}, lmax={self.lmax}, "
            f"num_channel={self.num_channel}, eps={self.eps})"
        )

    def forward(self, xs: torch.Tensor):
        """
        xs shape:
            [batch, (lmax+1)^2 - lmin^2, num_channel]
        """
        out = []

        for l in range(self.lmin, self.lmax + 1):
            start = l * l - self.lmin * self.lmin
            length = 2 * l + 1
            feature = xs.narrow(dim=1, start=start, length=length)

            if l == 0 and self.lmin == 0:
                mean = feature.mean(dim=2, keepdim=True)
                feature = feature - mean

            if self.normalization == "norm":
                norm = feature.pow(2).sum(dim=1, keepdim=True) 
            else:  
                norm = feature.pow(2).mean(dim=1, keepdim=True)  
            norm = norm.mean(dim=2, keepdim=True)           
            inv_std = (norm + self.eps).pow(-0.5)
            if self.elementwise_affine:
                weight = self.weight[l - self.lmin].view(1, 1, -1)
                inv_std = inv_std * weight

            feature = feature * inv_std

            if (
                self.elementwise_affine
                and self.bias is not None
                and l == 0
                and self.lmin == 0
            ):
                feature = feature + self.bias.view(1, 1, -1)

            out.append(feature)

        return torch.cat(out, dim=1)
