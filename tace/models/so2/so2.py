################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Union


import torch
import torch.nn.functional as F


from ..mlp import ScaledSigmoid
from .utils import so2_expand_index, satisfy


class Linear(torch.nn.Module):

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.randn((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.alpha = 1.0 / math.sqrt(in_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight * self.alpha, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class SO2MLinear(torch.nn.Module):
    """
    Based on https://github.com/atomicarchitects/equiformer_v3/blob/main/experimental/models/equiformer_v3/so2_ops.py
    Original Paper: https://proceedings.mlr.press/v202/passaro23a.html
    """
    def __init__(
        self,
        m: int, 
        num_channel_in: int,
        num_channel_out: int,
        num_components_in: int,
        num_components_out: int,
    ):
        super().__init__()

        self.m = m
        self.num_channel_in = num_channel_in
        self.num_channel_out = num_channel_out
        self.num_components_in = num_components_in
        self.num_components_out = num_components_out
        assert self.num_components_in > 0
        assert self.num_components_out > 0

        self.fc = Linear(
            self.num_components_in * self.num_channel_in,
            self.num_components_out * self.num_channel_out * 2,
            bias=False,
        )
        self.fc.weight.data.mul_(1 / math.sqrt(2))

        self._Cout = self.num_components_out * self.num_channel_out

    def forward(self, x, concat_outputs=True) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # [batch, 2, -1]
        x = self.fc(x)
        w1_x = x.narrow(2, 0, self._Cout)
        w2_x = x.narrow(2, self._Cout, self._Cout)
        xr = w1_x.narrow(1, 0, 1) - w2_x.narrow(1, 1, 1) # w1_x+m - w2x-m
        xi = w1_x.narrow(1, 1, 1) + w2_x.narrow(1, 0, 1) # w1_x-m + w2x+m
        x_out = (xr, xi)
        if concat_outputs:
            x_out = torch.cat(x_out, dim=1)
        return x_out


class SO2Linear(torch.nn.Module):
    """
    Based on https://github.com/atomicarchitects/equiformer_v3/blob/main/experimental/models/equiformer_v3/so2_ops.py
    Original Paper: https://proceedings.mlr.press/v202/passaro23a.html
    """
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel_in: int,
        num_channel_out: int,
        num_components_in: Union[None, list[int]] = None,
        num_components_out: Union[None, list[int]] = None,
    ):
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channel_in = num_channel_in
        self.num_channel_out = num_channel_out

        if num_components_in is None:
            self.num_components_in = [lmax + 1 -m for m in range(mmax + 1)]
        else:
            self.num_components_in = num_components_in
        assert isinstance(self.num_components_in, list)
        if num_components_out is None:
            self.num_components_out = [lmax + 1 -m for m in range(mmax + 1)]
        else:
            self.num_components_out = num_components_out
        assert isinstance(self.num_components_out, list)


        self.m0_rlinear = Linear(
            self.num_channel_in * self.num_components_in[0], 
            self.num_channel_out * self.num_components_out[0], 
            bias=True,
        )
        self.ms_clinear = torch.nn.ModuleList()
        for m in range(1, self.mmax + 1):
            self.ms_clinear.append(
                SO2MLinear(
                    m,
                    self.num_channel_in,
                    self.num_channel_out,
                    self.num_components_in[m], 
                    self.num_components_out[m], 
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, num_components, num_channel],
        # layout of m components is (0, 0, ...), (1, 1, ...), ...

        B = x.size(0)
        Cout = self.num_channel_out

        outputs = []

        # m = 0
        xm0 = x.narrow(1, 0, self.num_components_in[0])
        xm0 = xm0.reshape(B, -1)
        xm0 = self.m0_rlinear(xm0)
        xm0 = xm0.view(B, -1, Cout)
        outputs.append(xm0)

        # m > 0
        offset = self.lmax + 1
        for m in range(1, self.mmax + 1):
            xm = x.narrow(1, offset, 2 * (self.num_components_in[m]))
            offset = offset + 2 * self.num_components_in[m]
            xm = xm.reshape(B, 2, -1)
            xm = self.ms_clinear[m - 1](xm, concat_outputs=False)
            xr, xi = xm[0], xm[1]
            xr = xr.view(B, -1, Cout)
            xi = xi.view(B, -1, Cout)
            outputs.append(xr)
            outputs.append(xi)
        outputs = torch.cat(outputs, dim=1)

        return outputs
        
    def __repr__(self) -> str:
        p = {
            0: 'e',
            1: 'o',
        }
        ins = []
        outs = []
        for m in range(self.mmax + 1):
            n1 = self.num_components_in[m] 
            n2 = self.num_components_out[m] 
            ins.append(f"{self.num_channel_in * n1}x{m}{p[m % 2]}")
            outs.append(f"{self.num_channel_out * n2}x{m}{p[m % 2]}")
        num_weights = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return (
            f"{self.__class__.__name__}"
            f"({'+'.join(ins)} -> "
            f"{'+'.join(outs)} | "
            f"{num_weights} weights)"
            f"(bias={True})"
        )
    

class SO2Gate(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        channel_wise: bool = False,
    ):
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel

        if not channel_wise:
            self.num_components, expand_index = so2_expand_index(mmax, lmax)
        else:
            expand_index = []
            offset = 0
            for m in range(mmax + 1):
                index = torch.arange((lmax + 1))
                index = index + offset
                expand_index.append(index)
                if m > 0:
                    expand_index.append(index)    # +- m
                offset = offset + len(index)
            expand_index = torch.cat(expand_index, dim=0)
            expand_index = expand_index.long()
            self.num_components = offset

        self.register_buffer('expand_index', expand_index, persistent=False)

        self.activation = ScaledSigmoid()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        g = self.activation(g).view(B, self.num_components, -1)
        g = torch.index_select(g, dim=1, index=self.expand_index)
        return g * x 


class uuuSO2TensorProduct(torch.nn.Module):

    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channels: int,
        m1m2: Union[str, None] = None,
        internal_weights: bool = True,
    ):
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channels = num_channels
        self.m1m2 = m1m2
        self.cmul = self.cmul2
        self.instructions = []

        weight_numel = 0
        for m3 in range(mmax + 1):
            paths = self.enumerate_paths(m3)
            self.instructions.append(paths)
            weight_numel += num_channels * (lmax+1) * len(paths) 
            
        self.weight_numel = weight_numel
        if internal_weights:
            self.weight = torch.nn.Parameter(torch.randn(1, self.weight_numel))
        else:
            self.register_buffer("weight", None)
        self.internal_weights = internal_weights   

        output_scales = []
        n = lmax + 1
        # m = 0
        scale0 = torch.full((n,), 1.0 / math.sqrt(len(self.instructions[0])))
        output_scales.append(scale0)
        # m > 0
        for m3 in range(1, mmax + 1):
            scale = 1.0 / math.sqrt(len(self.instructions[m3]))
            output_scales.append(torch.full((2 * n,), scale))
        output_scales = torch.cat(output_scales)
        self.register_buffer("output_scales", output_scales, persistent=False)

        # for m3, paths in enumerate(self.instructions):
        #     for m1, m2, mode in paths:
        #         print(m1, m2, m3, mode)
        # print()

    def enumerate_paths(self, m3: int) -> list[tuple[int, int, str]]:
        paths = []

        for m1 in range(self.mmax + 1):
            for m2 in range(self.mmax + 1):
                if satisfy(m1, m2, self.m1m2):
                    # x1 * x2
                    if m1 + m2 == m3:
                        paths.append((m1, m2, "sum"))
                    # x1 * conj(x2)
                    elif abs(m1 - m2) == m3:
                        paths.append((m1, m2, "diff"))

        return paths

    def rmul(self, x, y): 
        # [B, n, C] * [B, n, C] =>  [B, n, C]
        z = x * y
        return z
    
    def cmul1(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
        '''Layout damei, should be 2 in last dim'''
        # [B, 2, n, C] * [B, 2, n, C] => [B, 2, n, C]
        x = x.permute(0,2,3,1).contiguous()
        y = y.permute(0,2,3,1).contiguous()
        x = torch.view_as_complex(x)
        y = torch.view_as_complex(y)
        if mode == "diff":
            y = y.conj()
        z = x * y
        B = z.size(0)
        C = self.num_channels
        z = z.reshape(B, -1, C)
        z = torch.view_as_real(z)
        z = z.permute(0,3,1,2)

        return z
    
    def cmul2(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
        # [B, 2, n, C] * [B, 2, n, C] => [B, 2, n, C]
        a = x[:, 0]
        b = x[:, 1]
        c = y[:, 0]
        d = y[:, 1]

        if mode == "sum":
            real = a * c - b * d
            imag = a * d + b * c
        else:
            real = a * c + b * d
            imag = b * c - a * d

        B = real.size(0)
        C = real.size(-1)

        real = real.reshape(B, -1, C)
        imag = imag.reshape(B, -1, C)

        out = torch.stack([real, imag], dim=1)

        return out
    
    def to_list(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        out = []
        offset = 0
        n = self.lmax + 1
        # m = 0
        out.append(x[:, offset:offset+n])
        offset += n
        # m > 0
        for m in range(1, self.mmax + 1):
            xm = x[:, offset:offset+2*n]
            xm = xm.view(B, 2, n, self.num_channels)
            out.append(xm)
            offset += 2 * n
        return out

    def forward(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor, 
            weight: Union[torch.Tensor, None] = None,
        ) -> torch.Tensor:

        xs = self.to_list(x) #  m = 0 [B, lmax+1, C]
        ys = self.to_list(y) #  m > 0 [B, 2, lmax+1, C]
        if self.internal_weights:
            ws = self.weight
        else:
            ws = weight


        C = self.num_channels

        outputs = []
        w_offset = 0

        # m = 0
        n = self.lmax + 1
        m0 = 0.0
        w_numel = C * n

        for m1, m2, mode in self.instructions[0]:
            w = ws[:, w_offset:w_offset+w_numel] # [B, C] or [1, C]
            w = w.view(-1, n, C)
            w_offset += w_numel

            # 0 x 0
            if m1 == 0 and m2 == 0:
                z = self.rmul(xs[0], ys[0])
                out = z * w
                m0 = m0 + out

            # m > 0 and m1 -m2 = 0
            elif m1 > 0 and m2 > 0:
                z = self.cmul(xs[m1], ys[m2], "diff")
                out = z[:, 0] * w # The imaginary part is 0
                m0 = m0 + out

        outputs.append(m0)

        # m > 0
        for m3 in range(1, self.mmax + 1):
            real = 0.0
            imag = 0.0
            for m1, m2, mode in self.instructions[m3]:
                w = ws[:, w_offset:w_offset+w_numel]
                w_offset += w_numel
                w = w.view(-1, 1, n, C)
                if m1 == 0 or m2 == 0:
                    continue
                
                if m1 < m2 and mode == 'diff':
                    z = self.cmul(ys[m2], xs[m1], mode)
                else:
                    z = self.cmul(xs[m1], ys[m2], mode)

                out = z * w
                real = real + out[:, 0]
                imag = imag + out[:, 1]

                # z = self.cmul(xs[m1], ys[m2], mode)
                # out = self.complex_channel_wise_fc(z, w)
                # real = real + out[:, 0]
                # if m1 < m2 and mode == 'diff':
                #     imag = imag - out[:, 1]
                # else:
                #     imag = imag + out[:, 1]

                # real = real + self.real_channel_wise_fc(z[:, 0], w)
                # imag = imag + self.real_channel_wise_fc(z[:, 1], w)

            outputs.append(real)
            outputs.append(imag)

        out = torch.cat(outputs, dim=1)
        out = out * self.output_scales.view(1, -1, 1)
        return out
        
    def __repr__(self):
        lines = []
        lines.append(
            f"{self.__class__.__name__}("
        )
        lines.append(
            f"  mmax={self.mmax}, "
            f"lmax={self.lmax}, "
            f"channels={self.num_channels}, "
            f"weights={self.weight_numel}"
        )
        lines.append("")
        lines.append("  instructions:")
        total_paths = 0
        for m3, paths in enumerate(self.instructions):
            total_paths += len(paths)
            path_strs = []
            for m1, m2, mode in paths:
                if mode == "sum":
                    expr = f"{m1}+{m2}"
                else:
                    expr = f"{m1}-{m2}"
                path_strs.append(expr)
            joined = ", ".join(path_strs)
            lines.append(
                f"    m={m3:<2} : "
                f"{len(paths):<2} paths | "
                f"{joined}"
            )
        lines.append("")
        lines.append(f"  total_paths={total_paths}")
        lines.append(")")
        return "\n".join(lines)