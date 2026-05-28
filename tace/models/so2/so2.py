################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Union, NamedTuple, Iterator


import torch


from ..mlp import ScaledSigmoid
from ..linear import torchLinear
from .utils import so2_expand_index, satisfy


class uvSO2MLinear(torch.nn.Module):
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
        weight_type: str = 'w1_w2',
    ):
        super().__init__()

        self.m = m
        self.num_channel_in = num_channel_in
        self.num_channel_out = num_channel_out
        self.num_components_in = num_components_in
        self.num_components_out = num_components_out
        self.weight_type = weight_type
        assert self.num_components_in > 0
        assert self.num_components_out > 0

        if weight_type == 'w1_w2':
            self.fc = torchLinear(
                self.num_components_in * self.num_channel_in,
                self.num_components_out * self.num_channel_out * 2,
                bias=False,
            )
            self.fc.weight.data.mul_(1 / math.sqrt(2))
        elif weight_type == 'w1_w1':
            self.fc = torchLinear(
                self.num_components_in * self.num_channel_in,
                self.num_components_out * self.num_channel_out,
                bias=False,
            )
            self.fc.weight.data.mul_(1 / math.sqrt(2))
        else:
            self.fc = torchLinear(
                self.num_components_in * self.num_channel_in,
                self.num_components_out * self.num_channel_out,
                bias=False,
            )
        self._Cout = self.num_components_out * self.num_channel_out


    def forward(self, x, concat_outputs=True):
        # [batch, 2, -1]
        if self.weight_type == 'w1_w2':
            return self.w1_w2_forward(x, concat_outputs)
        elif self.weight_type == 'w1_w1':
            return self.w1_w1_forward(x, concat_outputs)
        else:
            return self.w1_forward(x, concat_outputs)
    
    def w1_w2_forward(self, x, concat_outputs=True) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        x = self.fc(x)
        w1_x = x.narrow(2, 0, self._Cout)
        w2_x = x.narrow(2, self._Cout, self._Cout)
        xr = w1_x.narrow(1, 0, 1) - w2_x.narrow(1, 1, 1) # w1_x+m - w2x-m
        xi = w1_x.narrow(1, 1, 1) + w2_x.narrow(1, 0, 1) # w1_x-m + w2x+m
        x_out = (xr, xi)
        if concat_outputs:
            x_out = torch.cat(x_out, dim=1)
        return x_out

    def w1_w1_forward(self, x, concat_outputs=True):
        xr = x.narrow(1, 0, 1)
        xi = x.narrow(1, 1, 1)
        # yr = W(xr - xi)
        # yi = W(xi + xr)
        yr_in = xr - xi
        yi_in = xi + xr
        yr = self.fc(yr_in)
        yi = self.fc(yi_in)
        x_out = (yr, yi)
        if concat_outputs:
            x_out = torch.cat(x_out, dim=1)
        return x_out
    
    def w1_forward(self, x, concat_outputs=True):
        x = self.fc(x)
        if concat_outputs:
            return x
        return (x.narrow(1, 0, 1), x.narrow(1, 1, 1))


class uvSO2Linear(torch.nn.Module):
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
        weight_type: str = 'w1_w2' # [w1_w2, w1_w1, w1]
    ):
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channel_in = num_channel_in
        self.num_channel_out = num_channel_out
        self.weight_type = weight_type

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


        self.m0_rlinear = torchLinear(
            self.num_channel_in * self.num_components_in[0], 
            self.num_channel_out * self.num_components_out[0], 
            bias=True,
        )
        self.ms_clinear = torch.nn.ModuleList()
        for m in range(1, self.mmax + 1):
            self.ms_clinear.append(
                uvSO2MLinear(
                    m,
                    self.num_channel_in,
                    self.num_channel_out,
                    self.num_components_in[m], 
                    self.num_components_out[m], 
                    weight_type=weight_type,
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
            f"(weight_type={self.weight_type})"
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
    

class SO2Norm(torch.nn.Module): # TODO

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

        self.ns = [
            self.lmax + 1
            if channel_wise
            else self.lmax + 1 - m
            for m in range(mmax + 1)
        ]   

        class ComplexSigmoid(torch.nn.Module):
            def __init__(self, eps: float = 1e-5):
                super().__init__()
                self.eps = eps
                self.act = ScaledSigmoid()

            def intensity(self, x: torch.Tensor) -> torch.Tensor:
                return (x**2).sum(dim=1, keepdim=True) * 0.5

            def magnitude(self, x: torch.Tensor) -> torch.Tensor:
                return (self.intensity(x) + self.eps).pow(-0.5)
              
            def forward(self, x: torch.Tensor) ->  torch.Tensor:
                # return self.act(self.intensity(x))
                return self.act(self.magnitude(x))
            
        activation = [ScaledSigmoid()]
        for m in range(1, mmax + 1):
            activation.append(ComplexSigmoid())
        self.activation = torch.nn.ModuleList(activation)

    def forward(self, x: torch.Tensor, g: Union[torch.Tensor, None] = None) -> torch.Tensor:
        # m=0: [B, n, C]
        # m>0: [B, 2, n, C]
        xs = self.to_list(x)
        out = []
        # m = 0
        x0 = xs[0]
        x0 = self.activation[0](x0) * x0
        out.append(x0)
        # m > 0
        for m, xm in enumerate(xs[1:], start=1):
            xm = self.activation[m](xm) * xm
            out.append(xm)
        res = self.from_list(out)
        return res

    def to_list(self, x: torch.Tensor) -> list[torch.Tensor]:
        B = x.size(0)
        out = []
        offset = 0
        n = self.ns[0]
        x0 = x[:, offset:offset + n]
        x0 = x0.view(B, n, self.num_channel)
        out.append(x0)
        offset += n
        for m in range(1, self.mmax + 1):
            n = self.ns[m]
            xm = x[:, offset:offset + 2 * n]
            xm = xm.view(B, 2, n, self.num_channel)
            out.append(xm)
            offset += 2 * n
        return out

    def from_list(self, xs: list[torch.Tensor]) -> torch.Tensor:
        out = []
        x0 = xs[0]
        B, n, C = x0.shape
        out.append(x0)
        for xm in xs[1:]:
            B, _, n, C = xm.shape
            xm = xm.reshape(B, 2 * n, C)
            out.append(xm)
        return torch.cat(out, dim=1)


class SO2ComplexMul(torch.nn.Module):

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

        self.ns = [
            self.lmax + 1
            if channel_wise
            else self.lmax + 1 - m
            for m in range(mmax + 1)
        ]

        self.slices = []
        offset = 0
        for m, n in zip(range(mmax + 1), self.ns):
            if m == 0:
                self.slices.append(slice(offset, offset + n))
                offset += n
            else:
                self.slices.append(slice(offset, offset + 2 * n))
                offset += 2 * n

        if not channel_wise:
            self.num_components, expand_index = so2_expand_index(mmax, lmax)
        else:
            expand_index = []
            offset = 0
            for m in range(mmax + 1):
                index = torch.arange(lmax + 1)
                index = index + offset
                expand_index.append(index)
                if m > 0:
                    expand_index.append(index)
                offset += len(index)
            expand_index = torch.cat(expand_index, dim=0).long()
            self.num_components = offset

        self.register_buffer("expand_index", expand_index, persistent=False)

        self.scale = 1.0 / math.sqrt(2)

    def forward(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        scale: bool = True,
    ) -> torch.Tensor:

        B = xs.size(0)
        out = []
        for m, s in enumerate(self.slices):
            x = xs[:, s]
            y = ys[:, s]
            n = self.ns[m]
            # m = 0
            if m == 0:
                z = x * y
            # m > 0
            else:
                x = x.view(B, 2, n, self.num_channel)
                y = y.view(B, 2, n, self.num_channel)
                xr = x[:, 0]
                xi = x[:, 1]
                yr = y[:, 0]
                yi = y[:, 1]
                zr = xr * yr - xi * yi
                zi = xr * yi + xi * yr
                z = torch.stack([zr, zi], dim=1)
                z = z.reshape(B, 2 * n, self.num_channel)
                if scale:
                    z = z * self.scale
            out.append(z)

        return torch.cat(out, dim=1)
    
    def complex_forward(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        scale: bool = True,
    ) -> torch.Tensor:
        
        x_list = self.to_complex(xs)
        y_list = self.to_complex(ys)
        z_list = []

        # m = 0
        z_list.append(x_list[0] * y_list[0])

        # m > 0
        for x, y in zip(x_list[1:], y_list[1:]):
            z_list.append(x * y)

        return self.from_complex(z_list, scale)
    
    def to_complex(
        self,
        xs: torch.Tensor
    ):
        B = xs.size(0)
        out_list = []
        for m, s in enumerate(self.slices):
            x = xs[:, s]
            n = self.ns[m]
            if m == 0:
                xc = torch.complex(
                    x, 
                    torch.zeros_like(x),
                )
            else:
                x = x.view(B, 2, n, self.num_channel)
                xc = torch.complex(
                    x[:, 0],
                    x[:, 1],
                )
            out_list.append(xc)

        return out_list

    def from_complex(
        self,
        xs,
        scale,
    ):
        out = []
        for m, x in enumerate(xs):
            if m == 0:
                out.append(x.real)
            else:
                xr = x.real
                xi = x.imag
                x = torch.stack(
                    [xr, xi],
                    dim=1,
                )
                B = x.size(0)
                x = x.reshape(
                    B,
                    -1,
                    self.num_channel,
                )
                if scale:
                    x = x * self.scale
                out.append(x)
        return torch.cat(out, dim=1)
 

class LegacyuuSO2TensorProduct(torch.nn.Module):
    """
    The results of all paths are directly summed.
    """
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
                out = z[:, 0] * w # imag is also invariant, but nod add here
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

                if m1 == 0:
                    z = xs[m1].unsqueeze(1) * ys[m2]
                elif m2 == 0:
                    z = xs[m1] * ys[m2].unsqueeze(1)
                else:
                    if m1 < m2 and mode == 'diff':
                        z = self.cmul(ys[m2], xs[m1], mode)
                    else:
                        z = self.cmul(xs[m1], ys[m2], mode)

                out = z * w
                real = real + out[:, 0]
                imag = imag + out[:, 1]

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


class SO2Rot90(torch.nn.Module):

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

        self.ns = [
            self.lmax + 1
            if channel_wise
            else self.lmax + 1 - m
            for m in range(mmax + 1)
        ]

        self.slices = []
        offset = 0
        for m, n in zip(range(mmax + 1), self.ns):
            if m == 0:
                self.slices.append(slice(offset, offset + n))
                offset += n
            else:
                self.slices.append(slice(offset, offset + 2 * n))
                offset += 2 * n

    def forward(
        self,
        xs: torch.Tensor,
    ) -> torch.Tensor:

        B = xs.size(0)
        out = []
        for m, s in enumerate(self.slices):
            x = xs[:, s]
            n = self.ns[m]
            # m = 0
            if m == 0:
                z = x
            # m > 0
            else:
                x = x.view(B, 2, n, self.num_channel)
                r = x.narrow(1, 1, 1) * (-1.0)
                i = x.narrow(1, 0, 1)
                z = torch.cat([r, i], dim=1)
                z = z.reshape(B, 2 * n, self.num_channel)
            out.append(z)

        return torch.cat(out, dim=1)
    

class uuLinearInstruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple[int, ...]
    path_weight: float
    m: int
    weight_mode: str
    output_slice: slice


class uuSO2Linear(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        weight_type: str = "w1",
        path_mode: str = "sum", # TODO, concat may be have bug
        path_norm: bool = True,
    ) -> None:
        super().__init__()
        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.weight_type = weight_type
        self.path_mode = path_mode
        self.path_norm = path_norm
        if self.weight_type not in {"w1_w2", "w1_w1", "w1"}:
            raise ValueError(
                f"Unknown uuSO2Linear weight_type={weight_type!r}; "
                "expected 'w1_w2', 'w1_w1', or 'w1'."
            )
        if self.path_mode not in {"sum", "concat"}:
            raise ValueError(
                f"Unknown uuSO2Linear path_mode={path_mode!r}; "
                "expected 'sum' or 'concat'."
            )

        path_specs = []
        concat_path_specs = []
        instructions = []
        path_count = {}
        weight_offset = 0
        output_offset = 0

        def add_instruction(lin: int, lout: int, m: int, path_weight: float) -> int:
            nonlocal weight_offset, output_offset
            weight_mode = "real" if m == 0 or self.weight_type != "w1_w2" else "complex"
            path_shape = (self.num_channel,) if weight_mode == "real" else (2, self.num_channel)
            output_width = 1 if m == 0 else 2
            output_slice = slice(output_offset, output_offset + output_width)
            instructions.append(
                uuLinearInstruction(
                    i_in=lin,
                    i_out=lout,
                    path_shape=path_shape,
                    path_weight=path_weight,
                    m=m,
                    weight_mode=weight_mode,
                    output_slice=output_slice,
                )
            )
            offset = weight_offset
            weight_offset += math.prod(path_shape) // self.num_channel
            output_offset += output_width
            return offset

        for lout in range(self.lmax + 1):
            for lin in range(self.lmax + 1):
                for m in range(min(lout, lin, self.mmax) + 1):
                    path_count[(lout, m)] = path_count.get((lout, m), 0) + 1

        for m in range(self.mmax + 1):
            for lout in range(m, self.lmax + 1):
                for lin in range(m, self.lmax + 1):
                    if m == 0:
                        path_weight = (
                            1.0 / math.sqrt(path_count[(lout, 0)])
                            if self.path_norm else 1.0
                        )
                        weight_index = add_instruction(lin, lout, m, path_weight)
                        concat_out = instructions[-1].output_slice.start
                        path_specs.append((
                            self._component_index(lout, 0, False),
                            self._component_index(lin, 0, False),
                            weight_index,
                            path_weight,
                        ))
                        concat_path_specs.append((
                            concat_out,
                            self._component_index(lin, 0, False),
                            weight_index,
                            path_weight,
                        ))
                    elif self.weight_type == "w1_w2":
                        path_weight = (
                            1.0 / math.sqrt(2 * path_count[(lout, m)])
                            if self.path_norm else 1.0
                        )
                        weight_index = add_instruction(lin, lout, m, path_weight)
                        concat_out = instructions[-1].output_slice.start
                        w_real = weight_index
                        w_imag = weight_index + 1
                        out_r = self._component_index(lout, m, False)
                        out_i = self._component_index(lout, m, True)
                        in_r = self._component_index(lin, m, False)
                        in_i = self._component_index(lin, m, True)
                        path_specs.extend([
                            (out_r, in_r, w_real, path_weight),
                            (out_i, in_i, w_real, path_weight),
                            (out_r, in_i, w_imag, -path_weight),
                            (out_i, in_r, w_imag, path_weight),
                        ])
                        concat_path_specs.extend([
                            (concat_out, in_r, w_real, path_weight),
                            (concat_out + 1, in_i, w_real, path_weight),
                            (concat_out, in_i, w_imag, -path_weight),
                            (concat_out + 1, in_r, w_imag, path_weight),
                        ])
                    elif self.weight_type == "w1_w1":
                        path_weight = (
                            1.0 / math.sqrt(2 * path_count[(lout, m)])
                            if self.path_norm else 1.0
                        )
                        weight_index = add_instruction(lin, lout, m, path_weight)
                        concat_out = instructions[-1].output_slice.start
                        out_r = self._component_index(lout, m, False)
                        out_i = self._component_index(lout, m, True)
                        in_r = self._component_index(lin, m, False)
                        in_i = self._component_index(lin, m, True)
                        path_specs.extend([
                            (out_r, in_r, weight_index, path_weight),
                            (out_i, in_i, weight_index, path_weight),
                            (out_r, in_i, weight_index, -path_weight),
                            (out_i, in_r, weight_index, path_weight),
                        ])
                        concat_path_specs.extend([
                            (concat_out, in_r, weight_index, path_weight),
                            (concat_out + 1, in_i, weight_index, path_weight),
                            (concat_out, in_i, weight_index, -path_weight),
                            (concat_out + 1, in_r, weight_index, path_weight),
                        ])
                    else:
                        path_weight = (
                            1.0 / math.sqrt(path_count[(lout, m)])
                            if self.path_norm else 1.0
                        )
                        weight_index = add_instruction(lin, lout, m, path_weight)
                        concat_out = instructions[-1].output_slice.start
                        out_r = self._component_index(lout, m, False)
                        out_i = self._component_index(lout, m, True)
                        in_r = self._component_index(lin, m, False)
                        in_i = self._component_index(lin, m, True)
                        path_specs.extend([
                            (out_r, in_r, weight_index, path_weight),
                            (out_i, in_i, weight_index, path_weight),
                        ])
                        concat_path_specs.extend([
                            (concat_out, in_r, weight_index, path_weight),
                            (concat_out + 1, in_i, weight_index, path_weight),
                        ])

        self.instructions = instructions
        self.num_weights = weight_offset
        self.weight_numel = self.num_weights * self.num_channel
        self.num_components = self._num_so2_components()
        self.num_path_components = output_offset
        self.num_components_per_m = self._num_components_per_m()
        self.num_expanded_components_per_m = [
            n if m == 0 else 2 * n for m, n in enumerate(self.num_components_per_m)
        ]
        self.num_output_components = (
            self.num_components if self.path_mode == "sum" else self.num_path_components
        )
        self.path_out = torch.tensor([p[0] for p in path_specs], dtype=torch.long)
        self.path_in = torch.tensor([p[1] for p in path_specs], dtype=torch.long)
        self.path_weight_index = torch.tensor([p[2] for p in path_specs], dtype=torch.long)
        self.path_weight = torch.tensor(
            [p[3] for p in path_specs],
            dtype=torch.get_default_dtype(),
        )
        self.concat_path_out = torch.tensor(
            [p[0] for p in concat_path_specs],
            dtype=torch.long,
        )
        self.concat_path_in = torch.tensor(
            [p[1] for p in concat_path_specs],
            dtype=torch.long,
        )
        self.concat_path_weight_index = torch.tensor(
            [p[2] for p in concat_path_specs],
            dtype=torch.long,
        )
        self.concat_path_weight = torch.tensor(
            [p[3] for p in concat_path_specs],
            dtype=torch.get_default_dtype(),
        )
    def _num_so2_components(self) -> int:
        return (self.lmax + 1) + sum(
            2 * (self.lmax + 1 - m) for m in range(1, self.mmax + 1)
        )

    def _num_components_per_m(self) -> list[int]:
        if self.path_mode == "sum":
            return [self.lmax + 1 - m for m in range(self.mmax + 1)]
        counts = [0 for _ in range(self.mmax + 1)]
        for ins in self.instructions:
            counts[ins.m] += 1
        return counts

    def _component_index(self, l: int, m: int, imag: bool) -> int:
        if m == 0:
            return l
        offset = self.lmax + 1
        for mm in range(1, m):
            offset += 2 * (self.lmax + 1 - mm)
        n = self.lmax + 1 - m
        index = offset + (l - m)
        if imag:
            index += n
        return index

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.num_channel:
            raise ValueError(
                f"uuSO2Linear expected {self.num_channel} channels, "
                f"got {x.size(-1)}"
            )
        batch = x.size(0)
        shared_weight = weight.dim() == 1
        if not shared_weight and weight.size(0) == 1 and batch != 1:
            weight = weight.expand(batch, -1)
        if not shared_weight and weight.size(0) != batch:
            raise ValueError(
                "uuSO2Linear weight batch dimension must be 1 or match input"
            )
        if weight.numel() == self.weight_numel:
            weight = weight.view(self.num_weights, self.num_channel)
            shared_weight = True
        elif weight.size(-1) == self.weight_numel:
            weight = weight.view(batch, self.num_weights, self.num_channel)
            shared_weight = False
        else:
            raise ValueError(
                f"uuSO2Linear expected {self.weight_numel} weights, "
                f"got {weight.size(-1)}"
            )

        if self.path_mode == "sum":
            out_width = self.num_components
            path_out = self.path_out.to(device=x.device)
            path_in = self.path_in.to(device=x.device)
            path_weight_index = self.path_weight_index.to(device=x.device)
            path_weight = self.path_weight.to(dtype=x.dtype, device=x.device)
        else:
            out_width = self.num_path_components
            path_out = self.concat_path_out.to(device=x.device)
            path_in = self.concat_path_in.to(device=x.device)
            path_weight_index = self.concat_path_weight_index.to(device=x.device)
            path_weight = self.concat_path_weight.to(dtype=x.dtype, device=x.device)

        x_path = torch.index_select(x, dim=1, index=path_in)
        if shared_weight:
            w_path = torch.index_select(weight, dim=0, index=path_weight_index)
            y_path = x_path * w_path.unsqueeze(0)
        else:
            w_path = torch.index_select(weight, dim=1, index=path_weight_index)
            y_path = x_path * w_path
        y_path = y_path * path_weight.view(1, -1, 1)
        out = x.new_zeros(batch, out_width, self.num_channel)
        out.index_add_(1, path_out, y_path)
        return out

    def _get_weights(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if weight.size(-1) != self.weight_numel:
            raise ValueError(
                f"uuSO2Linear expected {self.weight_numel} weights, "
                f"got {weight.size(-1)}"
            )
        return weight

    def weight_view_for_instruction(
        self,
        instruction: int,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        offset = sum(math.prod(ins.path_shape) for ins in self.instructions[:instruction])
        ins = self.instructions[instruction]
        weight = self._get_weights(weight)
        batchshape = weight.shape[:-1]
        flat_size = math.prod(ins.path_shape)
        return weight.narrow(-1, offset, flat_size).view(batchshape + ins.path_shape)

    def weight_views(
        self,
        weight: torch.Tensor,
        yield_instruction: bool = False,
    ) -> Iterator[torch.Tensor]:
        weight = self._get_weights(weight)
        batchshape = weight.shape[:-1]
        offset = 0
        for ins_i, ins in enumerate(self.instructions):
            flat_size = math.prod(ins.path_shape)
            this_weight = weight.narrow(-1, offset, flat_size).view(
                batchshape + ins.path_shape
            )
            offset += flat_size
            if yield_instruction:
                yield ins_i, ins, this_weight
            else:
                yield this_weight

    def extra_repr(self) -> str:
        return (
            f"mmax={self.mmax}, lmax={self.lmax}, "
            f"num_channel={self.num_channel}, mode=uu, path_norm={self.path_norm}, "
            f"weight_type={self.weight_type}, path_mode={self.path_mode}, "
            f"num_weights={self.num_weights}, "
            f"num_paths={self.path_out.numel()}, "
            f"num_components_per_m={self.num_components_per_m}, "
            f"num_output_components={self.num_output_components}, "
            f"weight_numel={self.weight_numel}"
        )
    

class uuSO2TensorProductInstruction(NamedTuple):
    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    path_shape: tuple[int, ...]
    path_weight: float
    output_slice: slice


class uuSO2TensorProduct(torch.nn.Module):
    """
    SO(2) u-u tensor product for the reduced ``linear_up(path_mode="sum")``
    layout.

    The number of complex components is fixed by ``mmax`` and ``lmax``:
    ``num_components_per_m[m] = lmax + 1 - m``. Correlation 1 is an identity
    branch. Correlation 2 uses internal, non-edge-dependent weights and sums all
    valid paths back into the same SO(2) layout.
    """

    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channels: int,
        m1m2: Union[str, None] = "<=",
        weight_type: str = "w1",
        correlation: int = 2,
        path_norm: bool = True,
    ) -> None:
        super().__init__()
        self.mmax = mmax
        self.lmax = lmax
        self.num_channels = num_channels
        self.num_components_per_m = [lmax + 1 - m for m in range(mmax + 1)]
        self.m1m2 = m1m2
        self.correlation = correlation
        self.weight_type = weight_type
        self.path_norm = path_norm
        if self.correlation not in {1, 2}:
            raise ValueError("uuSO2TensorProduct only supports correlation=1 or 2")
        if self.weight_type not in {"w1_w2", "w1_w1", "w1"}:
            raise ValueError(
                f"Unknown uuSO2TensorProduct weight_type={weight_type!r}; "
                "expected 'w1_w2', 'w1_w1', or 'w1'."
            )

        self.num_input_components = sum(
            n if m == 0 else 2 * n
            for m, n in enumerate(self.num_components_per_m)
        )
        self.instructions = []
        self.num_output_components_per_m = list(self.num_components_per_m)
        self.num_output_components = self.num_input_components
        self.weight_numel = 0
        self.output_weight = torch.ones(self.num_output_components)

        if self.correlation == 2:
            self._setup_correlation2()
            self.weight = torch.nn.Parameter(torch.randn(1, self.weight_numel))
            if self.weight_type in {"w1_w2", "w1_w1"}:
                self.weight.data.mul_(1.0 / math.sqrt(2.0))
        else:
            self.register_parameter("weight", None)

    def _setup_correlation2(self) -> None:
        grouped_paths = [self._enumerate_m_paths(m3) for m3 in range(self.mmax + 1)]
        instructions = []
        weight_offset = 0
        output_path_counts = torch.zeros(self.num_output_components)

        for m3, paths in enumerate(grouped_paths):
            for m1, m2, mode in paths:
                min_l = max(m1, m2, m3)
                n_out = self.lmax + 1 - min_l
                if n_out <= 0:
                    continue
                out_start = self._component_index(min_l, m3, False)
                output_slice = slice(out_start, out_start + n_out)

                if self.weight_type == "w1_w2" and not (m1 == 0 and m2 == 0):
                    path_shape = (2, n_out, self.num_channels)
                else:
                    path_shape = (n_out, self.num_channels)
                instructions.append(
                    uuSO2TensorProductInstruction(
                        i_in1=m1,
                        i_in2=m2,
                        i_out=m3,
                        connection_mode=mode,
                        path_shape=path_shape,
                        path_weight=1.0,
                        output_slice=output_slice,
                    )
                )
                weight_offset += math.prod(path_shape)
                output_path_counts[output_slice] += 1.0
                if m3 > 0:
                    imag_start = self._component_index(min_l, m3, True)
                    output_path_counts[imag_start:imag_start + n_out] += 1.0

        self.instructions = instructions
        self.weight_numel = weight_offset
        if self.path_norm:
            output_weight = torch.where(
                output_path_counts > 0,
                output_path_counts.rsqrt(),
                torch.zeros_like(output_path_counts),
            )
        else:
            output_weight = torch.ones_like(output_path_counts)
        self.output_weight = output_weight

    def _enumerate_m_paths(self, m3: int) -> list[tuple[int, int, str]]:
        paths = []
        for m1 in range(self.mmax + 1):
            for m2 in range(self.mmax + 1):
                if satisfy(m1, m2, self.m1m2):
                    if m1 + m2 == m3:
                        paths.append((m1, m2, "sum"))
                    elif abs(m1 - m2) == m3:
                        paths.append((m1, m2, "diff"))
        return paths

    def _to_list(self, x: torch.Tensor) -> list[torch.Tensor]:
        parts = []
        offset = 0
        parts.append(x[:, offset:offset + self.num_components_per_m[0]])
        offset += self.num_components_per_m[0]
        for m in range(1, self.mmax + 1):
            n = self.num_components_per_m[m]
            xm = x[:, offset:offset + 2 * n]
            parts.append(xm.view(x.size(0), 2, n, self.num_channels))
            offset += 2 * n
        return parts

    def _component_index(self, l: int, m: int, imag: bool) -> int:
        if m == 0:
            return l
        offset = self.lmax + 1
        for mm in range(1, m):
            offset += 2 * (self.lmax + 1 - mm)
        n = self.lmax + 1 - m
        index = offset + (l - m)
        if imag:
            index += n
        return index

    def _complex_product_parts(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mode: str,
        reverse_diff: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xr = x[:, 0]
        xi = x[:, 1]
        yr = y[:, 0]
        yi = y[:, 1]
        if mode == "sum":
            real = xr * yr - xi * yi
            imag = xr * yi + xi * yr
        else:
            real = xr * yr + xi * yi
            if reverse_diff:
                imag = xr * yi - xi * yr
            else:
                imag = xi * yr - xr * yi
        return real, imag

    def _path_weight(
        self,
        weight: torch.Tensor,
        offset: int,
        ins: uuSO2TensorProductInstruction,
    ) -> tuple[torch.Tensor, Union[torch.Tensor, None], int]:
        flat_size = math.prod(ins.path_shape)
        view = weight.narrow(-1, offset, flat_size).view(weight.shape[:-1] + ins.path_shape)
        if self.weight_type == "w1_w2" and len(ins.path_shape) == 3:
            return view[:, 0], view[:, 1], offset + flat_size
        return view, None, offset + flat_size

    def _weight_real_output(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        wr: torch.Tensor,
        wi: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        if self.weight_type == "w1_w2" and wi is not None:
            return real * wr - imag * wi
        if self.weight_type == "w1_w1":
            return (real - imag) * wr
        return real * wr

    def _weight_complex_output(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        wr: torch.Tensor,
        wi: Union[torch.Tensor, None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.weight_type == "w1_w2" and wi is not None:
            return real * wr - imag * wi, real * wi + imag * wr
        if self.weight_type == "w1_w1":
            return (real - imag) * wr, (imag + real) * wr
        return real * wr, imag * wr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.num_channels:
            raise ValueError(
                f"uuSO2TensorProduct expected {self.num_channels} channels, "
                f"got {x.size(-1)}"
            )
        if x.size(1) != self.num_input_components:
            raise ValueError(
                f"uuSO2TensorProduct expected {self.num_input_components} "
                f"SO2 components, got {x.size(1)}"
            )
        if self.correlation == 1:
            return x

        xs = self._to_list(x)
        out = x.new_zeros(x.size(0), self.num_output_components, self.num_channels)
        weight = self.weight
        w_offset = 0

        for ins in self.instructions:
            m1 = ins.i_in1
            m2 = ins.i_in2
            m3 = ins.i_out
            min_l = max(m1, m2, m3)
            n_out = self.lmax + 1 - min_l
            out_start = self._component_index(min_l, m3, False)
            in1_start = min_l - m1
            in2_start = min_l - m2
            wr, wi, w_offset = self._path_weight(weight, w_offset, ins)

            if m3 == 0:
                if m1 == 0 and m2 == 0:
                    real = (
                        xs[0][:, in1_start:in1_start + n_out]
                        * xs[0][:, in2_start:in2_start + n_out]
                    )
                    imag = torch.zeros_like(real)
                else:
                    real, imag = self._complex_product_parts(
                        xs[m1][:, :, in1_start:in1_start + n_out],
                        xs[m2][:, :, in2_start:in2_start + n_out],
                        "diff",
                    )
                out[:, out_start:out_start + n_out] = (
                    out[:, out_start:out_start + n_out]
                    + self._weight_real_output(real, imag, wr, wi)
                )
                continue

            if m1 == 0:
                real = (
                    xs[m1][:, in1_start:in1_start + n_out]
                    * xs[m2][:, 0, in2_start:in2_start + n_out]
                )
                imag = (
                    xs[m1][:, in1_start:in1_start + n_out]
                    * xs[m2][:, 1, in2_start:in2_start + n_out]
                )
            elif m2 == 0:
                real = (
                    xs[m1][:, 0, in1_start:in1_start + n_out]
                    * xs[m2][:, in2_start:in2_start + n_out]
                )
                imag = (
                    xs[m1][:, 1, in1_start:in1_start + n_out]
                    * xs[m2][:, in2_start:in2_start + n_out]
                )
            else:
                real, imag = self._complex_product_parts(
                    xs[m1][:, :, in1_start:in1_start + n_out],
                    xs[m2][:, :, in2_start:in2_start + n_out],
                    ins.connection_mode,
                    reverse_diff=(m1 < m2 and ins.connection_mode == "diff"),
                )

            out_real, out_imag = self._weight_complex_output(real, imag, wr, wi)
            imag_start = self._component_index(min_l, m3, True)
            out[:, out_start:out_start + n_out] = (
                out[:, out_start:out_start + n_out] + out_real
            )
            out[:, imag_start:imag_start + n_out] = (
                out[:, imag_start:imag_start + n_out] + out_imag
            )

        return out * self.output_weight.to(dtype=x.dtype, device=x.device).view(1, -1, 1)

    def extra_repr(self) -> str:
        return (
            f"mmax={self.mmax}, lmax={self.lmax}, channels={self.num_channels}, "
            f"correlation={self.correlation}, weight_type={self.weight_type}, "
            f"path_norm={self.path_norm}, "
            f"num_components_per_m={self.num_components_per_m}, "
            f"weight_numel={self.weight_numel}"
        )
