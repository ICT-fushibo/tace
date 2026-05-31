# TODO, refactor
################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Union


import torch


from ..mlp import SmoothLeakyReLU, ScaledSigmoid
from ..so2 import LegacyuuSO2TensorProduct, uuSO2TensorProduct, SO3VstpGrid

# legacy
class SO2EdgeProductBasis(torch.nn.Module):

    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channels: int,
        num_elements: int,
        m1m2: Union[str, None] = '<=',
        agnostic: bool = True,
    ):
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_components = lmax+1
        self.num_channel = num_channels
        self.agnostic = agnostic

        self.ace = LegacyuuSO2TensorProduct(
            mmax, 
            lmax,
            num_channels, 
            m1m2=m1m2, 
            internal_weights=agnostic,
        )
        self.weight_numel = self.ace.weight_numel

        if not agnostic:
            self.num_c1_weight = (mmax+1) * (lmax+1) * num_channels
            self.weight_numel += self.num_c1_weight
            self.source_coefs = torch.nn.Parameter(
                torch.randn(num_elements, self.weight_numel)
            )
            self.target_coefs = torch.nn.Parameter(
                torch.randn(num_elements, self.weight_numel)
            )
            self.source_coefs.data.mul_(1 / math.sqrt(2))
            self.target_coefs.data.mul_(1 / math.sqrt(2))

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
   
    def forward(
            self, 
            x, 
            node_attes, 
            edge_index
        ) -> torch.Tensor:

        if self.agnostic:
            return x + self.ace(x, x)

        B = x.size(0)
        C = self.num_channel

        node_type = node_attes.argmax(dim=-1)
        src_type = node_type[edge_index[0]]
        dst_type = node_type[edge_index[1]]
        source_coefs = self.source_coefs[src_type]
        target_coefs = self.target_coefs[dst_type]
        
        # nu = 1
        w1 = (
            source_coefs[:, :self.num_c1_weight]
            + target_coefs[:, :self.num_c1_weight]
        )
        w1 = w1.view(B, -1, C)
        w1 = torch.index_select(w1, dim=1, index=self.expand_index)
        corr_feats1 = x * w1

        # nu = 2
        w2 = (
            source_coefs[:, self.num_c1_weight:]
            + target_coefs[:, self.num_c1_weight:]
        )
        corr_feats2 = self.ace(x, x, w2)

        return corr_feats1 + corr_feats2


    def extra_repr(self) -> str:
        p = {
            0: 'e',
            1: 'o',
        }
        irreps = []
        for m in range(self.mmax + 1):
            irreps.append(f"{self.num_channel*(self.lmax+1)}x{m}{p[m % 2]}")
        num_weights = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return (
            f"{self.__class__.__name__}"
            f"({'+'.join(irreps)} x {'+'.join(irreps)} -> "
            f"{'+'.join(irreps)} | "
            f"{num_weights} weights)"
        )


class ComplexSwiGLU(torch.nn.Module):
    def __init__(
            self, 
            mmax: int, 
            lmax: int, 
            num_channel: int, 
            resolution: list[int],  
            use_m_primary: bool = True,
        ):
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channel =  num_channel
        self.resolution = resolution
        self.use_m_primary = use_m_primary
        self.tp = uuSO2TensorProduct(
            self.mmax,
            self.lmax,
            self.num_channel,
        )
        self.sigmoid = ScaledSigmoid()

    def forward(self, t, s):
        B = s.size(0)
        s = s.view(B, 1, 3*self.num_channel)
        o_s = s.narrow(2, 0, 2*self.num_channel)
        o_s1, o_s2 = torch.chunk(o_s, chunks=2, dim=-1)
        o_s = torch.nn.functional.silu(o_s1) * o_s2
        g_s = s.narrow(2, self.num_channel*2, self.num_channel)    
        g_s = self.sigmoid(g_s)
        t1, t2 = torch.chunk(t, chunks=2, dim=-1)
        o_t = self.tp(t1, t2)
        o_t = g_s * o_t
        o_t[:, 0:1, :] = o_t.narrow(1, 0, 1) + o_s
        return o_t


class VectorSwiGLU(torch.nn.Module):
    def __init__(
            self, 
            mmax: int, 
            lmax: int, 
            num_channel: int, 
            resolution: list[int],  
            use_m_primary: bool = True,
        ):
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channel =  num_channel
        self.resolution = resolution
        self.use_m_primary = use_m_primary
        self.grid = SO3VstpGrid(
            self.lmax,
            self.mmax,
            resolution_list=self.resolution,
            use_m_primary=self.use_m_primary,
        )
        self.sigmoid = ScaledSigmoid()

    def forward(self, t, s):
        B = s.size(0)
        s = s.view(B, 1, 3*self.num_channel)
        o_s = s.narrow(2, 0, 2*self.num_channel)
        o_s1, o_s2 = torch.chunk(o_s, chunks=2, dim=-1)
        o_s = torch.nn.functional.silu(o_s1) * o_s2
        g_s = s.narrow(2, self.num_channel*2, self.num_channel)    
        g_s = self.sigmoid(g_s)
        t1, t2 = torch.chunk(t, chunks=2, dim=-1)
        # o_t = self.grid.from_grid(
        #     self.grid.full_grid_product(
        #         self.grid.to_grid(t1),
        #         self.grid.to_grid(t2),
        #         symmetric_scale=1.0,
        #         antisym_scale=1.0,
        #     )
        # )
        o_t = self.grid.symmetric_product(t1, t2)
        # o_t = self.grid.antisymmetric_product(t1, t2)
        o_t = g_s * o_t
        o_t[:, 0:1, :] = o_t.narrow(1, 0, 1) + o_s
        return o_t


def satisfy(l1: int, l2: int, restriction: Union[str, None] = None) -> bool:
    if restriction == None:
        return True
    elif restriction == "<":
        return l1 < l2
    elif restriction == "<=":
        return l1 <= l2
    elif restriction == ">":
        return l1 > l2
    elif restriction == ">=":
        return l1 >= l2
    elif restriction == "==":
        return l1 == l2
    elif restriction == "!=":
        return l1 != l2
    else:
        raise ValueError(f"Unknown restriction: {restriction}")
    

class uuPlainSO2TensorProduct(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        m1m2: Union[str, None] = None,
    ):
        super().__init__()
        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.m1m2 = m1m2
        self.instructions = []
        for m3 in range(mmax + 1):
            paths = self.enumerate_paths(m3)
            self.instructions.append(paths)

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

    def cmul(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
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
        n = self.lmax + 1
        B = x.size(0)

        out = []
        offset = 0
        # m = 0
        out.append(x[:, offset:offset+n])
        offset += n
        # m > 0
        for m in range(1, self.mmax + 1):
            xm = x[:, offset:offset+2*n]
            xm = xm.view(B, 2, n, self.num_channel)
            out.append(xm)
            offset += 2 * n
        return out

    def forward(self, x: torch.Tensor,  y: torch.Tensor) -> torch.Tensor:
        xs = self.to_list(x) #  m = 0 [B, lmax+1, C]
        ys = self.to_list(y) #  m > 0 [B, 2, lmax+1, C]

        outputs = []

        # m = 0
        m0 = 0.0
        for m1, m2, mode in self.instructions[0]:
            # 0 x 0
            if m1 == 0 and m2 == 0:
                m0 = m0 + self.rmul(xs[0], ys[0])
            # m > 0 and m1 -m2 = 0
            elif m1 > 0 and m2 > 0:
                z = self.cmul(xs[m1], ys[m2], "diff")
                m0 = m0 + z[:, 0]# imag is also invariant, but nod add here
        outputs.append(m0)

        # m > 0
        for m3 in range(1, self.mmax + 1):
            real = 0.0
            imag = 0.0
            for m1, m2, mode in self.instructions[m3]:
                if m1 == 0:
                    z = xs[m1].unsqueeze(1) * ys[m2]
                elif m2 == 0:
                    z = xs[m1] * ys[m2].unsqueeze(1)
                else:
                    if m1 < m2 and mode == 'diff':
                        z = self.cmul(ys[m2], xs[m1], mode)
                    else:
                        z = self.cmul(xs[m1], ys[m2], mode)
                real = real + z[:, 0]
                imag = imag + z[:, 1]
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
        # lines.append(
        #     f"  mmax={self.mmax}, "
        #     f"lmax={self.lmax}, "
        #     f"channels={self.num_channel}, "
        # )
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
        lines.append(f"  total_paths={total_paths}")
        lines.append(")")
        return "\n".join(lines)


class ComplexSwiGLU(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        m1m2: Union[str, None] = '<=',
    ):
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_components = lmax+1
        self.num_channel = num_channel
        self.tp = uuPlainSO2TensorProduct(
            mmax, 
            lmax,
            num_channel, 
            m1m2=m1m2, 
        )
        self.sigmoid = ScaledSigmoid()

    def forward(self, t, s):
        B = s.size(0)
        s = s.view(B, 1, 3*self.num_channel)
        o_s = s.narrow(2, 0, 2*self.num_channel)
        o_s1, o_s2 = torch.chunk(o_s, chunks=2, dim=-1)
        o_s = torch.nn.functional.silu(o_s1) * o_s2
        g_s = s.narrow(2, self.num_channel*2, self.num_channel)    
        g_s = self.sigmoid(g_s)
        t1, t2 = torch.chunk(t, chunks=2, dim=-1)
        o_t = self.tp(t1, t2)
        o_t = g_s * o_t
        o_t[:, 0:1, :] = o_t.narrow(1, 0, 1) + o_s
        return o_t