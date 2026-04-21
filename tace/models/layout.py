################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch
from e3nn import o3

class LayoutTransform2(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.muls = []
        self.dims = []

        for mul, ir in self.irreps:
            d = ir.dim
            self.muls.append(mul)
            self.dims.append(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        start = 0
        out = []
        batch = x.size(0)
        dim = x.dim()
        if dim == 2:
            for mul, d in zip(self.muls, self.dims):
                field = x[:, start:start+mul*d]  
                start += mul * d
                field = field.reshape(batch, mul, d)
                out.append(field)
        else:
            head = x.size(1)
            for mul, d in zip(self.muls, self.dims):
                field = x[:, :, start:start+mul*d]  
                start += mul * d
                field = field.reshape(batch,  head, mul, d)
                out.append(field)
        return torch.cat(out, dim=-1)
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        start = 0
        out = []
        batch = x.size(0)
        dim = x.dim()
        if dim == 4:
            head = x.size(1)
            for _, d in zip(self.muls, self.dims):
                field = x[:, :, start:start+d]  
                start += d
                field = field.reshape(batch, head, -1)
                out.append(field)
        else: # dim = 3
            for _, d in zip(self.muls, self.dims):
                field = x[:, :, start:start+d]  
                start += d
                field = field.reshape(batch, -1)
                out.append(field)

        return torch.cat(out, dim=-1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps})"
    
class LayoutTransform(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.muls = []
        self.dims = []

        for mul, ir in self.irreps:
            d = ir.dim
            self.muls.append(mul)
            self.dims.append(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """e3nn to eqt layout"""
        start = 0
        out = []
        batch = x.size(0)
        for mul, d in zip(self.muls, self.dims):
            field = x[:, start:start+mul*d]  
            start += mul * d
            field = field.reshape(batch, mul, d)
            out.append(field)
        return torch.cat(out, dim=-1).transpose(-1, -2).contiguous()
    
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """eqt to e3nn layout"""
        x = x.transpose(-1, -2).contiguous()
        start = 0
        out = []
        batch = x.size(0)
        for _, d in zip(self.muls, self.dims):
            field = x[:, :, start:start+d]  
            start += d
            field = field.reshape(batch, -1)
            out.append(field)
        return torch.cat(out, dim=-1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps})"