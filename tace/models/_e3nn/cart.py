################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import sys 
import math

import torch
from e3nn import o3

from ..ictd import ICTD
from ..layout import LayoutTransform2
from .linear import ElementLinear

# torch.set_printoptions(sci_mode=False, precision=4)

class uuuICTC(torch.nn.Module):
    def __init__(self, lmax: int, num_channel: int, num_elements: int, irreps_out) -> None:
        super().__init__()
   
        self.lmax = lmax
        self.num_channel = num_channel
        self.irreps_in = [(num_channel, (l, (-1)**l)) for l in range(lmax+1)]
        self.reshape = LayoutTransform2(self.irreps_in)
        self.sph_slices = []
        self.cart_slices = []
        self.irreps_out = irreps_out
        self.num_elements = num_elements

        
        start1 = 0
        start2 = 0
        for l in range(self.lmax+1):
            self.sph_slices.append(slice(start1, start1+(2*l+1)))
            self.cart_slices.append(slice(start2, start2+(3**l)))
            start1 = start1 + start1+(2*l+1)
            start2 = start2 + start2+(3**l)

        for l in range(self.lmax+1):
            PS, DS, CS, SS = ICTD(l)
            self.register_buffer(f"S{l}", SS[0])

        for l in range(self.lmax+1):
            PS, DS, CS, SS = ICTD(l)
            self.register_buffer(f"C{l}", CS[0])

        self.paths = []
        for l1 in range(lmax+1):
            for l2 in range(lmax+1): 
                for l3 in range(abs(l1 - l2), min(lmax, l1 + l2) + 1, 2):  
                    k = (l1 + l2 - l3) // 2
                    if k > 0:
                        self.paths.append((l1, l2, l3, k))  
        self.paths.sort(key=lambda x: x[2])

        self.l3_count = {}
        for l1, l2, l3, k in self.paths:
            if l3 in self.l3_count:
                self.l3_count[l3] += 1
            else:
                self.l3_count[l3] = 1    

        self.linear = ElementLinear(
            o3.Irreps([(num_channel*count, (l3, (-1)**l3)) for l3, count in self.l3_count.items()]).regroup(),
            self.irreps_out,
            bias=True,
            num_elements=self.num_elements,
        )

    def forward(self, x: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        B, C = x.size(0), self.num_channel

        x = self.reshape(x)

        x_dict = {}
        for l in range(self.lmax+1):
            this_x = x[:, :, self.sph_slices[l]]
            this_x = torch.einsum('bci, ij -> bcj', this_x, getattr(self, f"S{l}"))
            x_dict[l] = this_x

        out_dict = {l: [] for l in range(self.lmax+1)}

        for l1, l2, l3 , k in self.paths:
            x = x_dict[l1].reshape(B, C, 3**(l1-k), 3**k)
            y = x_dict[l2].reshape(B, C, 3**k, 3**(l2-k))
            z = torch.einsum('bcik, bckj -> bcij', x, y).view(B, C, -1) / math.sqrt(3**k)
            z = torch.einsum('bci, ij -> bcj', z, getattr(self, f"C{l3}")).view(B, -1)
            out_dict[l3].append(z)

        out_list = []
        for l3 in sorted(out_dict.keys()):
            if len(out_dict[l3]) > 0:
                out_list.append(
                    torch.cat(out_dict[l3], dim=1).view(B, -1)
                )
        out = torch.cat(out_list, dim=-1)
    
        return self.linear(out, node_attrs)