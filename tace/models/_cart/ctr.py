# ###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
# check: ✔
# ###############################################################################


from typing import Dict, List, Tuple, Optional


import torch
from torch import nn
from torch_scatter import scatter_sum
from cartnn import ICTD

from ..utils import expand_dims_to
from .einsum import InterEinsumTC


class Contraction(torch.nn.Module):
    """No operator fusion"""
    
    weight_numel: int
    
    def __init__(
        self,
        lmax_in: int,
        lmax_out: int,
        num_channel: int,
        combs,
    ):
        super().__init__()

        # ==== ICTP + ICTC ====
        self.tcs = nn.ModuleList()
        for comb in combs:
            self.tcs.append(InterEinsumTC(comb))

        # === conv_weights slices ===
        self.ws_slices = []
        start = 0
        for _ in combs:
            stop = start + num_channel
            self.ws_slices.append(slice(start, stop))
            start = stop

        self.combs = combs
        self.lmax_in = lmax_in
        self.lmax_out = lmax_out
        self.weight_numel = num_channel * len(combs)

        for l in range(lmax_out+1):
            DS = ICTD(l, l)[1]
            self.register_buffer(f"D_{l}_{l}_1", DS[0].to(torch.get_default_dtype()), persistent=False)
            del DS

    def D(self, l: int):
        return dict(self.named_buffers())[f"D_{l}_{l}_1"]
    
    def forward(
        self,
        x: Dict[int, torch.Tensor],
        y: Dict[int, torch.Tensor],
        ws: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:

        num_nodes = x[0].size(0)
        for l1 in range(self.lmax_in + 1):
            x[l1] = x[l1][edge_index[0]]

        buffer = {l3: [] for l3 in range(self.lmax_out + 1)}
        for i, tc in enumerate(self.tcs):
            l1, l2, l3, _ = self.combs[i]
            out  = tc(x[l1], y[l2])
            w = ws[:, self.ws_slices[i]]
            w = expand_dims_to(w, out.ndim, dim=1)
            w_out = w * out
            buffer[l3].append(w_out)

        m_ji = {}
        for l3 in range(self.lmax_out+1):
            m_ji[l3] = torch.cat(buffer[l3], dim=-1)
        m_i = {}
        for r in m_ji.keys():
            m_i[r] = scatter_sum(
                src=m_ji[r],
                index=edge_index[1],
                dim=0,
                dim_size=num_nodes,
            )
            
        outs = {}
        for r, t in m_i.items():
            B = t.size(0)
            C = t.size(-1)
            REST = (3,) * r
            outs[r] = torch.einsum("bic, ij -> bjc", t.reshape(B, -1, C), self.D(r)).reshape((B,) + REST + (C,))

        return outs

