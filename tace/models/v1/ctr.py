# ###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
# check: ✔
# ###############################################################################


from typing import Dict, List, Tuple


import torch
from torch import nn
from cartnn.util import scatter_sum


from .paths import count_irreps
from .linear import Linear
from .einsum import InterEinsumTC



class Contraction(torch.nn.Module):
    def __init__(
        self,
        combs: List[Tuple],
        ictp_lw: bool = False,
        ictc_lw: bool = False,
        ictp_hw: bool = True,
        ictc_hw: bool = True,
        num_channel: int = 64,
        num_channel_hidden: int = 64,
        lmax_in: int = 3,
        lmax_out: int = 3,
        kernel: str = 'scatter',
    ):
        super().__init__()

        # ==== ICTP + ICTC ====
        self.tcs = nn.ModuleList()
        for comb in combs:
            self.tcs.append(InterEinsumTC(comb))
        l3_count = count_irreps(
            combs, 
            ictp_lower_weight=ictp_lw, 
            ictc_lower_weight=ictc_lw, 
            ictp_highest_weight=ictp_hw,
            ictc_highest_weight=ictc_hw,
        )

        # === conv_weights slices ===
        self.ws_slices = []
        start = 0
        for _ in combs:
            stop = start + num_channel
            self.ws_slices.append(slice(start, stop))
            start = stop

        # === linear slices ===
        self.linear_slices = {}
        for l3, count in l3_count.items():
            self.linear_slices[l3] = []
            start = 0
            for _ in range(count):
                stop = start + num_channel
                self.linear_slices[l3].append(slice(start, stop))
                start = stop

        # === linear ===
        self.linear_downs = nn.ModuleList(
            [
                Linear(
                    num_channel * count,
                    num_channel_hidden,
                    bias=False,
                    l=l3,
                    in_channel=num_channel,
                    out_channel=num_channel_hidden,
                )
                for l3, count in l3_count.items()
            ]
        )

        self.combs = combs
        self.lmax_in = lmax_in
        self.lmax_out = lmax_out
        self.kernel = kernel
        self.num_channel = num_channel
        self.num_channel_hidden = num_channel_hidden

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

        if self.kernel == 'scatter':
            buffer = {l3: [] for l3 in range(self.lmax_out + 1)}
            for i, tc in enumerate(self.tcs):
                l1, l2, l3, _ = self.combs[i]
                out  = tc(x[l1], y[l2])
                w = ws[:, self.ws_slices[i]]
                for _ in range(out.ndim - 2):
                    w = w.unsqueeze(-1)
                w_out = w * out
                buffer[l3].append(w_out)
            m_ji = {}
            for i, linear in enumerate(self.linear_downs):
                m_ji[i] = linear(torch.cat(buffer[i], dim=1))
            m_i = {}
            for r in m_ji.keys():
                m_i[r] = scatter_sum(
                    src=m_ji[r],
                    index=edge_index[1],
                    dim=0,
                    dim_size=num_nodes,
                )
        else:
            dtype = x[0].dtype
            device = x[0].device
            m_i = {
                l3: torch.zeros(
                    num_nodes,
                    self.num_channel_hidden,
                    *((3,) * l3), 
                    device=device,
                    dtype=dtype,
                )
                for l3 in range(self.lmax_out + 1)
            }
            l3_count = {l3: 0 for l3 in range(self.lmax_out + 1)}
            for i, tc in enumerate(self.tcs):
                l1, l2, l3, _ = self.combs[i]
                out = tc(x[l1], y[l2])
                w = ws[:, self.ws_slices[i]]
                w = w.view(w.size(0), w.size(1), *((1,) * (out.ndim -2)))
                w_out = w * out
                tmp = self.linear_downs[l3](w_out, s=self.linear_slices[l3][l3_count[l3]])
                scattered = scatter_sum(
                    src=tmp,
                    index=edge_index[1],
                    dim=0,
                    dim_size=num_nodes,
                )
                m_i[l3].add_(scattered)
                l3_count[l3] += 1

        return m_i

