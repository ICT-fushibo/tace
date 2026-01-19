# ###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
# check: ✔
# ###############################################################################


from typing import Dict, List, Tuple, Optional


import torch
from torch import nn


from .paths import count_irreps, generate_combinations
from .linear import Linear
from .einsum import InterEinsumTC
from ...utils.torch_scatter import scatter_sum


class Contraction(torch.nn.Module):
    """No operator fusion"""
    
    weight_numel: int
    
    def __init__(
        self,
        num_channel: int = 64,
        num_channel_hidden: int = 64,
        lmax_in: int = 3,
        lmax_out: int = 3,
        l1l2: Optional[str] = None,
        filter_combs: Optional[Tuple[int, int, int ,int]] = None,
    ):
        super().__init__()

        old_combs = generate_combinations(
            lmax_in,
            lmax_out,
            lmax_out,
            l1l2=l1l2,
        )

        assert isinstance(filter_combs, List) or filter_combs is None
        filter_combs = filter_combs or []

        new_combs = []
        for comb in old_combs:
            if list(comb) not in filter_combs:
                new_combs.append(comb)

        # ==== ICTP + ICTC ====
        self.tcs = nn.ModuleList()
        for comb in new_combs:
            self.tcs.append(InterEinsumTC(comb))
        l3_count = count_irreps(new_combs, False, False, True, True)

        # === conv_weights slices ===
        self.ws_slices = []
        start = 0
        for _ in new_combs:
            stop = start + num_channel
            self.ws_slices.append(slice(start, stop))
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

        self.combs = new_combs
        self.lmax_in = lmax_in
        self.lmax_out = lmax_out
        self.weight_numel = num_channel * len(new_combs)

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
            for _ in range(out.ndim - 2):
                w = w.unsqueeze(-1)
            w_out = w * out
            buffer[l3].append(w_out)

        # # === legacy ===
        # m_ji = {}
        # for i, linear in enumerate(self.linear_downs):
        #     m_ji[i] = linear(torch.cat(buffer[i], dim=1))
        # m_i = {}
        # for r in m_ji.keys():
        #     m_i[r] = scatter_sum(
        #         src=m_ji[r],
        #         index=edge_index[1],
        #         dim=0,
        #         dim_size=num_nodes,
        #     )

        m_ji = {}
        for l3 in range(self.lmax_out+1):
            m_ji[l3] = torch.cat(buffer[l3], dim=1)
        m_i = {}
        for r in m_ji.keys():
            m_i[r] = scatter_sum(
                src=m_ji[r],
                index=edge_index[1],
                dim=0,
                dim_size=num_nodes,
            )
        for i, linear in enumerate(self.linear_downs):
            m_i[i] = linear(m_i[i])

        return m_i

