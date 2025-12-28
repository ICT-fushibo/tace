###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from math import sqrt
from typing import Dict, List

import torch
from torch import Tensor, nn
from cartnn import o3

from .utils import add_to_left
from .layers import NormNonlinearity
from .linear import Linear, ElementLinear, CWLinear, ElementCWLinear, SelfInteraction
from .paths import satisfy, generate_prod_paths
from .einsum import ProdEinsumTC

PATH = 4
BATCH = 5
CHANNEL = 6

ProdLinear = {
    (False, True): Linear,
    (True, True): ElementLinear,
    (False, False): CWLinear,
    (True, False): ElementCWLinear,
}


class SelfContraction(torch.nn.Module):
    def __init__(
        self,
        num_channel: int = 64,
        num_channel_hidden: int = 64,
        lmax_in: int = 3,
        rank_of_out: List[int] = 2,
        atomic_numbers: List[int] = [],
        prod: Dict = {},
        bias: bool = False,
        layer: int = -1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        # === ICT ===
        for r in range(lmax_in + 1):
            DS = o3.ICTD(r, r)[1]
            self.register_buffer(f"D_{r}_{r}_1", DS[0].to(torch.get_default_dtype()))
            del DS
        
        # === used in init === 
        element_aware = prod.get("element_aware", True)
        coupled_channel = prod.get("coupled_channel", True)


        linear_type = {}
        for l in rank_of_out:
            if isinstance(coupled_channel, bool):
                    linear_type[l] = (element_aware, coupled_channel)
            elif isinstance(coupled_channel, dict):
                    linear_type[l] = (element_aware, coupled_channel.get(l, True))
            elif isinstance(coupled_channel, list):
                linear_type[l] = (element_aware, coupled_channel[layer].get(l, True))


        l1l2 = prod.get("l1l2", [None] * num_layers)[layer]
        l3l1 = prod.get("l3l1", [None] * num_layers)[layer]

        correlation = prod.get("correlation", [3,] * num_layers)[layer]
        max_left = prod.get("max_left", [[lmax_in] * correlation for _ in range(num_layers)])[layer]
        max_right = prod.get("max_right", [[lmax_in] * correlation for _ in range(num_layers)])[layer]
        max_hidden = prod.get("max_hidden", [[lmax_in] * correlation for _ in range(num_layers)])[layer]

        # === init ===
        self.cat = {}
        for k, v in linear_type.items():
            self.cat[k] = v[1]
        self.correlation = correlation
        self.lmax_in = lmax_in
        self.rank_of_out = rank_of_out
        self.layer = layer
  
        # === l3_count for each nu ===
        # nu = 1 
        nu_l3_count = {1: {l3: 0 for l3 in range(lmax_in + 1)}}  
        for l3 in range(max_hidden[0] + 1):
            nu_l3_count[1][l3] += 1

        # nu > 1
        for nu in range(2, self.correlation + 1):
            nu_l3_count[nu] = {l3: 0 for l3 in range(lmax_in + 1)} 
            for l1 in range(lmax_in + 1):
                for l2 in range(lmax_in+ 1):
                    for l3 in range(abs(l1 - l2), min(lmax_in, l1 + l2) + 1, 2):
                        if l1 <= max_left[nu-1] and l2 <= max_right[nu-1] and l3 <= max_hidden[nu-1]:
                            if satisfy(l1, l2, l1l2) and satisfy(l3, l1, l3l1):
                                nu_l3_count[nu][l3] += nu_l3_count[nu-1][l1]

        # === linear slices ===
        self.nu_linear_slices = {}  
        for nu, l3_count in nu_l3_count.items():
            linear_slices = {}
            for l3, count in l3_count.items():
                linear_slices[l3] = []
                start = 0
                for _ in range(count):
                    stop = start + num_channel_hidden
                    linear_slices[l3].append(slice(start, stop))
                    start = stop 
                self.nu_linear_slices[nu] = linear_slices
              

        # === lienar ===
        self.linears = nn.ModuleDict()
        for nu in range(1, self.correlation+1):
            inner_dict = nn.ModuleDict()
            for l3 in rank_of_out:
                if sum([nu_l3_count[nu][l3]]) > 0: # TODO for xzm, check if exists BUG
                    linear_layer = ProdLinear[linear_type[l3]](
                        num_channel_hidden * sum([nu_l3_count[nu][l3]]),
                        num_channel_hidden,
                        bias=(l3 == 0 and bias),
                        l=l3,
                        atomic_numbers=atomic_numbers,
                        groups=prod.get('groups', None),
                    )
                    inner_dict[str(l3)] = linear_layer
            self.linears[str(nu)] = inner_dict

        self.linear = SelfInteraction(
            in_channel=num_channel_hidden,
            out_channel=num_channel,
            rs=rank_of_out,
            bias=bias and layer == num_layers -1,
        )

        # === prod ===
        self.paths_list_list, self.exprs_list_list = generate_prod_paths(
            max_left, max_right, max_hidden, lmax_in, rank_of_out, self.correlation, l1l2, None, l3l1,
        )

        self.ctrs = nn.ModuleList()
        for v in range(self.correlation - 1):
            ctrs = nn.ModuleList()
            for comb, expr in zip(
                self.paths_list_list[v], self.exprs_list_list[v]
            ):
                # === expr ===
                ctrs.append(ProdEinsumTC((comb)))
            self.ctrs.append(ctrs)


    def D(self, l: int):
            return dict(self.named_buffers())[f"D_{l}_{l}_1"]
    
    def forward(
        self,
        node_feats: Dict[int, Tensor],
        node_attrs: Tensor,
        sc: Dict[int, Tensor],
    ) -> Dict[int, Tensor]:
        

        TMP = {
            0: {
                r: [node_feats[r]] for r in node_feats
            }
        }

        for v, ctrs in enumerate(self.ctrs):
            TMP[v + 1] = {r: [] for r in range(self.lmax_in + 1)}
            for idx, ctr in enumerate(ctrs):
                r_1, r_2, r_o, k = self.paths_list_list[v][idx]

                tmp = ctr(
                    torch.stack(TMP[v][r_1], dim=0),
                    node_feats[r_2],
                )

                P = tmp.size(0)
                B = tmp.size(1)
                C = tmp.size(2)
                REST = (3,) * r_o

                tmp = torch.bmm(
                    tmp.reshape(P, B * C, -1), self.D(r_o).repeat(P, 1, 1)
                ).reshape((P, B, C) + REST)

                tmp = torch.unbind(tmp, dim=0)
                TMP[v + 1][r_o].extend(tmp)

        outs = {}
        for nu_str, linears in self.linears.items():
            nu = int(nu_str)

            for l3_str, linear in linears.items():
                l3 = int(l3_str)

                if self.cat[l3]:
                    merged = torch.cat([t for t in TMP[nu-1][l3]], dim=1)
                else:
                    merged = torch.stack([t for t in TMP[nu-1][l3]], dim=0)

                out = linear(merged, node_attrs)

                if l3 in outs:
                    outs[l3] += out
                else:
                    outs[l3] = out

        return add_to_left(self.linear(outs), sc)
