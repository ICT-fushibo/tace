###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


from typing import Dict, List


import torch
from cartnn import ICTD, Irreps, SymmetricContraction


from .utils import add_dict_to_left
from .paths import satisfy, generate_prod_paths
from .linear import SelfInteraction, LinearDict
from .einsum import ProdEinsumTC

PATH = 4
BATCH = 5
CHANNEL = 6


class SelfContraction(torch.nn.Module):
    def __init__(
        self,
        num_channel: int = 64,
        num_channel_hidden: int = 64,
        lmax_in: int = 3,
        ls_out: List[int] = 2,
        atomic_numbers: List[int] = [],
        prod: Dict = {},
        bias: bool = False,
        layer: int = -1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        # === ICT ===
        for r in range(lmax_in + 1):
            DS = ICTD(r, r)[1]
            self.register_buffer(f"D_{r}_{r}_1", DS[0].to(torch.get_default_dtype()))
            del DS
        
        # === used in init === 
        l1l2 = prod.get("l1l2", [None] * num_layers)[layer]
        l3l1 = prod.get("l3l1", [None] * num_layers)[layer]
        correlation = prod.get("correlation", [3,] * num_layers)[layer]

        if isinstance(prod.get("element_aware", True), bool):
            element_aware = {}
            for l in ls_out:
                element_aware[l] = prod.get("element_aware", True)
        else:
             element_aware = prod["element_aware"]
        if isinstance(prod.get("coupled_channel", True), bool):
            coupled_channel = {}
            for l in ls_out:
                coupled_channel[l] = prod.get("coupled_channel", True)
        else:
            coupled_channel = prod["coupled_channel"]

        linear_type = {}
        for l in ls_out:
            linear_type[l] = (element_aware[l], not coupled_channel[l])

        # === init ===
        self.correlation = correlation
        self.lmax_in = lmax_in
        self.ls_out = ls_out
        self.layer = layer
        self.coupled_channel = coupled_channel

        # Below TODO for xzm
  
        # === prod ===
        self.paths_list_list, self.exprs_list_list = generate_prod_paths(
            lmax_in, ls_out, self.correlation, l1l2, None, l3l1,
        )
        
        filter_idx = prod.get("filter_idx", None)
        if filter_idx:
            assert self.correlation == 2, (
                "filter_idx can only be used when correlation = 2, "
                "because implementing higher-order cases is cumbersome"
            )
            filter_nu2_combs = []
            filter_nu2_exprs = []
            nu2_combs = self.paths_list_list[0]
            nu2_exprs = self.exprs_list_list[0]
            for idx, (comb, expr) in enumerate(zip(nu2_combs, nu2_exprs)):
                if idx not in filter_idx:
                    filter_nu2_combs.append(comb)
                    filter_nu2_exprs.append(expr)
            self.paths_list_list[0] = filter_nu2_combs
            self.exprs_list_list[0] = filter_nu2_exprs

        self.ctrs = torch.nn.ModuleList()
        for v in range(self.correlation - 1):
            ctrs = torch.nn.ModuleList()
            for comb, expr in zip(
                self.paths_list_list[v], self.exprs_list_list[v]
            ):
                ctrs.append(ProdEinsumTC((comb)))
            self.ctrs.append(ctrs)

        # === l3_count for each nu ===
        # nu = 1 
        nu_l3_count = {1: {l3: 0 for l3 in range(lmax_in + 1)}}  
        for l3 in range(lmax_in + 1):
            nu_l3_count[1][l3] += 1

        # nu > 1
        for nu in range(2, self.correlation + 1):
            nu_l3_count[nu] = {l3: 0 for l3 in range(lmax_in + 1)} 
            for l1 in range(lmax_in + 1):
                for l2 in range(lmax_in+ 1):
                    for l3 in range(abs(l1 - l2), min(lmax_in, l1 + l2) + 1, 2):
                        k = (l1 + l2 - l3) // 2
                        if satisfy(l1, l2, l1l2) and satisfy(l3, l1, l3l1):
                            if filter_idx:
                                for comb in filter_nu2_combs:
                                    this_comb = (l1, l2, l3 ,k)
                                    if this_comb == comb:
                                        nu_l3_count[nu][l3] += nu_l3_count[nu-1][l1]
                            else:
                                nu_l3_count[nu][l3] += nu_l3_count[nu-1][l1]


        # === lienar ===
        self.linears = torch.nn.ModuleDict()
        for nu in range(1, self.correlation+1):
            inner_dict = torch.nn.ModuleDict()
            for l3 in ls_out:
                if sum([nu_l3_count[nu][l3]]) > 0: 
                    linear_layer = LinearDict[linear_type[l3]](
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
            ls=ls_out,
            bias=bias and layer == num_layers -1,
        )


    def D(self, l: int):
            return dict(self.named_buffers())[f"D_{l}_{l}_1"]
    
    def forward(
        self,
        node_feats: Dict[int, torch.Tensor],
        node_attrs: torch.Tensor,
        sc: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        
        corr_feats = {
            0: {
                l: [node_feats[l]] for l in node_feats
            }
        }

        for nu, ctrs in enumerate(self.ctrs):
            corr_feats[nu + 1] = {l: [] for l in range(self.lmax_in + 1)}
            for idx, ctr in enumerate(ctrs):
                l1, l2, l3, _ = self.paths_list_list[nu][idx]
                tmp = ctr(
                    torch.stack(corr_feats[nu][l1], dim=0),
                    node_feats[l2],
                )

                P = tmp.size(0); B = tmp.size(1); C = tmp.size(2)

                tmp = torch.bmm(
                    tmp.reshape(P, B * C, -1), self.D(l3).repeat(P, 1, 1)
                ).reshape((P, B, C) + (3,) * l3)

                tmp = torch.unbind(tmp, dim=0)
                corr_feats[nu + 1][l3].extend(tmp)

        out_dict = {}
        for nu_str, linears in self.linears.items():
            nu = int(nu_str)

            for l3_str, linear in linears.items():
                l3 = int(l3_str)

                if self.coupled_channel[l3]:
                    merged = torch.cat([t for t in corr_feats[nu-1][l3]], dim=1)
                else:
                    merged = torch.stack([t for t in corr_feats[nu-1][l3]], dim=0)

                out = linear(merged, node_attrs)

                if l3 in out_dict:
                    out_dict[l3] += out
                else:
                    out_dict[l3] = out

        return add_dict_to_left(self.linear(out_dict), sc)


class PrecomputedSelfContraction(torch.nn.Module):
    def __init__(
        self,
        num_channel: int = 64,
        num_channel_hidden: int = 64,
        lmax_in: int = 3,
        ls_out: List[int] = 2,
        atomic_numbers: List[int] = [],
        prod: Dict = {},
        bias: bool = False,
        layer: int = -1,
        num_layers: int = 2,
    ) -> None:
        
        '''
        This product basis is different from original TACE's SelfContraction.
        Simply put, 
        Self contraction: tensor product + tensor contraction;
        PrecomputedSelfContraction: tensor product and precomute cartesian_nj.
        '''
        super().__init__()

        self.lmax_in = lmax_in
        self.ls_out = ls_out
        self.num_channel = num_channel
        self.num_channel_hidden = num_channel_hidden

        if isinstance(prod.get("element_aware", True), bool):
            element_aware = {}
            for l in ls_out:
                element_aware[l] = prod.get("element_aware", True)
        else:
             element_aware = prod["element_aware"]
        if isinstance(prod.get("coupled_channel", True), bool):
            coupled_channel = {}
            for l in ls_out:
                coupled_channel[l] = prod.get("coupled_channel", True)
        else:
            coupled_channel = prod["coupled_channel"]

        correlation = prod.get("correlation", [3,] * num_layers)[layer]
        assert correlation == 2, "Only nu=2 precomputed product basis are useful in Cartesian space "
    
        node_feats_irreps = "+".join(
            f"{num_channel_hidden}x{l}e" for l in range(lmax_in + 1)
        )
        target_irreps = "+".join(f"{num_channel_hidden}x{l}e" for l in ls_out)

        self.symmetric_contractions = SymmetricContraction(
            irreps_in=Irreps(node_feats_irreps),
            irreps_out=Irreps(target_irreps),
            correlation=correlation,
            num_elements=len(atomic_numbers),
            element_aware=element_aware,
            coupled_channel=coupled_channel,
        )

        self.linear = SelfInteraction(
            in_channel=num_channel_hidden,
            out_channel=num_channel,
            ls=ls_out,
            bias=bias and layer == num_layers -1,
        )

    def forward(
        self,
        node_feats: Dict[int, torch.Tensor],
        node_attrs: torch.Tensor,
        sc: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        
        B = node_feats[0].size(0)
        C = node_feats[0].size(1)

        node_feats_list = []
        for l in range(self.lmax_in + 1):
            node_feats_list.append(node_feats[l].view(B, C, -1))
        node_feats = torch.cat(node_feats_list, dim=-1)
        node_feats = self.symmetric_contractions(node_feats, node_attrs)

        out_dict = {}
        for idx, l in enumerate(self.ls_out):
            out_dict[l] = node_feats[idx].view(*((B, self.num_channel) + (3,) * l))

        return add_dict_to_left(self.linear(out_dict), sc)