###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import sys
from typing import Dict, List, Optional


import torch
from e3nn import o3
from e3nn.o3 import TensorProduct


from .paths import generate_prod_paths
from .acc import AccLinear, AccElementLinear


class SelfContraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        atomic_numbers: List[int] = [],
        prod: Dict = {},
        layer: int = -1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        # === used in init === 
        l1l2 = prod.get("l1l2", [None] * num_layers)[layer]
        l3l1 = prod.get("l3l1", [None] * num_layers)[layer]
        correlation = prod.get("correlation", [3,] * num_layers)[layer]

        # === init ===
        self.correlation = correlation
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        self.aces = torch.nn.ModuleList()
        self.coefs = torch.nn.ModuleList()
        self.coefs.append(
            AccElementLinear(
                num_elements=len(atomic_numbers),
                irreps_in=irreps_in.regroup(),
                irreps_out=irreps_out,
            )
        )
        
        irreps_in1 = irreps_in
        irreps_in2 = irreps_in

        for nu in range(2, self.correlation+1):
            # print(nu)

            if nu == correlation:
                target_irreps = irreps_out # TODO, deleted path
            else:
                target_irreps = irreps_in

            irreps_mid, instructions = generate_prod_paths(
                irreps_in1,
                irreps_in2,
                target_irreps,
                l1l2=l1l2,
                l2l3=None,
                l3l1=l3l1,
            )

            self.aces.append(
                TensorProduct(
                    irreps_in1,
                    irreps_in2,
                    irreps_mid,
                    instructions=instructions,
                    shared_weights=True,
                    internal_weights=True,
                )
            )

            irreps_in1 = irreps_mid
            irreps_in2 = irreps_in

            self.coefs.append(
                AccElementLinear(
                    num_elements=len(atomic_numbers),
                    irreps_in=irreps_mid.regroup(),
                    irreps_out=irreps_out,
                )
            )

        self.linear = AccLinear(
            irreps_in=irreps_out,
            irreps_out=irreps_out,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        sc: Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        corr_feats = {
            1: node_feats
        }
        for nu in range(2, self.correlation+1):
            idx = nu - 2
            corr_feats[nu] = self.aces[idx](corr_feats[nu-1], node_feats)
        # nu = 1
        node_feats = self.coefs[0](corr_feats[1], node_attrs)
        # nu > 1 
        # It should be vectorized among paths, 
        # but the corresponding implementation was not found in cueq and oeq
        for nu in range(2, self.correlation+1):
            idx = nu - 1
            node_feats += self.coefs[idx](corr_feats[nu], node_attrs)
        node_feats =  self.linear(node_feats)
        if sc is not None:
            node_feats =  node_feats + sc
        return node_feats

