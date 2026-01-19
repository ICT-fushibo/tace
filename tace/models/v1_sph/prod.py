###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import sys
from typing import Dict, List, Optional, Tuple


import torch
from e3nn import o3
from e3nn.o3 import TensorProduct


from .paths import generate_prod_paths
from .linear import WrapElementLinear



class SelfContraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        atomic_numbers: List[int] = [],
        prod: Dict = {},
        bias: bool = False,
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

        self.atomic_cluster_expansions = torch.nn.ModuleList()
        self.atomic_cluster_expansions_coefficients = torch.nn.ModuleList()
        self.atomic_cluster_expansions_coefficients.append(
            WrapElementLinear(
                num_elements=len(atomic_numbers),
                irreps_in=irreps_in.regroup(),
                irreps_out=irreps_out,
                bias=bias,
                cueq_config=None,
                oeq_config=None,
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

            self.atomic_cluster_expansions.append(
                TensorProduct(
                    irreps_in1,
                    irreps_in2,
                    irreps_mid,
                    instructions=instructions,
                    shared_weights=True,
                    internal_weights=True,
                    # cueq_config=None,
                    # oeq_config=None,
                )
            )
            # # print()
            # print(self.atomic_cluster_expansions[nu-2].irreps_out.regroup())
            # import sys 
            # sys.exit()
            # # sys.exit()

            irreps_in1 = irreps_mid
            irreps_in2 = irreps_in

            self.atomic_cluster_expansions_coefficients.append(
                WrapElementLinear(
                    num_elements=len(atomic_numbers),
                    irreps_in=irreps_mid.regroup(),
                    irreps_out=irreps_out,
                    bias=bias,
                    cueq_config=None,
                    oeq_config=None,
                )
            )

        self.linear = o3.Linear(
            irreps_in=irreps_out,
            irreps_out=irreps_out,
        )

    def forward(
        self,
        node_feats: Dict[int, torch.Tensor],
        node_attrs: torch.Tensor,
        sc: Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        
        corr_feats = {
            1: node_feats
        }

        for nu in range(2, self.correlation+1):
            idx = nu - 2
            this_ace = self.atomic_cluster_expansions[idx]
            corr_feats[nu] = this_ace(corr_feats[nu-1], node_feats)

        # nu = 1
        this_ace_coefs  = self.atomic_cluster_expansions_coefficients[0]

        node_feats = this_ace_coefs(corr_feats[1], node_attrs)
        
        # nu > 1
        for nu in range(2, self.correlation+1):
            idx = nu - 1
            this_ace_coefs  = self.atomic_cluster_expansions_coefficients[idx]
            node_feats += this_ace_coefs(corr_feats[nu], node_attrs)

        node_feats =  self.linear(node_feats)
        if sc is not None:
            node_feats =  node_feats + sc
        return node_feats

