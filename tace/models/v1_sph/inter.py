################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
"""
So far, all the functions have been concentrated within one interaction module,
and LAMMPS utils are modified from MACE
"""
import sys 
import abc
from typing import Dict, List, Tuple, Optional, Any


import torch
from torch import Tensor


from .acc import FusedTensorProduct, AccLinear, AccElementLinear


from tace.models.v1.mlp import MLP
from tace.models.v1.utils import Graph, LAMMPS_MP
from .acc import AccElementLinear

from ...utils.torch_scatter import scatter_sum

from .paths import generate_inter_paths

class InteractionBase(torch.nn.Module):
    def __init__(
        self,
        layer: int,
        num_layers: int,
        num_channel: int,
        irreps_in: int,
        irreps_out: int,
        irreps_sc: List[int],
        sh_irreps: str, 
        atomic_numbers: int,
        avg_num_neighbors: float,
        num_radial_basis: int,
        radial_mlp: Dict,
        inter: Dict,
        enable_oeq: bool
    ) -> None:
        super().__init__()
        self.layer = layer
        self.num_layers = num_layers
        self.num_channel = num_channel
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.irreps_sc = irreps_sc
        self.sh_irreps = sh_irreps
        self.atomic_numbers = atomic_numbers
        self.num_elements = len(atomic_numbers)
        self.num_radial_basis = num_radial_basis
        self.radial_mlp = radial_mlp
        self.inter = inter
        self.avg_num_neighbors = avg_num_neighbors
        self.enable_oeq = enable_oeq
        self._setup()
    
    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError



class Interaction(InteractionBase):
    def _setup(self) -> None:
        # # === resnet === 
        # if self.inter.get('use_resnet', False):
        #     self.resnet = AccElementLinear(
        #         num_elements=self.num_elements,
        #         irreps_in=self.irreps_in,
        #         irreps_out=self.irreps_in,
        #         biases=self.bias,
        #         cueq_config=None,
        #         oeq_config=None,
        #     )

        # === linear up ===  
        self.linear_up = AccLinear(
                irreps_in=self.irreps_in,
                irreps_out=self.irreps_in,
            )

        # === ICTP and ICTC ===
        l1l2 = self.inter.get("l1l2", [None] * self.num_layers)[self.layer]

        irreps_mid, instructions = generate_inter_paths(
            self.irreps_in,
            self.sh_irreps,
            self.irreps_out,
            l1l2=l1l2,
        )

        self.conv_tp = FusedTensorProduct(
            self.irreps_in,
            self.sh_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            enable_oeq=self.enable_oeq
        )

        self.linear = AccLinear(
            irreps_mid.regroup(),
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

        # === self / skip connection ===
        self.sc_from = None
        if self.inter.get('sc', {}).get('use_first_sc', False) \
        or self.layer > 0 \
        or self.num_layers == 1:
            self.sc_from = self.inter.get('sc', {}).get('from', "current_message") 
            self.scs = AccElementLinear(
                num_elements=self.num_elements,
                irreps_in=self.irreps_out,
                irreps_out=self.irreps_sc,
            )
                    
        # ==== conv weights ====
        conv_weights_type = self.inter.get('conv_weights', ['edge_ij'])
        assert 'edge_ij' in conv_weights_type, 'edge_ij must be in conv_weights' 
        radial_in_dim = self.num_radial_basis + self.num_channel * (len(conv_weights_type)-1)
        need_layer_norm = radial_in_dim != self.num_radial_basis
        self.radial_net = MLP(
            radial_in_dim,
            self.conv_tp.weight_numel,
            self.radial_mlp["hidden"][self.layer],
            act=self.radial_mlp["act"],
            bias=self.radial_mlp.get('bias', False),
            forward_weight_init=True,
            enable_layer_norm=need_layer_norm,
        )

        # ==== normalizer ====
        self.normalizer = self.inter.get('normalizer', 'avg_num_neighbors')
        assert self.normalizer in ["avg_num_neighbors", "density_v1"]
    
    def forward(
        self,
        node_feats: Dict[int, Tensor],
        node_attrs_total: Tensor,
        node_attrs_slice: Tensor,
        edge_feats: Tensor,
        edge_attrs: Dict[int, Tensor],
        edge_index: Tensor,
        cutoff: Tensor,
        graph: Graph,
    ) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:

        # === LAMMPS pretreatment ===
        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        # === linear up === 
        node_feats = self.linear_up(node_feats)
 
        # === conv_weights === 
        edge_embedding: list[torch.Tensor] = [edge_feats]
        full_edge_feats = torch.cat(edge_embedding, dim=-1)
        conv_weights = self.radial_net(full_edge_feats)
        if cutoff is not None:
            conv_weights = conv_weights * cutoff

        # === ICTP and ICTC ===
        if self.enable_oeq:
            m_i = self.conv_tp(
                node_feats, edge_attrs, conv_weights, edge_index[0], edge_index[1]
            )
        else:
            m_ij = self.conv_tp(node_feats[edge_index[0]], edge_attrs, conv_weights)
            m_i = scatter_sum(
                src=m_ij, index=edge_index[1], dim=0, dim_size=node_feats.shape[0]
            )
            
        m_i = self.linear(m_i) / self.avg_num_neighbors
        scs = None
        if self.sc_from == 'current_message':
            scs = self.scs(m_i, node_attrs_total)
 
        return m_i, scs



