################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
import sys
from typing import Dict, List, Any


import torch
from torch import Tensor, nn
from e3nn import o3


from tace.models.v1.radial import RadialBasis
from tace.models.v1.utils import Graph
from .layers import LinearNodeEmbedding
from .inter import Interaction
from .prod import SelfContraction


class SphTACEDescriptor(torch.nn.Module):
    def __init__(
        self,
        cutoff: float,
        ls_in: List[o3.Irreps],
        ls_hidden: List[o3.Irreps],
        ls_out: List[o3.Irreps],
        sh_irreps: o3.Irreps,
        avg_num_neighbors: int,
        num_layers: int,
        atomic_numbers: List[int],
        num_channel: List[int] = 64,
        radial_basis: Dict = {},
        radial_mlp: Dict = {},
        inter: Dict = {},
        prod: Dict = {},
        group: str = "SO(3)",
        enable_oeq: bool = False,
        **kwargs,
    ):
        super().__init__()
        # === init ===
        self.register_buffer("num_layers", torch.tensor(num_layers, dtype=torch.int64))
        self.register_buffer("atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64))
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype()))

        # === element embedding ===
        self.node_embedding = LinearNodeEmbedding(
            irreps_in=o3.Irreps([(len(atomic_numbers), (0, 1))]),
            irreps_out=o3.Irreps([(num_channel, (0, 1))]),
        )

        # === radial basis ===
        self.radial_embedding = RadialBasis(
            cutoff=cutoff,
            num_basis=radial_basis.get('num_radial_basis', 8),
            polynomial_cutoff=radial_basis.get('polynomial_cutoff', 5),
            radial_basis=radial_basis.get('radial_basis', 'j0'),
            distance_transform=radial_basis.get('distance_transform', None),
            order=radial_basis.get('order', 0),
            trainable=radial_basis.get('trainable', False),
            apply_cutoff=radial_basis.get("apply_cutoff", True)
        )

        # === angular basis ===
        self.angular_embedding = o3.SphericalHarmonics(
            sh_irreps, 
            normalize=True, 
            normalization="component",
        )

        # === Interaction Layer ===
        self.interactions = nn.ModuleList(
            [
                Interaction(
                    idx,
                    num_layers,
                    num_channel,
                    ls_in[idx],
                    ls_hidden[idx],
                    ls_out[idx],
                    sh_irreps,
                    atomic_numbers,
                    avg_num_neighbors,
                    self.radial_embedding.out_dim,
                    radial_mlp,
                    inter,
                    enable_oeq,
                )
                for idx in range(num_layers)
            ]
        )
 
        # === Product Layer ===
        self.precompute = prod.get("precompute", False)
        self.products = nn.ModuleList(
            [
                SelfContraction(
                    ls_hidden[idx],
                    ls_out[idx],
                    atomic_numbers,
                    prod,
                    idx,
                    num_layers,
                )
                for idx in range(num_layers)
            ]
        )

    def forward(self, data: Dict[str, Tensor], graph: Graph) -> Dict[str, Any]:

        lmp = graph.lmp
        nlocal, _ = graph.lmp_natoms
  
        # === node initialize (element and uie) ===
        node_feats = self.node_embedding(data['node_attrs'])

        # === edge initialize (radial and angular) ===
        edge_feats, cutoff = self.radial_embedding(
            graph.edge_length,
            data['node_attrs'],
            data['edge_index'],
            self.atomic_numbers,
        )
        normed_edge_vector = graph.edge_vector / graph.edge_length
        edge_attrs = self.angular_embedding(normed_edge_vector)

        # === representation Learning ===
        descriptors = []
        for idx, (inter, prod) in enumerate(zip(self.interactions, self.products)):
            node_attrs_total = data['node_attrs']
            node_attrs_slice = data['node_attrs']
            if lmp and idx > 0:
                node_attrs_slice = node_attrs_slice[:nlocal] 
            node_feats, sc = inter(
                node_feats,
                node_attrs_total, 
                node_attrs_slice, 
                edge_feats, 
                edge_attrs, 
                data['edge_index'],
                cutoff,
                graph,
            )
            if lmp and idx == 0:
                node_attrs_slice = node_attrs_slice[:nlocal] # nlocal
            node_feats = prod(node_feats, node_attrs_slice, sc)
            descriptors.append(node_feats)

        return {
            "descriptors": descriptors,
        }

