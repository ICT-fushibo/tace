################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
from typing import Dict, List, Any


import torch


from ..mlp import MLP
from ..radial import RadialBasis
from .ch import CartesianHarmonics
from .ue import UniversalInvariantEmbedding, UniversalEquivariantEmbedding
from ..lammps import Graph
from .node import NODE_EMBEDDING
from .edge import EDGE_EMBEDDING
from .inter import INTERACTION
from .prod import PRODUCT

class Representation(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        atomic_numbers: List[int],
        cutoff: float,
        avg_num_neighbors: float,
        Lmax: int,
        lmax: int,
        num_channel: int,
        target_weight: List[int],
        node_embedding: Dict,
        edge_embedding: Dict,
        radial_basis: Dict,
        angular_basis: Dict,
        atomic_basis: Dict,
        product_basis: Dict,
        invariant_property: List[str],
        equivariant_property: List[str],
        universal_embedding: Dict,
    ):
        super().__init__()

        target_weight = list(set(target_weight))
        self.num_elements = len(atomic_numbers)
        self.invariant_property = invariant_property
        self.equivariant_property = equivariant_property
        self.register_buffer('atomic_numbers', torch.tensor(atomic_numbers, dtype=torch.int64))

        # === radial basis ===
        self.radial_basis = RadialBasis(
            cutoff=cutoff,
            num_basis=radial_basis['num_radial_basis'],
            cutoff_fn=radial_basis['cutoff_fn'],
            polynomial_cutoff=radial_basis['polynomial_cutoff'],
            radial_basis=radial_basis['radial_basis'],
            distance_transform=radial_basis['distance_transform'],
            order=radial_basis['order'],
            trainable=radial_basis['trainable'],
            apply_cutoff=radial_basis['apply_cutoff'],
        )

        # === angular basis ===
        self.angular_basis = CartesianHarmonics(
            lmax,
            norm=angular_basis['norm'],
            traceless=angular_basis['traceless'],       
        )

        # === node/edge embedding ===
        self.node_embedding = NODE_EMBEDDING[node_embedding['type']](
            num_elements=self.num_elements,
            num_radial_basis=radial_basis['num_radial_basis'],
            num_channel=num_channel,
            bias=False,
        )
        self.edge_embedding = EDGE_EMBEDDING[edge_embedding['type']](
            num_elements=self.num_elements,
            num_radial_basis=radial_basis['num_radial_basis'],
            num_channel=num_channel,
            bias=False,
        )

        # if edge_embedding['update_edge_feats']:
        #     self.update_edge_feats = torch.nn.ModuleList(
        #         MLP(
        #             [num_channel * 2, num_channel, num_channel, self.edge_embedding.out_dim],
        #             bias=False,
        #             layer_norm=False,
        #             act='silu',
        #         ) for _ in range(1, num_layers)
        #     )

        # === universal embedding ===
        if len(self.invariant_property) > 0:
            self.uie_embedding = UniversalInvariantEmbedding(
                num_channel,
                {
                    k: v for k, v in universal_embedding.items() 
                    if k in self.invariant_property
                },
                bias=False,
            )
        if len(self.equivariant_property) > 0:
            self.uee_embedding = torch.nn.ModuleList()
            for _ in range(num_layers):
                self.uee_embedding.append(
                    UniversalEquivariantEmbedding(
                        {
                            k: v for k, v in universal_embedding.items() 
                            if k in self.equivariant_property
                        },
                        len(atomic_numbers),
                        num_channel, # norm bias hardcore to true
                    )
                )

        # === Interaction ===
        self.interactions = torch.nn.ModuleList(
            [
                INTERACTION[atomic_basis['type']](
                    layer=layer,
                    num_layers=num_layers,
                    num_elements=self.num_elements,
                    avg_num_neighbors=avg_num_neighbors,
                    Lmax=Lmax,
                    lmax=lmax,
                    num_channel=num_channel,
                    edge_feats_channel=self.edge_embedding.out_dim,
                    target_weight=target_weight,
                    num_radial_basis=radial_basis['num_radial_basis'],
                    radial_mlp=radial_basis['hidden'],
                    radial_bias=radial_basis['bias'],
                    l1l2=atomic_basis['l1l2'],
                    norm=atomic_basis['norm'],
                    resnet=atomic_basis['resnet'],
                    conv_weights=atomic_basis['conv_weights'],
                    ictp_ictc_like=atomic_basis['ictp_ictc_like'],
                    nonlinear=atomic_basis['nonlinear'],
                    has_linear_after_nonlinear=atomic_basis['has_linear_after_nonlinear'],
                    correlation=product_basis['correlation'],
                    bias=True,
                )
                for layer in range(num_layers)
            ]
        )

        # === Product ===
        self.products = torch.nn.ModuleList(
            [
                PRODUCT[product_basis['type']](
                    layer=layer,
                    num_layers=num_layers,
                    num_elements=self.num_elements,
                    Lmax=Lmax,
                    lmax=lmax,
                    num_channel=num_channel,
                    num_hidden_channel=product_basis['num_channel'],
                    target_weight=target_weight,
                    correlation=product_basis['correlation'],
                    l1l2=product_basis['l1l2'],     
                    num_latitude=product_basis['num_latitude'],
                    num_longitude=product_basis['num_longitude'],
                    truncation=product_basis['truncation'],
                    trainable_scale=product_basis['trainable_scale'],
                    bias=True,
                )
                for layer in range(num_layers)
            ]
        )
             
    def forward(self, data: Dict[str, torch.Tensor], graph) -> Dict[str, torch.Tensor]:
  
        lmp = graph.lmp
        nlocal, _ = graph.lmp_natoms

        # === edge initialize (radial) ===
        edge_feats, cutoff = self.radial_basis(
            graph.edge_length,
            data['node_attrs'],
            data['edge_index'],
            self.atomic_numbers,
        )

        # === node initialize ===
        node_feats = self.node_embedding(
            data['node_attrs'],
            edge_feats,
            data['edge_index'],
            cutoff,
        )

        if hasattr(self, "uie_embedding"):
            uie_data = {}
            for k in self.invariant_property:
                uie_data.update({k: data[k]})
            uie_feats = self.uie_embedding(data["batch"], uie_data)
            node_feats = node_feats + uie_feats
        else:
            uie_feats = None
        if hasattr(self, 'uee_embedding'):
            uee_data = {}
            for k in self.equivariant_property:
                uee_data.update({k: data[k]})

        edge_feats = self.edge_embedding(
            data['node_attrs'],
            edge_feats,
            data['edge_index'],
            cutoff,
        )
        edge_attrs = self.angular_basis(graph.edge_vector / graph.edge_length)

        # === representation Learning ===
        descriptors = []
        for idx, (inter, prod) in enumerate(zip(self.interactions, self.products)):
            node_attrs_total = data['node_attrs']
            node_attrs_slice = data['node_attrs']
            if lmp and idx > 0:
                node_attrs_slice = node_attrs_slice[:nlocal] 
            # if idx > 0 and hasattr(self, "update_edge_feats"):
            #     source, target = data['edge_index']
            #     this_node_feats = node_feats[0]
            #     edge_feats = edge_feats + self.update_edge_feats[idx-1](torch.cat([this_node_feats[source], this_node_feats[target]], dim=-1))
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
            if hasattr(self, 'uee_embedding'):
                node_feats = self.uee_embedding[idx](node_feats, node_attrs_slice, data["batch"], uee_data)
            node_feats = prod(node_feats, node_attrs_slice, sc)
            descriptors.append(node_feats)

        return {
            "descriptors": descriptors,
            "uie_feats": uie_feats,
        }

