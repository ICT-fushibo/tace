################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
from typing import Dict, List, Any


import torch
from torch import Tensor, nn


from .radial import RadialBasis
from .ch import LegacyCartesianHarmonics2
from .mlp import MLP
from .inter import Interaction
from .prod import SelfContraction, PrecomputedSelfContraction
from .embedding import UniversalInvariantEmbedding, UniversalEquivariantEmbedding
from .utils import Graph


class TACEDescriptor(torch.nn.Module):
    def __init__(
        self,
        cutoff: float,
        avg_num_neighbors: int,
        num_layers: int,
        atomic_numbers: List[int],
        Lmax: int,
        lmax: int,
        num_channel: List[int] = 64,
        num_channel_hidden: List[int] = 64,
        bias: bool = False,
        radial_basis: Dict = {},
        angular_basis: Dict = {},
        radial_mlp: Dict = {},
        inter: Dict = {},
        prod: Dict = {},
        universal_embedding: Dict[str, Dict[str, bool | int | float |str]] = {},
        target_irreps: List[str] = [],
        **kwargs,
    ):
        super().__init__()

        # === init ===
        self.invariant_embedding_property = universal_embedding['invariant_embedding_property']
        self.equivariant_embedding_property = universal_embedding['equivariant_embedding_property']
        self.register_buffer("num_layers", torch.tensor(num_layers, dtype=torch.int64))
        self.register_buffer("atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64))
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype()))

        # input, hiiden, output irreps
        ls_in = []      # in of inter
        ls_hidden = []  # out of inter and in of prod
        ls_out = []     # out of prod, out of sc
        for idx in range(num_layers):
            ls_hidden.append(list(range(lmax[idx] + 1)))
            ls_in.append([0]) if idx == 0 else ls_in.append(list(range(Lmax[idx] + 1)))
            ls_out.append(target_irreps if idx == num_layers - 1 else list(range(Lmax[idx] + 1)))

        # === element embedding ===
        self.node_embedding = MLP(
            len(atomic_numbers),
            num_channel,
            hidden_dim=[],
            act=None,
            bias=False,
            forward_weight_init=True,
            enable_layer_norm=False,
        )

        # === universal embedding ===
        if len(universal_embedding['invariant_embedding_property']) > 0:
            self.uie_embedding = UniversalInvariantEmbedding(
                num_channel,
                universal_embedding.get("invariant", {}),
            )
        if len(universal_embedding['equivariant_embedding_property']) > 0:
            self.uee_embeddings = nn.ModuleList()
            for _ in range(num_layers):
                self.uee_embeddings.append(
                    UniversalEquivariantEmbedding(
                        universal_embedding.get("equivariant", {}),
                        atomic_numbers,
                        num_channel,
                    )
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
        self.angular_embedding = LegacyCartesianHarmonics2(
            max(lmax), 
            angular_basis.get('norm', True), 
            angular_basis.get('traceless', True)
        )

        # === Interaction Layer ===
        self.interactions = nn.ModuleList(
            [
                Interaction(
                    idx,
                    num_layers,
                    num_channel,
                    num_channel_hidden,
                    bias,
                    max(ls_in[idx]),
                    max(ls_hidden[idx]),
                    ls_out[idx],
                    atomic_numbers,
                    avg_num_neighbors,
                    self.radial_embedding.out_dim,
                    radial_mlp,
                    inter,
                )
                for idx in range(num_layers)
            ]
        )

        # === Product Layer ===
        self.precompute = prod.get("precompute", False)
        if not self.precompute:
            self.products = nn.ModuleList(
                [
                    SelfContraction(
                        num_channel,
                        num_channel_hidden,
                        max(ls_hidden[idx]),
                        ls_out[idx],
                        atomic_numbers,
                        prod,
                        bias,
                        idx,
                        num_layers,
                    )
                    for idx in range(num_layers)
                ]
            )
        else:
             self.products = nn.ModuleList(
                [
                    PrecomputedSelfContraction(
                        num_channel,
                        num_channel_hidden,
                        max(ls_hidden[idx]),
                        ls_out[idx],
                        atomic_numbers,
                        prod,
                        bias,
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
        node_feats = {0: self.node_embedding(data['node_attrs'])}
        uie_feats = None
        if hasattr(self, "uie_embedding"):
            uie_data = {}
            for k in self.invariant_embedding_property:
                uie_data.update({k: data[k]})
            uie_feats = self.uie_embedding(data["batch"], uie_data)
            node_feats[0] = node_feats[0] + uie_feats
        if hasattr(self, 'uee_embeddings'):
            uee_data = {}
            for k in self.equivariant_embedding_property:
                    uee_data.update({k: data[k]})

        # === edge initialize (radial and angular) ===
        edge_feats, cutoff = self.radial_embedding(
            graph.edge_length,
            data['node_attrs'],
            data['edge_index'],
            self.atomic_numbers,
        )
        edge_attrs = {}
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
            if hasattr(self, 'uee_embeddings'):
                node_feats = self.uee_embeddings[idx](node_feats, node_attrs_slice, data["batch"], uee_data)
            node_feats = prod(node_feats, node_attrs_slice, sc)
            descriptors.append(node_feats)

        return {
            "uie_feats": uie_feats,
            "descriptors": descriptors,
        }

