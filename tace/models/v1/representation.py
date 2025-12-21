################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
from typing import Dict, List, Optional, Union, Any


import torch
from torch import Tensor, nn
from cartnn.math import RadialBasis
from cartnn.o3 import LegacyCartesianHarmonics2


from .mlp import MLP
from .inter import Interaction
from .prod import SelfContraction
from .embedding import UniversalInvariantEmbedding, UniversalEquivariantEmbedding
from .utils import Graph
from ...dataset.quantity import get_target_irreps


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
        universal_embedding: Optional[List[Dict[str, Union[int, str]]]] = None,
        use_nolinear_tensor_readout: bool = True,
        target_property: Dict = {},
        **kwargs,
    ):
        super().__init__()

        # === init ===
        self.register_buffer("num_layers", torch.tensor(num_layers, dtype=torch.int64))
        self.register_buffer("atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64))
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype()))

        # === target_irreps ===
        target_irreps = get_target_irreps(target_property, use_nolinear_tensor_readout)
        if max(Lmax) < max(target_irreps):
            raise ValueError(
                f"cfg.model.config.Lmax {max(Lmax)} should be greatet than"
                f"the tensor property you want to predict {max(target_property)}."
            )
    
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
        )

        # === universal embedding ===
        if universal_embedding is not None:
            self.invariant_embeddings = universal_embedding.get("invariant", None)
            self.equivariant_embeddings = universal_embedding.get("equivariant", None)
            if self.invariant_embeddings is not None:
                self.uie_embedding = UniversalInvariantEmbedding(
                    num_channel,
                    self.invariant_embeddings,
                )
            if self.equivariant_embeddings is not None:
                self.uee_embeddings = nn.ModuleList()
                for _ in range(num_layers):
                    self.uee_embeddings.append(
                        UniversalEquivariantEmbedding(
                            self.equivariant_embeddings,
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
                    atomic_numbers,
                    num_channel,
                    num_channel_hidden,
                    max(ls_in[idx]),
                    ls_out[idx],
                    max(ls_hidden[idx]),
                    avg_num_neighbors,
                    self.radial_embedding.out_dim,
                    radial_mlp,
                    inter,
                    bias,
                    layer=idx,
                    num_layers=num_layers,
                )
                for idx in range(num_layers)
            ]
        )

        # === Product Layer ===
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
 
    def forward(self, data: Dict[str, Tensor], graph: Graph) -> Dict[str, Any]:

        lmp = graph.lmp
        nlocal, _ = graph.lmp_natoms
        edge_vector = graph.edge_vector
        edge_length = graph.edge_length

        # === radial and angular ===
        edge_feats, cutoff = self.radial_embedding(
            edge_length,
            data['node_attrs'],
            data['edge_index'],
            self.atomic_numbers,
        )
        edge_attrs = {}
        normed_edge_vector = edge_vector / edge_length
        edge_attrs = self.angular_embedding(normed_edge_vector)

        # === node initialize (element and uie) ===
        node_feats = {0: self.node_embedding(data['node_attrs'])}
        uie_feats = None
        if hasattr(self, "uie_embedding"):
            uie_data = {}
            for k, _ in self.invariant_embeddings.items():
                p = k
                uie_data.update({p: data[p]})
            uie_feats = self.uie_embedding(data["batch"], uie_data)
            node_feats[0] = node_feats[0] + uie_feats

        # === representation Learning ===
        descriptors = []
        for idx, (inter, prod) in enumerate(zip(self.interactions, self.products)):
            node_attrs_lmp = data['node_attrs']
            if lmp and idx > 0:
                node_attrs_lmp = node_attrs_lmp[:nlocal]
            node_feats, sc = inter(
                node_feats,
                data['node_attrs'], 
                node_attrs_lmp, 
                edge_feats, 
                edge_attrs, 
                data['edge_index'],
                cutoff,
                graph,
            )
            if hasattr(self, 'uee_embeddings'):
                node_feats = self.uee_embeddings[idx](node_feats, data)
            if lmp and idx == 0:
                node_attrs_lmp = node_attrs_lmp[:nlocal]
            node_feats = prod(node_feats, node_attrs_lmp, sc)
            descriptors.append(node_feats)

        return {
            "uie_feats": uie_feats,
            "descriptors": descriptors,
        }

