################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import sys
import math
from typing import Dict, List


import torch
from e3nn import o3


from ...utils.env import get_tace_use_dens
from ..radial import RadialBasis
from ..angular import SphericalHarmonics
from ..so2 import SO3Rotation
from ..layout import LayoutTransform
from .node import NODE_EMBEDDING
from .edge import EDGE_EMBEDDING, EDGE_UPDATE
from .inter import INTERACTION # , SO2EdgeInteraction
from .prod import PRODUCT
from .ue import UniversalInvariantEmbedding, UniversalEquivariantEmbedding
from .layer_norm import get_normalization_layer
from ..linear import e3nnLinear



class Representation(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        atomic_numbers: List[int],
        cutoff: float,
        avg_num_neighbors: float,
        mmax: int,
        Lmax: int,
        lmax: int,
        num_channel: int,
        target_irreps: o3.Irreps,  
        node_embedding: Dict,
        edge_embedding: Dict,
        edge_update: Dict,
        radial_basis: Dict,
        atomic_basis: Dict,
        resnet: Dict,
        product_basis: Dict,
        invariant_property: List[str],
        equivariant_property: List[str],
        universal_embedding: Dict,
        layer_norm: Dict,
        dropout: Dict,
        parity: bool,
    ):
        super().__init__()

        self.num_elements = len(atomic_numbers)
        self.num_channel = num_channel
        self.num_layers = num_layers
        self.invariant_property = invariant_property
        self.equivariant_property = equivariant_property
        self.register_buffer('atomic_numbers', torch.tensor(atomic_numbers, dtype=torch.int64))
        self.resnet_type = resnet['type']
        self.use_dens = get_tace_use_dens() == '1'

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
            gaussian_width=radial_basis['gaussian_width'],
            use_dydynamic_cutoff=radial_basis['use_dydynamic_cutoff'],
            dydynamic_cutoff_mu=radial_basis['dydynamic_cutoff_mu'],
            num_elements=len(atomic_numbers),
        )
        
        # === angular basis ===
        self.use_so2 = (
            any(t.endswith('so2') for t in atomic_basis['type'])
            or node_embedding["type"] == 'so2_tensor' 
            or True in atomic_basis["use_graph_softmax"]
        )
        self.use_o3 = any(t != 'so2' for t in atomic_basis['type']) or node_embedding["type"] == 'tensor'
        if self.use_so2:
            # assert Lmax == lmax, "SO2Interaciton require Lmax == lmax in TACE"
            self.so2_angular_basis = SO3Rotation(Lmax, mmax, use_rotation_mask=True)
        if self.use_o3:
            self.o3_angular_basis = SphericalHarmonics(
                o3.Irreps.spherical_harmonics(lmax, p=-1),
                normalize=False,
                normalization="component",
            )

        # === node/edge embedding ===
        self.node_embedding = NODE_EMBEDDING[node_embedding['type']](
            num_elements=self.num_elements,
            num_radial_basis=self.radial_basis.num_basis,
            num_channel=num_channel,
            Lmax=Lmax,
            lmax=lmax,
            avg_num_neighbors=avg_num_neighbors,
            bias=False,
            so2_angular_basis=self.so2_angular_basis if self.use_so2 else None
        )
        self.edge_embedding = EDGE_EMBEDDING[edge_embedding['type']](
            num_elements=self.num_elements,
            num_radial_basis=self.radial_basis.num_basis,
            num_channel=num_channel,
            bias=False,
        )

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

        # === Edge Update ===
        self.edge_updates = torch.nn.ModuleList(
            [
                EDGE_UPDATE[edge_update['type']](
                    layer=layer,
                    num_layers=num_layers,
                    num_elements=self.num_elements,
                    num_radial_basis=self.radial_basis.num_basis,
                    edge_embedding_channel=self.edge_embedding.out_dim,
                    num_channel=num_channel,
                )
                for layer in range(num_layers)
            ]
        )

        # === Interaction ===
        for_interactions = {
            "num_layers": num_layers,
            "num_elements": self.num_elements,
            "avg_num_neighbors": avg_num_neighbors,
            "mmax": mmax,
            "Lmax": Lmax,
            "lmax": lmax,
            "num_channel": num_channel,
            "target_irreps": target_irreps,
            "num_radial_basis": radial_basis["num_radial_basis"],
            "radial_mlp": radial_basis["hidden"],
            "radial_bias": radial_basis["bias"],
            "l1l2": atomic_basis["l1l2"],
            "scatter_norm": atomic_basis["scatter_norm"],
            "correlation": product_basis["correlation"],
            "edge_info_type": atomic_basis["edge_info_type"],
            "resnet_type": resnet["type"],
            "resnet_linear_type": resnet["linear_type"],
            "use_first_resnet": resnet["use_first_resnet"],
            "pre_norm_type": layer_norm["pre_norm_type"],
            "use_first_pre_norm": layer_norm["use_first_pre_norm"],
            "parity": parity,
            "bias": True,
            "node_wise_hidden": atomic_basis["node_wise_hidden"],
            "edge_wise_hidden": atomic_basis["edge_wise_hidden"],
            "stochastic_depth": dropout['stochastic_depth'],

            "num_head": atomic_basis["num_head"],
            "so2_angular_basis": self.so2_angular_basis if self.use_so2 else None,
            "use_so2_edge_ace": atomic_basis["use_so2_edge_ace"],
            "so2_linear_type": atomic_basis["so2_linear_type"],
            "so2_l1l3": atomic_basis["so2_l1l3"],
        }

        self.interactions = torch.nn.ModuleList()
        if len(self.equivariant_property) > 0:
            self.uee_embeddings = torch.nn.ModuleList()
        else:
            self.uee_embeddings = None
        self.products = torch.nn.ModuleList()

        for layer in range(num_layers):
            # === Interaction ===
            self.interactions.append(
                INTERACTION[atomic_basis['type'][layer]](
                    **for_interactions,
                    layer=layer,
                    edge_feats_channel=self.edge_updates[layer].out_dim,
                    nonlinear=atomic_basis['nonlinear'][layer],
                    edge_nonlinear=atomic_basis['edge_nonlinear'][layer],
                    irreps_in=self.node_embedding.irreps_out if layer == 0 else self.products[layer-1].irreps_out,
                    use_graph_softmax=atomic_basis["use_graph_softmax"][layer],  
                )
            )
            inter_irreps_out = self.interactions[layer].irreps_out

            # === UEE ===
            if self.uee_embeddings is not None:
                self.uee_embeddings.append(
                    UniversalEquivariantEmbedding(
                        irreps_in=inter_irreps_out,
                        num_channel=num_channel,
                        num_elements=len(atomic_numbers),
                        config={
                            k: v for k, v in universal_embedding.items() 
                            if k in self.equivariant_property
                        },
                    ),
                )
                prod_irreps_in = self.uee_embeddings[layer].irreps_out
            else:
                prod_irreps_in = inter_irreps_out

            # === Product ===
            self.products.append(
                PRODUCT[product_basis['type'][layer]](
                    layer=layer,
                    num_layers=num_layers,
                    num_elements=self.num_elements,
                    Lmax=Lmax,
                    lmax=lmax,
                    num_channel=num_channel,
                    num_hidden_channel=product_basis['num_channel'],
                    target_irreps=target_irreps,
                    correlation=product_basis['correlation'],
                    l1l2=product_basis['l1l2'],     
                    resolution=product_basis['resolution'],
                    bias=True,
                    stochastic_depth=dropout['stochastic_depth'],
                    parity=parity,
                    irreps_in=prod_irreps_in,
                )
            )
            self.irreps_out = self.products[-1].irreps_out

        if layer_norm['final_norm_type'] is not None:
            self.final_norm = get_normalization_layer(layer_norm['final_norm_type'], ls=target_irreps.ls, num_channels=num_channel)
            self.final_reshape = LayoutTransform([(num_channel, ir) for _, ir in target_irreps])

        if self.use_dens:
            self.irreps_forces_sh = o3.Irreps.spherical_harmonics(lmax=Lmax)
            self.forces_embedding = e3nnLinear(
                self.irreps_forces_sh,
                self.interactions[0].irreps_out,
                bias=True,
            )
            self.decouple_edge_updates = torch.nn.ModuleList()
            self.decouple_interactions = torch.nn.ModuleList()
            self.decouple_products = torch.nn.ModuleList()

            for idx in range(2):
                self.decouple_edge_updates.append(
                    EDGE_UPDATE[edge_update['type']](
                        layer=num_layers-1,
                        num_layers=num_layers,
                        num_elements=self.num_elements,
                        num_radial_basis=self.radial_basis.num_basis,
                        edge_embedding_channel=self.edge_embedding.out_dim,
                        num_channel=num_channel,
                    )
                )
                self.decouple_interactions.append(
                    INTERACTION['cgtp'](
                        **for_interactions,
                        layer=num_layers-1,
                        edge_feats_channel=self.edge_updates[layer].out_dim,
                        nonlinear=atomic_basis['nonlinear'][layer],
                        edge_nonlinear=atomic_basis['edge_nonlinear'][layer],
                        irreps_in=self.products[-1].irreps_out,
                        use_graph_softmax=False,  
                    )
                )
                self.decouple_products.append(
                    PRODUCT[product_basis['type'][layer]](
                        layer=num_layers-1,
                        num_layers=num_layers,
                        num_elements=self.num_elements,
                        Lmax=Lmax,
                        lmax=lmax,
                        num_channel=num_channel,
                        num_hidden_channel=product_basis['num_channel'],
                        target_irreps=target_irreps,
                        correlation=product_basis['correlation'],
                        l1l2=product_basis['l1l2'],     
                        resolution=product_basis['resolution'],
                        bias=True,
                        stochastic_depth=0.0,
                        parity=parity,
                        irreps_in=self.decouple_interactions[idx].irreps_out,
                    )
                )

    def forward(self, data: Dict[str, torch.Tensor], graph) -> Dict[str, torch.Tensor]:
  
        # === edge initialize (radial) ===
        edge_feats, cutoff = self.radial_basis(
            graph.edge_length,
            data['node_attrs'],
            data['edge_index'],
            self.atomic_numbers,
            graph.dcutoff,
        )

        # === angular basis ===
        if self.use_so2:
            self.so2_angular_basis.set_wigner(graph.edge_vector)
            edge_attrs = None # wignerD here
        if self.use_o3:
            edge_attrs = self.o3_angular_basis(graph.edge_vector / graph.edge_length) # have added eps in adapter.py

        # === node initialize ===
        node_feats = self.node_embedding(
            data['node_attrs'],
            edge_feats,
            data['edge_index'],
            edge_attrs,
            cutoff,
        )
        if hasattr(self, "uie_embedding"):
            uie_feats = self.uie_embedding(data)
            node_feats = node_feats + uie_feats
        else:
            uie_feats = None

        edge_feats = self.edge_embedding(
            node_feats,
            data['node_attrs'],
            edge_feats,
            data['edge_index'],
            cutoff,
        )
  
        forces_embedding = None
        noise_mask_tensor = None
        dens_batch_mask_tensor = None
        if self.training and self.use_dens:
            forces_embedding, noise_mask_tensor, dens_batch_mask_tensor = self._forward_dens_forces_encoding(data)
        

        # === representation Learning ===
        prev_feats = []
        for idx, (edge_update, inter, prod) in enumerate(zip(self.edge_updates, self.interactions, self.products)):
            node_attrs_total = data['node_attrs']
            node_attrs_slice = data['node_attrs']
            this_edge_feats = edge_update(
                node_feats,
                node_attrs_total, 
                edge_feats, 
                data['edge_index'],
                cutoff,
            )
            if graph.lmp and idx > 0:
                node_attrs_slice = node_attrs_slice[:graph.lmp_natoms[0]]
            node_feats, sc = inter(
                node_feats,
                node_attrs_total, 
                node_attrs_slice, 
                this_edge_feats, 
                edge_attrs, 
                data['edge_index'],
                cutoff,
                graph,
            )
            if graph.lmp and idx == 0:
                node_attrs_slice = node_attrs_slice[:graph.lmp_natoms[0]] 
            if self.uee_embeddings is not None: 
                node_feats = self.uee_embeddings[idx](node_feats, node_attrs_slice, data)
            if forces_embedding is not None and idx == 0:
                node_feats = node_feats + forces_embedding
            node_feats = prod(node_feats, node_attrs_slice, sc, data["batch"])
            if idx == self.num_layers -1 and hasattr(self, "final_norm"):
                node_feats = self.final_reshape.inverse(self.final_norm(self.final_reshape(node_feats)))
            prev_feats.append(node_feats)

        decouple_node_feats1 = None
        decouple_node_feats2 = None
        if self.use_dens:
       
            decouple_edge_feats1 = self.decouple_edge_updates[0](
                node_feats,
                node_attrs_total, 
                edge_feats, 
                data['edge_index'],
                cutoff,
            )
            decouple_node_feats1, sc = self.decouple_interactions[0](
                node_feats,
                node_attrs_total, 
                node_attrs_slice, 
                decouple_edge_feats1, 
                edge_attrs, 
                data['edge_index'],
                cutoff,
                graph,
            )
            decouple_node_feats1 = self.decouple_products[0](decouple_node_feats1, node_attrs_slice, sc, data["batch"])

            decouple_edge_feats2 = self.decouple_edge_updates[1](
                node_feats,
                node_attrs_total, 
                edge_feats, 
                data['edge_index'],
                cutoff,
            )
            decouple_node_feats2, sc = self.decouple_interactions[1](
                node_feats,
                node_attrs_total, 
                node_attrs_slice, 
                decouple_edge_feats2, 
                edge_attrs, 
                data['edge_index'],
                cutoff,
                graph,
            )
            decouple_node_feats2 = self.decouple_products[1](decouple_node_feats2, node_attrs_slice, sc, data["batch"])

        return {
            "descriptors": prev_feats,
            # "edge_descriptors": edge_descriptors,
            "uie_feats": uie_feats,
            "noise_mask_tensor": noise_mask_tensor,
            "dens_batch_mask_tensor": dens_batch_mask_tensor,
            "decouple_node_feats1": decouple_node_feats1,
            "decouple_node_feats2": decouple_node_feats2,
        }


    def _generate_dens_data(self, data: Dict[str, torch.Tensor]):
        num_atoms = data['node_attrs'].size(0)
        num_graphs = len(data['ptr']) - 1
        # dtype = data['node_attrs'].dtype
        device = data['node_attrs'].device

        if 'direct_forces' in data:
            forces_data = data['direct_forces']
        else:
            forces_data = data['forces']
        if 'noise_mask' in data:
            noise_mask = data['noise_mask'].view(-1, 1)
        else:
            noise_mask = torch.ones((num_atoms, 1), dtype=torch.bool, device=device)
        if 'dens_batch_mask' in data:
            dens_batch_mask = data['dens_batch_mask'].view(-1, 1)
        else:
            dens_batch_mask = torch.ones((num_graphs, 1), dtype=torch.bool, device=device)

        forces_sh = o3.spherical_harmonics(
            l=self.irreps_forces_sh,
            x=forces_data,
            normalize=True,
            normalization='component'
        )
        
        return forces_data, forces_sh, noise_mask, dens_batch_mask
    
    def _forward_dens_forces_encoding(self, data):
        forces_data, forces_sh, noise_mask, dens_batch_mask = self._generate_dens_data(data)
        forces_norm = forces_data.norm(dim=-1, keepdim=True)
        forces_norm = forces_norm / math.sqrt(3.0)
        forces_embedding = forces_sh * forces_norm # [node, 3]
        forces_embedding = self.forces_embedding(forces_embedding)
        forces_embedding = forces_embedding * noise_mask
        return forces_embedding, noise_mask, dens_batch_mask


