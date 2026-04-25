################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, List


import torch
from e3nn import o3


from ..radial import RadialBasis
from ..angular import SphericalHarmonics
from ..layout import LayoutTransform
from .node import NODE_EMBEDDING
from .edge import EDGE_EMBEDDING, EDGE_UPDATE
from .inter import INTERACTION
from .prod import PRODUCT
from .ue import UniversalInvariantEmbedding, UniversalEquivariantEmbedding
from .layer_norm import get_normalization_layer


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
        target_weight: List[int],
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
    ):
        super().__init__()

        target_weight = list(set(target_weight))
        self.num_elements = len(atomic_numbers)
        self.num_channel = num_channel
        self.num_layers = num_layers
        self.invariant_property = invariant_property
        self.equivariant_property = equivariant_property
        self.register_buffer('atomic_numbers', torch.tensor(atomic_numbers, dtype=torch.int64))
        self.resnet_type = resnet['type']
        has_so2 = any(t == 'so2' for t in atomic_basis['type'])
        all_so2 = all(t == 'so2' for t in atomic_basis['type'])
        if has_so2 and not all_so2:
            raise ValueError("If any type is 'so2', then all types must be 'so2'")
        self.use_so2 = all_so2

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
        if self.use_so2:
            assert Lmax == lmax, "SO2 Tensor Product need Lmax == lmax in TACE"
            # from .._eqt.equitorch.nn import AlignToZWignerD
            # self.angular_basis = AlignToZWignerD(
            #     irreps="+".join(str(ir) for _, ir in self.interactions[-1].irreps_out),  
            #     normalized=False,
            # )  
            from .._so2.so3 import SO3Rotation
            self.angular_basis = SO3Rotation(lmax, mmax, use_rotation_mask=True)
        else:
            self.angular_basis = SphericalHarmonics(
                o3.Irreps.spherical_harmonics(lmax, p=-1),
                normalize=False,
                normalization="component",
            )

        # === node/edge embedding ===
        # node_embedding['type'] = node_embedding['type'] if not self.use_so2 else 'so2_tensor'
        self.node_embedding = NODE_EMBEDDING[node_embedding['type']](
            num_elements=self.num_elements,
            num_radial_basis=self.radial_basis.num_basis,
            num_channel=num_channel,
            Lmax=Lmax,
            lmax=lmax,
            avg_num_neighbors=avg_num_neighbors,
            bias=False,
            angular_basis=self.angular_basis if self.use_so2 else None
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
                        lmax=lmax,
                    )
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
                    Lmax=Lmax,
                    tensor_dot_channel=edge_update['tensor_dot_channel'],
                )
                for layer in range(num_layers)
            ]
        )


        # === Interaction ===
        self.interactions = torch.nn.ModuleList(
            [
                INTERACTION[atomic_basis['type'][layer]](
                    layer=layer,
                    num_layers=num_layers,
                    num_elements=self.num_elements,
                    avg_num_neighbors=avg_num_neighbors,
                    mmax=mmax,
                    Lmax=Lmax,
                    lmax=lmax,
                    num_channel=num_channel,
                    edge_feats_channel=self.edge_updates[layer].out_dim,
                    target_weight=target_weight,
                    num_radial_basis=radial_basis['num_radial_basis'],
                    radial_mlp=radial_basis['hidden'],
                    radial_bias=radial_basis['bias'],
                    l1l2=atomic_basis['l1l2'],
                    scatter_norm=atomic_basis['scatter_norm'],
                    ictp_ictc_like=atomic_basis['ictp_ictc_like'],
                    nonlinear=atomic_basis['nonlinear'],
                    edge_nonlinear=atomic_basis['edge_nonlinear'],
                    correlation=product_basis['correlation'],
                    edge_info_type=atomic_basis['edge_info_type'],
                    resnet_type=resnet['type'],
                    resnet_linear_type=resnet['linear_type'],
                    use_first_resnet=resnet['use_first_resnet'],
                    resnet_window=resnet['window'],
                    irreps_node_embedding=self.node_embedding.irreps_out,
                    pre_norm_type=layer_norm['pre_norm_type'],
                    use_first_pre_norm=layer_norm['use_first_pre_norm'],
                    angular_basis=self.angular_basis if self.use_so2 else None,
                    is_so2_layout=atomic_basis['is_so2_layout'],
                    resolution=atomic_basis['resolution'],
                    edge_ace_coefs_type=atomic_basis['edge_ace_coefs_type'],
                    so2_hidden_channel=atomic_basis['so2_hidden_channel'],
                    bias=True,
                )
                for layer in range(num_layers)
            ]
        )

        # === Product ===
        self.products = torch.nn.ModuleList(
            [
                PRODUCT[product_basis['type'][layer]](
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
                    l3s=product_basis['l3s'],     
                    ictp_ictc_like=product_basis['ictp_ictc_like'],
                    nonlinear=product_basis['nonlinear'],
                    resolution=atomic_basis['resolution'],
                    bias=True,
                )
                for layer in range(num_layers)
            ]
        )

        if layer_norm['final_norm_type'] is not None: # TODO, support l_list instead lmax
            # self.final_norm = get_normalization_layer(layer_norm['final_norm_type'], lmax=Lmax, num_channels=num_channel)
            # self.final_reshape = LayoutTransform([(num_channel, (l, (-1)**l)) for l in range(Lmax+1)])
            self.final_norm = get_normalization_layer(layer_norm['final_norm_type'], lmax=0, num_channels=num_channel)
            self.final_reshape = LayoutTransform([(num_channel, (l, (-1)**l)) for l in range(0+1)])

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
            self.angular_basis.set_wigner(graph.edge_vector)
            edge_attrs = None
        else:
            edge_attrs = self.angular_basis(graph.edge_vector / graph.edge_length)

        # === node initialize ===
        node_feats = self.node_embedding(
            data['node_attrs'],
            edge_feats,
            data['edge_index'],
            edge_attrs,
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
            node_feats,
            data['node_attrs'],
            edge_feats,
            data['edge_index'],
            cutoff,
        )

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
                prev_feats,
            )
            if graph.lmp and idx == 0:
                node_attrs_slice = node_attrs_slice[:graph.lmp_natoms[0]] 
            if hasattr(self, 'uee_embedding'): 
                node_feats = self.uee_embedding[idx](node_feats, node_attrs_slice, data["batch"], uee_data)
            node_feats = prod(node_feats, node_attrs_slice, sc)
            if idx == self.num_layers -1 and hasattr(self, "final_norm"):
                node_feats = self.final_reshape.inverse(self.final_norm(self.final_reshape(node_feats)))
            prev_feats.append(node_feats)

        return {
            "descriptors": prev_feats,
            "uie_feats": uie_feats,
        }



