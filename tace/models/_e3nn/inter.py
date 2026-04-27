################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
"""
Not all residual link are stable, such as BAB, BaB, BB_ba ...
"""
import math
import copy
from typing import Optional, Dict


import torch
from tace.utils.torch_scatter import scatter_sum
import e3nn
from e3nn import o3


from ..mlp import ACTIVATION, FFN
from ..layout import LayoutTransform
from .base import Interaction
from .linear import Linear, ElementLinear
from .fused import O3ScatterTensorProduct, SO2ScatterTensorProduct
from .nonlinear import GatedLinearUnit, NormLinearUnit, GridMLPUnit
from .layer_norm import get_normalization_layer


from tace.utils.torch_scatter import scatter_sum


class CgtpInteraction(Interaction):
    """
    An interaction module based on Clebsch-Gordan tensor products (CGTP).

    This module performs edge-level convolution using Clebsch-Gordan tensor
    products. It supports operator fusion via OpenEquivariance or CuEquivariance,
    which can significantly reduce memory consumption and improve efficiency.
    """

    def _setup(self) -> None:

        self.linear_up = Linear(
            self.irreps_in,
            self.irreps_in,
            bias=self.use_bias,
        )    

        self.rejector = O3ScatterTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            l1l2=self.l1l2,
            ictp_ictc_like=self.ictp_ictc_like,
        )

        linear_down_irreps_out = self.irreps_out
        if self.nonlinear_type is not None:
            if self.nonlinear_type == 'norm':
                self.nonlinearity = NormLinearUnit(
                    linear_down_irreps_out,
                    activation=ACTIVATION[self.nonlinear_act](),
                )
            elif self.nonlinear_type == 'grid':
                self.nonlinearity = GridMLPUnit(
                    linear_down_irreps_out,
                    activation=ACTIVATION[self.nonlinear_act](),
                    bias=False,
                )
            elif self.nonlinear_type == 'e3nngate':
                irreps_scalars = o3.Irreps(
                    [(mul, ir) for mul, ir in self.irreps_out if ir.l == 0]
                )
                irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l > 0])
                irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in irreps_gated)
                activation_fn = torch.nn.functional.silu
                act_gates_fn = torch.nn.functional.sigmoid
                self.nonlinearity = e3nn.nn.Gate(
                    irreps_scalars=irreps_scalars,
                    act_scalars=[activation_fn for _ in irreps_scalars],
                    irreps_gates=irreps_gates,
                    act_gates=[act_gates_fn] * len(irreps_gates),
                    irreps_gated=irreps_gated,
                )
                linear_down_irreps_out = self.nonlinearity.irreps_in.simplify()
            elif self.nonlinear_type == 'gate':
                irreps_gated = linear_down_irreps_out
                irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in linear_down_irreps_out)
                self.nonlinearity = GatedLinearUnit(
                    irreps_gates=irreps_gates,
                    act_gates=[ACTIVATION[self.nonlinear_act]()] * len(irreps_gates),
                    irreps_gated=irreps_gated,
                )
                linear_down_irreps_out = self.nonlinearity.irreps_in
            else:
                assert False, "Unknown Nonlinear"

            self.linear_nonlinearity = Linear(
                self.irreps_out, 
                self.irreps_out,  
                bias=self.use_bias,
            )

        self.linear_down = Linear(
            self.rejector.irreps_out.simplify(),
            linear_down_irreps_out,
            bias=self.use_bias,
        )

        self.edge_info = FFN[self.edge_info_type](
            [self.edge_feats_channel] + self.radial_mlp + [self.rejector.weight_numel],
            bias=self.radial_bias,
            layer_norm=self.radial_layer_norm,
            act=self.radial_act,
        )

        if self.scatter_norm == 'density' or self.scatter_norm == 'no_cutoff_density': 
            self.edge_density = FFN[self.edge_info_type](
                [self.edge_feats_channel, 64, 1],
                bias=self.radial_bias,
                layer_norm=self.radial_layer_norm,
                act=self.radial_act,
            ) # From MACE
            self.alpha = torch.nn.Parameter(torch.tensor(self.avg_num_neighbors))
            self.beta = torch.nn.Parameter(torch.tensor(0.0))


        if (self.use_first_resnet or self.layer > 0) and self.resnet_type == 'BB':
            if self.resnet_linear_type == 'agnostic':
                self.resnetBB = Linear(
                    irreps_in=self.irreps_in,
                    irreps_out=self.irreps_sc,
                    bias=self.use_bias,
                )
            else:
                self.resnetBB = ElementLinear(
                    irreps_in = self.irreps_in,
                    irreps_out = self.irreps_sc,
                    bias=self.use_bias,
                    num_elements=self.num_elements,
                )

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type == 'BAB':
            if self.resnet_linear_type == 'agnostic':
                self.resnetBA = Linear(
                    irreps_in = self.irreps_in,
                    irreps_out = self.irreps_out,
                    bias=self.use_bias,
                )
            else:
                self.resnetBA = ElementLinear(
                    irreps_in = self.irreps_in,
                    irreps_out = self.irreps_out,
                    bias=self.use_bias,
                    num_elements=self.num_elements,
                )

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type in ['AB', 'BAB']:
            if self.resnet_linear_type == 'agnostic':
                self.resnetAB = Linear(
                    irreps_in = self.irreps_out,
                    irreps_out = self.irreps_sc,
                    bias=self.use_bias,
                ) 
            else:
                self.resnetAB = ElementLinear(
                    irreps_in = self.irreps_out,
                    irreps_out = self.irreps_sc,
                    bias=self.use_bias,
                    num_elements=self.num_elements,
                ) 

        if (self.use_first_pre_norm or self.layer > 0) and self.pre_norm_type is not None:
            if self.resnet_type in ['BB', "BAB"]:
                self.norm1 = get_normalization_layer(
                    self.pre_norm_type,
                    ls=self.irreps_in.lmax,
                    num_channels=self.num_channel,
                )
                self.reshape1 = LayoutTransform(self.irreps_in)
            if self.resnet_type in ['AB', "BAB"]:
                self.norm2 = get_normalization_layer(
                    self.pre_norm_type,
                    ls=self.irreps_out.lmax,
                    num_channels=self.num_channel,
                )
                self.reshape2 = LayoutTransform(self.irreps_out)


    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs_total: torch.Tensor,
        node_attrs_slice: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor],
        graph,
        prev_feats: list[torch.Tensor],
    ):
    
        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        density = None
        resBB = None
        resBA = None
        resAB = None

        if hasattr(self, 'resnetBB'):
            if self.resnet_linear_type == 'aware':
                resBB = self.resnetBB(node_feats, node_attrs_slice)
            else:
                resBB = self.resnetBB(node_feats) 

        if hasattr(self, 'resnetBA'):
            if self.resnet_linear_type == 'aware':
                resBA = self.resnetBA(node_feats, node_attrs_slice)
            else:
                resBA = self.resnetBA(node_feats)

        if hasattr(self, 'norm1'):
            node_feats = self.reshape1.inverse(self.norm1(self.reshape1(node_feats)))

        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(node_feats, lmp_data, lmp_natoms, self.layer)
        
        conv_weights = self.edge_info(edge_feats)

        if cutoff is not None:
            conv_weights = conv_weights * cutoff

        m_i = self.linear_down(
            self.truncate_ghosts(
                self.rejector(node_feats, edge_attrs, conv_weights, edge_index), 
                nlocal
            )
        )

        if hasattr(self, "edge_density"):
            density = torch.tanh(self.edge_density(edge_feats) ** 2)
            # if cutoff is not None and self.apply_density_cutoff:
            #     density = density * cutoff
            if cutoff is not None:
                density = density * cutoff
            density = scatter_sum(density, edge_index[1], dim=0, dim_size=node_attrs_total.size(0))
            density  = self.truncate_ghosts(density , nlocal)
            density = density * self.beta + self.alpha
            density = density.masked_fill(density == 0, 1e-9)

        if self.scatter_norm is None:
            m_i = m_i
        elif self.scatter_norm == 'avg_num_neighbors':
            m_i = m_i / self.avg_num_neighbors
        else:
            m_i = m_i / density

        if hasattr(self, "nonlinearity"):
            m_i = self.nonlinearity(m_i)
            m_i = self.linear_nonlinearity(m_i)

        if resBA is not None:
            m_i = m_i + resBA

        if hasattr(self, 'resnetAB'):
            if self.resnet_linear_type == 'aware':
                resAB = self.resnetAB(m_i, node_attrs_slice)
            else:
                resAB = self.resnetAB(m_i)


        if hasattr(self, 'norm2'):
            m_i = self.reshape2.inverse(self.norm2(self.reshape2(m_i)))

        if resBB is not None:
            sc = resBB
        elif resAB is not None:
            sc = resAB
        else:
            sc = None


        return m_i, self.truncate_ghosts(sc, nlocal)
    

class SO2Interaction(Interaction):
    def _setup(self) -> None:

        self.linear_up = Linear(
            self.irreps_in,
            self.irreps_in,
            bias=self.use_bias,
        )    

        self.rejector = SO2ScatterTensorProduct(

            mmax=self.mmax,
            lmax=self.lmax,
            num_channel=self.num_channel,
            is_so2_layout=self.is_so2_layout,
            is_scalar_tp=(self.irreps_node_embedding.lmax == 0) and (self.layer == 0),
            edge_nonlinear=self.edge_nonlinear,

            so2_angular_basis=self.so2_angular_basis,
            reshape_in=LayoutTransform(self.irreps_in),
            reshape_out=LayoutTransform(self.irreps_out),


        )

        linear_down_irreps_out = self.irreps_out
        if self.nonlinear_type is not None:
            if self.nonlinear_type == 'norm':
                self.nonlinearity = NormLinearUnit(
                    linear_down_irreps_out,
                    activation=ACTIVATION[self.nonlinear_act](),
                )
            elif self.nonlinear_type == 'grid':
                self.nonlinearity = GridMLPUnit(
                    linear_down_irreps_out,
                    activation=ACTIVATION[self.nonlinear_act](),
                    bias=False,
                )
            elif self.nonlinear_type == 'e3nngate':
                irreps_scalars = o3.Irreps(
                    [(mul, ir) for mul, ir in self.irreps_out if ir.l == 0]
                )
                irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l > 0])
                irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in irreps_gated)
                activation_fn = torch.nn.functional.silu
                act_gates_fn = torch.nn.functional.sigmoid
                self.nonlinearity = e3nn.nn.Gate(
                    irreps_scalars=irreps_scalars,
                    act_scalars=[activation_fn for _ in irreps_scalars],
                    irreps_gates=irreps_gates,
                    act_gates=[act_gates_fn] * len(irreps_gates),
                    irreps_gated=irreps_gated,
                )
                linear_down_irreps_out = self.nonlinearity.irreps_in.simplify()
            elif self.nonlinear_type == 'gate':
                irreps_gated = linear_down_irreps_out
                irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in linear_down_irreps_out)
                self.nonlinearity = GatedLinearUnit(
                    irreps_gates=irreps_gates,
                    act_gates=[ACTIVATION[self.nonlinear_act]()] * len(irreps_gates),
                    irreps_gated=irreps_gated,
                )
                linear_down_irreps_out = self.nonlinearity.irreps_in
            else:
                assert False, "Unknown Nonlinear"

            self.linear_nonlinearity = Linear(
                self.irreps_out, 
                self.irreps_out,  
                bias=self.use_bias,
            )

        self.linear_down = Linear(
            self.irreps_out,
            linear_down_irreps_out,
            bias=self.use_bias,
        )

        self.edge_info = FFN[self.edge_info_type](
            [self.edge_feats_channel] + self.radial_mlp + [self.rejector.weight_numel],
            bias=self.radial_bias,
            layer_norm=self.radial_layer_norm,
            act=self.radial_act,
        )

        if self.scatter_norm == 'density' or self.scatter_norm == 'no_cutoff_density': 
            self.edge_density = FFN[self.edge_info_type](
                [self.edge_feats_channel, 64, 1],
                bias=self.radial_bias,
                layer_norm=self.radial_layer_norm,
                act=self.radial_act,
            ) # From MACE
            self.alpha = torch.nn.Parameter(torch.tensor(self.avg_num_neighbors))
            self.beta = torch.nn.Parameter(torch.tensor(0.0))


        if (self.use_first_resnet or self.layer > 0) and self.resnet_type == 'BB':
            if self.resnet_linear_type == 'agnostic':
                self.resnetBB = Linear(
                    irreps_in=self.irreps_in,
                    irreps_out=self.irreps_sc,
                    bias=self.use_bias,
                )
            else:
                self.resnetBB = ElementLinear(
                    irreps_in = self.irreps_in,
                    irreps_out = self.irreps_sc,
                    bias=self.use_bias,
                    num_elements=self.num_elements,
                )

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type == 'BAB':
            if self.resnet_linear_type == 'agnostic':
                self.resnetBA = Linear(
                    irreps_in = self.irreps_in,
                    irreps_out = self.irreps_out,
                    bias=self.use_bias,
                )
            else:
                self.resnetBA = ElementLinear(
                    irreps_in = self.irreps_in,
                    irreps_out = self.irreps_out,
                    bias=self.use_bias,
                    num_elements=self.num_elements,
                )

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type in ['AB', 'BAB']:
            if self.resnet_linear_type == 'agnostic':
                self.resnetAB = Linear(
                    irreps_in = self.irreps_out,
                    irreps_out = self.irreps_sc,
                    bias=self.use_bias,
                ) 
            else:
                self.resnetAB = ElementLinear(
                    irreps_in = self.irreps_out,
                    irreps_out = self.irreps_sc,
                    bias=self.use_bias,
                    num_elements=self.num_elements,
                ) 

        if (self.use_first_pre_norm or self.layer > 0) and self.pre_norm_type is not None:
            if self.resnet_type in ['BB', "BAB"]:
                self.norm1 = get_normalization_layer(
                    self.pre_norm_type,
                    ls=self.irreps_in.lmax,
                    num_channels=self.num_channel,
                )
                self.reshape1 = LayoutTransform(self.irreps_in)
            if self.resnet_type in ['AB', "BAB"]:
                self.norm2 = get_normalization_layer(
                    self.pre_norm_type,
                    ls=self.irreps_out.lmax,
                    num_channels=self.num_channel,
                )
                self.reshape2 = LayoutTransform(self.irreps_out)


    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs_total: torch.Tensor,
        node_attrs_slice: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor],
        graph,
        prev_feats: list[torch.Tensor],
    ):
    
        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        density = None
        resBB = None
        resBA = None
        resAB = None

        if hasattr(self, 'resnetBB'):
            if self.resnet_linear_type == 'aware':
                resBB = self.resnetBB(node_feats, node_attrs_slice)
            else:
                resBB = self.resnetBB(node_feats) 

        if hasattr(self, 'resnetBA'):
            if self.resnet_linear_type == 'aware':
                resBA = self.resnetBA(node_feats, node_attrs_slice)
            else:
                resBA = self.resnetBA(node_feats)

        if hasattr(self, 'norm1'):
            node_feats = self.reshape1.inverse(self.norm1(self.reshape1(node_feats)))

        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(node_feats, lmp_data, lmp_natoms, self.layer)
        
        conv_weights = self.edge_info(edge_feats)

        if cutoff is not None:
            conv_weights = conv_weights * cutoff

        m_i = self.linear_down(
            self.truncate_ghosts(
                self.rejector(node_feats, node_attrs_slice, conv_weights, edge_index), 
                nlocal
            ) # check   node_attrs_slice TODO
        )

        if hasattr(self, "edge_density"):
            density = torch.tanh(self.edge_density(edge_feats) ** 2)
            # if cutoff is not None and self.apply_density_cutoff:
            #     density = density * cutoff
            if cutoff is not None:
                density = density * cutoff
            density = scatter_sum(density, edge_index[1], dim=0, dim_size=node_attrs_total.size(0))
            density  = self.truncate_ghosts(density , nlocal)
            density = density * self.beta + self.alpha
            density = density.masked_fill(density == 0, 1e-9)

        if self.scatter_norm is None:
            m_i = m_i
        elif self.scatter_norm == 'avg_num_neighbors':
            m_i = m_i / self.avg_num_neighbors
        else:
            m_i = m_i / density

        if hasattr(self, "nonlinearity"):
            m_i = self.nonlinearity(m_i)
            m_i = self.linear_nonlinearity(m_i)

        if resBA is not None:
            m_i = m_i + resBA

        if hasattr(self, 'resnetAB'):
            if self.resnet_linear_type == 'aware':
                resAB = self.resnetAB(m_i, node_attrs_slice)
            else:
                resAB = self.resnetAB(m_i)


        if hasattr(self, 'norm2'):
            m_i = self.reshape2.inverse(self.norm2(self.reshape2(m_i)))

        if resBB is not None:
            sc = resBB
        elif resAB is not None:
            sc = resAB
        else:
            sc = None


        return m_i, self.truncate_ghosts(sc, nlocal)
    


# class EquivariantGraphAttention(Interaction):
#     def __init__(
#         self,
#         num_in_channels,
#         num_hidden_channels,
#         num_heads,
#         attn_alpha_channels,
#         attn_value_channels,
#         num_out_channels,
#     ):
#         super().__init__()


#         self.edge_info = FFN[self.edge_info_type](
#             [self.edge_feats_channel] + self.radial_mlp + [self.rejector.weight_numel],
#             bias=self.radial_bias,
#             layer_norm=self.radial_layer_norm,
#             act=self.radial_act,
#         )


#         self.num_in_channels = num_in_channels
#         self.num_hidden_channels = num_hidden_channels
#         self.num_heads = num_heads
#         self.attn_alpha_channels = attn_alpha_channels
#         self.attn_value_channels = attn_value_channels
#         self.num_out_channels = num_out_channels
 
        
  



#         self.num_hidden_channels = self.num_hidden_channels * 2

#         extra_m0_out_channels = self.num_heads * self.attn_alpha_channels
#         self.split_m0_channels_list = [extra_m0_out_channels]   # for `torch.split()`
#         temp = self.num_hidden_channels + (self.num_hidden_channels // 2)
#         extra_m0_out_channels = extra_m0_out_channels + temp
#         self.split_m0_channels_list.append(temp)

#         self.so2_linear_1 = SO2Linear(
#             ((2 * self.num_in_channels)),
#             self.num_hidden_channels,
#             self.lmax,
#             self.mmax,
#             extra_m0_out_channels=extra_m0_out_channels     # for attention weights and activation
#         )
        
#         # Graph attention
#         self.alpha_norm = torch.nn.LayerNorm(self.attn_alpha_channels)
#         self.alpha_act = SmoothLeakyReLU()
#         self.alpha_dot = torch.nn.Parameter(torch.randn(self.num_heads, self.attn_alpha_channels))
#         std = 1.0 / math.sqrt(self.attn_alpha_channels)
#         torch.nn.init.uniform_(self.alpha_dot, -std, std)
        
#         # S2/gate activation
#         self.act = SeparableGateS2Activation_SwiGLU_Merge(
#             lmax=self.lmax,
#             mmax=self.mmax,
#             grid_resolution_list=self.resolution,
#             use_m_primary=True
#         )

#         self.so2_linear_2 = SO2Linear(
#             self.num_hidden_channels // 2,
#             self.num_heads * self.attn_value_channels,
#             self.lmax,
#             self.mmax,
#             extra_m0_out_channels=None
#         )

#         # Since we add two type-0 vectors in merge activation, we divide the correpsonding weights by sqrt(2)
#         temp = self.so2_linear_2.num_in_channels
#         self.so2_linear_2.fc_m0.weight.data[0:temp, :].mul_(1.0 / math.sqrt(2.0))
#         self.proj = SO3Linear(self.num_heads * self.attn_value_channels, self.num_out_channels, lmax=self.lmax)

#     def forward(
#         self,
#         node_feats: torch.Tensor,
#         node_attrs_total: torch.Tensor,
#         node_attrs_slice: torch.Tensor,
#         edge_feats: torch.Tensor,
#         edge_attrs: torch.Tensor,
#         edge_index: torch.Tensor,
#         cutoff: Optional[torch.Tensor],
#         graph,
#         prev_feats: list[torch.Tensor],
#     ):
        
#         num_nodes = node_feats.size(0)


#         conv_weights = self.edge_info(edge_feats)


#         rotate 

#         x_message, gate,  x_alpha = self.so2_linear_1(x_message)
#         x_message = self.act(x_message, gate) # nonlinear + ACE
#         x_message = self.so2_linear_2(x_message)

#         # Graph attention
#         x_alpha = x_alpha.view(-1, self.num_heads, self.attn_alpha_channels)
#         x_alpha = self.alpha_norm(x_alpha) # layer norm
#         x_alpha = self.alpha_act(x_alpha)
#         alpha = torch.einsum('bik, ik -> bi', x_alpha, self.alpha_dot) # [head, head_dim] => [head, 1]
#         alpha = self.attn_softmax(alpha, edge_index[1], num_nodes=num_nodes, exp_rescale=cutoff)
#         if cutoff is not None:
#             alpha = alpha * cutoff
#         alpha = alpha.view(alpha.shape[0], 1, self.num_heads, 1)


#         attn = x_message
#         attn = attn.view(attn.shape[0], attn.shape[1], self.num_heads, self.attn_value_channels)
#         attn = attn * alpha
#         attn = attn.view(attn.shape[0], attn.shape[1], self.num_heads * self.attn_value_channels)
#         x_message = attn

#         # Rotate back the irreps
#         x_message = self.so3_rotation.rotate_inv(x_message)
        
#         # Compute the sum of the incoming neighboring messages for each target node
#         x_message = scatter_sum(
#             x_message,
#             edge_index[1], 
#             dim=0,
#             dim_size=num_nodes,
#         )

#         # Project
#         outputs = self.proj(x_message)

#         return outputs


INTERACTION: Dict[str, Interaction] = {
    "normal": CgtpInteraction,
    "spectral": CgtpInteraction,
    "cgtp": CgtpInteraction,
    "so2": SO2Interaction,
}
