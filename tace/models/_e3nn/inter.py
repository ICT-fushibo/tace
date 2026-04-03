################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
"""
Not all residual link are stable, such as BAB, BaB, BB_ba ...
"""
from typing import Optional, Dict


import torch
from tace.utils.torch_scatter import scatter_sum
import e3nn
from e3nn import o3


from ..mlp import ACTIVATION, FFN
from ..layout import LayoutTransform
from .base import Interaction
from .linear import Linear, ElementLinear
from .fused import ScatterTensorProduct
from .nonlinear import GatedLinearUnit, MixtralExpertsGatedLinearUnit, NormLinearUnit, GridMLPUnit
from .layer_norm import get_normalization_layer
from .residual import RESIDUAL, AttentionResidual


class SpectralInteraction(Interaction):
    def _setup(self) -> None:

        self.linear_up = Linear(
            self.irreps_in,
            self.irreps_in,
            bias=self.use_bias,
        )    

        self.rejector = ScatterTensorProduct(
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
                irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in self.irreps_out)
                self.nonlinearity = MixtralExpertsGatedLinearUnit(
                    self.irreps_out,
                    self.irreps_out,
                    act_gates=[ACTIVATION[self.nonlinear_act]()] * len(irreps_gates),
                    bias=self.use_bias,
                    num_experts=self.num_experts,
                    num_shared_experts=self.num_shared_experts,
                    top_k=self.top_k,
                )
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

        if self.scatter_norm == 'density' or self.scatter_norm == 'no_cutoff_density': # this block from MACE
            self.edge_density = FFN[self.edge_info_type](
                [self.edge_feats_channel, 64, 1],
                bias=self.radial_bias,
                layer_norm=self.radial_layer_norm,
                act=self.radial_act,
            )
            self.alpha = torch.nn.Parameter(torch.tensor(self.avg_num_neighbors))
            self.beta = torch.nn.Parameter(torch.tensor(0.0))

        if self.produce_env_feats:
            self.node_envs = FFN[self.edge_info_type](
                [self.edge_feats_channel, self.num_channel, self.num_channel],
                bias=self.radial_bias,
                layer_norm=self.radial_layer_norm,
                act=self.radial_act,
            )

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs_total: torch.Tensor,
        node_attrs_slice: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor],
        graph
    ):
    
        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        density = None

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
            if cutoff is not None and self.apply_density_cutoff:
                density = density * cutoff
            density = scatter_sum(density, edge_index[1], dim=0, dim_size=node_attrs_total.size(0))
            density  = self.truncate_ghosts(density , nlocal)
            density = density * self.beta + self.alpha
            density = density.masked_fill(density == 0, 1e-9)

        node_envs = None
        if hasattr(self, "node_envs"):
            node_envs = self.node_envs(edge_feats) 
            if cutoff is not None and self.apply_density_cutoff:
                node_envs = node_envs* cutoff
            node_envs = scatter_sum(node_envs, edge_index[1], dim=0, dim_size=node_attrs_total.size(0))
            node_envs  = self.truncate_ghosts(node_envs, nlocal)
            if density is not None:
                node_envs = node_envs / density
            else:
                node_envs = node_envs / self.avg_num_neighbors

        if density is not None:
            m_i = m_i / density
        else:
            m_i = m_i / self.avg_num_neighbors


        if hasattr(self, "nonlinearity"):
            if self.nonlinear_type == 'moe':
                m_i = self.nonlinearity(m_i, node_envs)
            else:
                m_i = self.nonlinearity(m_i)
            m_i = self.linear_nonlinearity(m_i)

        return m_i, None
    

class BB(SpectralInteraction):

    def _setup(self):
        super()._setup()

        if (self.use_first_resnet or self.layer > 0):
            self.resnetBB = ElementLinear(
                self.irreps_in,
                self.irreps_sc,
                bias=self.use_bias,
                num_elements=self.num_elements,
            )    

        if (self.use_first_pre_norm or  self.layer > 0) and self.pre_norm_type is not None:
            self.norm1 = get_normalization_layer(
                self.pre_norm_type,
                lmax=self.irreps_in.lmax,
                num_channels=self.num_channel,
            )
            self.reshape1 = LayoutTransform(self.irreps_in)

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
        prev_feats,
    ):
    
        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        resBB = None
        density = None

        if hasattr(self, 'resnetBB'):
            resBB = self.resnetBB(node_feats, node_attrs_slice)
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

        node_envs = None
        if hasattr(self, "node_envs"):
            node_envs = self.node_envs(edge_feats) 
            # if cutoff is not None and self.apply_density_cutoff:
            #     node_envs = node_envs* cutoff
            if cutoff is not None:
                node_envs = node_envs * cutoff
            node_envs = scatter_sum(node_envs, edge_index[1], dim=0, dim_size=node_attrs_total.size(0))
            node_envs  = self.truncate_ghosts(node_envs, nlocal)
            if density is not None:
                node_envs = node_envs / density
            else:
                node_envs = node_envs / self.avg_num_neighbors

        if density is not None:
            m_i = m_i / density
        else:
            m_i = m_i / self.avg_num_neighbors

        if hasattr(self, "nonlinearity"):
            if self.nonlinear_type == 'moe':
                m_i = self.nonlinearity(m_i, node_envs)
            else:
                m_i = self.nonlinearity(m_i)
            m_i = self.linear_nonlinearity(m_i)

        return m_i, self.truncate_ghosts(resBB, nlocal)


class AB(SpectralInteraction):

    def _setup(self):
        super()._setup()

        if isinstance(self.resnet_linear_type, list):
            assert self.resnet_linear_type == 1
        else:
            self.resnet_linear_type = [self.resnet_linear_type] * 1

        if (self.use_first_resnet or self.layer > 0):
            self.resnetAB = RESIDUAL[self.resnet_linear_type[0]](
                layer=self.layer,
                num_layers=self.num_layers,
                irreps_in = self.irreps_out,
                irreps_out = self.irreps_sc,
                bias=self.use_bias,
                num_elements=self.num_elements,
                num_channel=self.num_channel,
            )     

        if (self.use_first_pre_norm or  self.layer > 0) and self.pre_norm_type is not None:
            self.norm1 = get_normalization_layer(
                self.pre_norm_type,
                lmax=self.irreps_out.lmax,
                num_channels=self.num_channel,
            )
            self.reshape1 = LayoutTransform(self.irreps_in)

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
        prev_feats,
    ):
    
        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        resAB = None
        density = None

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

        node_envs = None
        if hasattr(self, "node_envs"):
            node_envs = self.node_envs(edge_feats) 
            # if cutoff is not None and self.apply_density_cutoff:
            #     node_envs = node_envs* cutoff
            if cutoff is not None:
                node_envs = node_envs * cutoff
            node_envs = scatter_sum(node_envs, edge_index[1], dim=0, dim_size=node_attrs_total.size(0))
            node_envs  = self.truncate_ghosts(node_envs, nlocal)
            if density is not None:
                node_envs = node_envs / density
            else:
                node_envs = node_envs / self.avg_num_neighbors

        if density is not None:
            m_i = m_i / density
        else:
            m_i = m_i / self.avg_num_neighbors

        if hasattr(self, "nonlinearity"):
            if self.nonlinear_type == 'moe':
                m_i = self.nonlinearity(m_i, node_envs)
            else:
                m_i = self.nonlinearity(m_i)
            m_i = self.linear_nonlinearity(m_i)
        
        if hasattr(self, "resnetAB"):
            resAB = self.resnetAB(m_i, node_attrs_slice)

        if hasattr(self, 'norm1'):
            m_i = self.reshape1.inverse(self.norm1(self.reshape1(m_i)))

        return m_i, self.truncate_ghosts(resAB, nlocal)
    

class AttnRes(SpectralInteraction):

    def _setup(self):
        super()._setup()

        if isinstance(self.resnet_linear_type, list):
            assert self.resnet_linear_type == 1
        else:
            self.resnet_linear_type = [self.resnet_linear_type] * 1

        if (self.use_first_resnet or self.layer > 0):
            self.resnetBB = AttentionResidual(
                layer=self.layer,
                num_layers=self.num_layers,
                irreps_in = self.irreps_in,
                irreps_out = self.irreps_sc,
                num_elements=self.num_elements,
                num_channel=self.num_channel,
                bias=self.use_bias,
            )    

        if (self.use_first_pre_norm or  self.layer > 0) and self.pre_norm_type is not None:
            self.norm1 = get_normalization_layer(
                self.pre_norm_type,
                lmax=self.irreps_in.lmax,
                num_channels=self.num_channel,
            )
            self.reshape1 = LayoutTransform(self.irreps_in)

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
        prev_feats,
    ):
    
        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        resBB = None
        density = None

        if hasattr(self, "resnetBB"):
            resBB = self.resnetBB(prev_feats, node_attrs_slice)
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

        node_envs = None
        if hasattr(self, "node_envs"):
            node_envs = self.node_envs(edge_feats) 
            # if cutoff is not None and self.apply_density_cutoff:
            #     node_envs = node_envs* cutoff
            if cutoff is not None:
                node_envs = node_envs * cutoff
            node_envs = scatter_sum(node_envs, edge_index[1], dim=0, dim_size=node_attrs_total.size(0))
            node_envs  = self.truncate_ghosts(node_envs, nlocal)
            if density is not None:
                node_envs = node_envs / density
            else:
                node_envs = node_envs / self.avg_num_neighbors

        if density is not None:
            m_i = m_i / density
        else:
            m_i = m_i / self.avg_num_neighbors

        if hasattr(self, "nonlinearity"):
            if self.nonlinear_type == 'moe':
                m_i = self.nonlinearity(m_i, node_envs)
            else:
                m_i = self.nonlinearity(m_i)
            m_i = self.linear_nonlinearity(m_i)

        return m_i, self.truncate_ghosts(resBB, nlocal)
   
     
INTERACTION: Dict[str, Interaction] = {
    "spectral": {
        'BB': BB,
        'AB': AB,
        'AttnRes': AttnRes,
        '0e_AttnRes': AttnRes,
    },
}
INTERACTION['normal'] = INTERACTION['spectral']