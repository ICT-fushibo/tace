################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


from typing import Optional, Dict


import torch
from tace.utils.torch_scatter import scatter_sum
import e3nn
from e3nn import o3


from .base import Interaction
from ..mlp import ACTIVATION, FFN
from .linear import Linear, ElementLinear
from .fused import ScatterTensorProduct
from .nonlinear import GatedLinearUnit, MixtralExpertsGatedLinearUnit, NormLinearUnit, GridMLPUnit
from .layer_norm import get_normalization_layer
from ..layout import LayoutTransform

class SpectralInteraction(Interaction):
    def _setup(self) -> None:

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type in ['BB']:
            self.resnetBB = ElementLinear(
                self.irreps_in,
                self.irreps_sc,
                bias=self.use_bias,
                num_elements=self.num_elements,
            )    

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

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type in ['BAB']:
            self.resnetBA = ElementLinear(
                self.irreps_in,
                self.irreps_out,
                bias=self.use_bias,
                num_elements=self.num_elements,
            )      
        
        if (self.use_first_resnet or self.layer > 0) and self.resnet_type in ['AB', 'BAB']:
            self.resnetAB = ElementLinear(
                self.irreps_out,
                self.irreps_sc,
                bias=self.use_bias,
                num_elements=self.num_elements,
            )    

        self.edge_info = FFN[self.edge_info_type](
            [self.edge_feats_channel] + self.radial_mlp + [self.rejector.weight_numel],
            bias=self.radial_bias,
            layer_norm=self.radial_layer_norm,
            act=self.radial_act,
        )

        if self.scatter_norm == 'density': # this block from MACE
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

        if (self.use_first_pre_norm or  self.layer > 0) and self.pre_norm_type is not None:
            if self.resnet_type in ['BB', 'BAB']:
                self.norm1 = get_normalization_layer(
                    self.pre_norm_type,
                    lmax=self.irreps_in.lmax,
                    num_channels=self.num_channel,
                ) # [BB, BAB]
                self.reshape1 = LayoutTransform(self.irreps_in)
            if self.resnet_type in ['AB', 'BAB']:
                self.norm2 = get_normalization_layer(
                    self.pre_norm_type,
                    lmax=self.irreps_out.lmax,
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
        graph
    ):
    
        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        resBA = None
        resBB = None
        resAB = None
        density = None

        if hasattr(self, "resnetBB"):
            resBB = self.resnetBB(node_feats, node_attrs_slice)
        if hasattr(self, "resnetBA"):
            resBA = self.resnetBA(node_feats, node_attrs_slice)

        if hasattr(self, 'norm1'):
            node_feats = self.reshape1.inverse(self.norm1(self.reshape1(node_feats)))

        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(node_feats, lmp_data, lmp_natoms, self.layer)
        
        conv_weights = self.edge_info(edge_feats)

        if cutoff is not None:
            conv_weights = conv_weights * cutoff

        m_i = self.truncate_ghosts(
            self.rejector(node_feats, edge_attrs, conv_weights, edge_index), 
            nlocal
        )

        if hasattr(self, "edge_density"):
            density = torch.tanh(self.edge_density(edge_feats) ** 2)
            if cutoff is not None:
                density = density * cutoff
            density = scatter_sum(density, edge_index[1], dim=0, dim_size=node_attrs_total.size(0))
            density  = self.truncate_ghosts(density , nlocal)
            density = density * self.beta + self.alpha
            density = density.masked_fill(density == 0, 1e-9)

        node_envs = None
        if hasattr(self, "node_envs"):
            node_envs = self.node_envs(edge_feats) 
            if cutoff is not None:
                node_envs = node_envs* cutoff
            node_envs = scatter_sum(node_envs, edge_index[1], dim=0, dim_size=node_attrs_total.size(0))
            node_envs  = self.truncate_ghosts(node_envs, nlocal)
            if density is not None:
                node_envs = node_envs / density
            else:
                node_envs = node_envs / self.avg_num_neighbors

        m_i = self.linear_down(m_i)

        if hasattr(self, "nonlinearity"):
            if self.nonlinear_type == 'moe':
                m_i = self.nonlinearity(m_i, node_envs)
            else:
                m_i = self.nonlinearity(m_i)
            m_i = self.linear_nonlinearity(m_i)

        if density is not None:
            m_i = m_i / density
        else:
            m_i = m_i / self.avg_num_neighbors

        if resBA is not None:
            m_i = m_i + resBA

        if hasattr(self, "resnetAB"):
            resAB = self.resnetAB(m_i, node_attrs_slice)

        if hasattr(self, 'norm2'):
            m_i = self.reshape2.inverse(self.norm2(self.reshape2(m_i)))

        if resAB is not None:
            sc = resAB
        elif resBB is not None:
            sc = resBB
        else:
            sc = None
        sc = self.truncate_ghosts(sc, nlocal)

        return m_i, sc
    

INTERACTION: Dict[str, torch.nn.Module] = {
    "normal": SpectralInteraction,
    "spectral": SpectralInteraction,
}