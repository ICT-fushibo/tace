################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional, Dict


import torch
import torch.nn.functional as F
from torch_scatter import scatter_sum
import equitorch as eqt


from ..env import TACE_USE_OEQ
from ..activation import ACTIVATION
from ..mlp import MLP
from .base import Interaction
from .linear import Linear, ElementLinear
from .paths import generate_eqt_e3nn_paths
from .nonlinear import GateNonlinear, NormNonlinear, GridNonlinear, GLUNonlinear
try:
    from .._oeq import eqtOeqTensorProduct
except Exception:
    pass


class SpectralInteraction(Interaction):
    def _setup(self) -> None:
            
        if self.layer > 0 and self.resnet in ['BB']:
            self.resnetBB = ElementLinear(
                self.irreps_in,
                self.irreps_sc,
                self.num_channel,
                self.num_channel,
                bias=self.use_bias,
                num_elements=self.num_elements,
            )    

        self.linear_up = Linear(
            self.irreps_in,
            self.irreps_in,
            self.num_channel,
            self.num_channel,
            bias=self.use_bias,
        )    

        # === Tensor Porduct ===
        path, out_irreps, e3nn_path, e3nn_out_irreps = generate_eqt_e3nn_paths(
            irreps_out=self.irreps_out,
            irreps_in1=self.irreps_in,
            irreps_in2=self.irreps_sh,
            num_channel=self.num_channel,
            l1l2=self.l1l2,
            ictp_ictc_like=self.ictp_ictc_like,
        )
        self.use_oeq = TACE_USE_OEQ == '1'
        if self.use_oeq:
            self.rejector = eqtOeqTensorProduct(
                irreps_in1=self.irreps_in,
                irreps_in2=self.irreps_sh,
                irreps_out=out_irreps,
                e3nn_irreps_out=e3nn_out_irreps,
                num_channel=self.num_channel,
                instructions=e3nn_path,
            )
        else:
            self.rejector = eqt.nn.SO3Linear(
                irreps_in1=self.irreps_in,
                irreps_in2=self.irreps_sh,
                irreps_out=out_irreps,
                channels_in=self.num_channel,
                channels_out=self.num_channel,
                internal_weights=False,
                feature_mode='uu',
                path_norm=True,
                channel_norm=False,
                path=path,
            )
            
        linear_down_irreps_out = self.irreps_out
        self.nonlinear_type = None
        if self.nonlinear is not None:
            self.nonlinear_act, self.nonlinear_type = self.nonlinear.split('_')
            if self.nonlinear_type == 'norm':
                self.nonlinearity = NormNonlinear(
                    self.irreps_out,
                    num_channel=self.num_channel,
                    activation=ACTIVATION[self.nonlinear_act](),
                    bias=True,
                )
            elif self.nonlinear_type == 'grid':
                self.nonlinearity = GridNonlinear(
                    self.irreps_out,
                    num_channel=self.num_channel,
                    activation=ACTIVATION[self.nonlinear_act](),
                    bias=False,
                )
            elif self.nonlinear_type == 'glu':
                self.nonlinearity = GLUNonlinear(
                    self.irreps_out,
                    self.irreps_out,
                    num_channel=self.num_channel,
                    num_hidden_channel=self.num_channel,
                    activation=ACTIVATION[self.nonlinear_act](),
                    bias=True,
                )
            else:
                self.nonlinearity = GateNonlinear(
                    self.irreps_out,
                    activation=ACTIVATION[self.nonlinear_act](),
                    irrep_wise=True,
                )
                linear_down_irreps_out = (
                    linear_down_irreps_out + 
                    eqt.irreps.Irreps(f'{self.nonlinearity.num_gates}x0e') 
                )

            if self.has_linear_after_nonlinear:
                self.linear_nonlinearity = Linear(
                    self.irreps_out, 
                    self.irreps_out, 
                    self.num_channel, 
                    self.num_channel, 
                    bias=self.use_bias,
                )

        if self.layer > 0 and self.resnet in ['BAB']:
            self.resnetBA = ElementLinear(
                self.irreps_in,
                linear_down_irreps_out,
                self.num_channel,
                self.num_channel,
                bias=self.use_bias,
                num_elements=self.num_elements,
            )    
        
        self.linear_down = Linear(
            self.rejector.irreps_out.simplified(),
            linear_down_irreps_out,
            self.num_channel,
            self.num_channel,
            bias=self.use_bias,
        )    

        if self.layer > 0 and self.resnet in ['AB', 'BAB']:
            self.resnetAB = ElementLinear(
                self.irreps_out,
                self.irreps_sc,
                self.num_channel,
                self.num_channel,
                bias=self.use_bias,
                num_elements=self.num_elements,
            )    

        # self.edge_info = EdgeMoE(
        #     [self.radial_in_channel] + self.radial_mlp + [self.rejector.weight_numel],
        #     bias=True,
        #     layer_norm=self.radial_layer_norm,
        #     act=self.radial_act,
        #     num_experts=self.num_experts,
        #     top_k=self.top_k,
        #     aux_loss_weight=self.aux_loss_weight,
        #     z_loss_weight=self.z_loss_weight,
        # )

        self.edge_info = MLP(
            [self.radial_in_channel] + self.radial_mlp + [self.rejector.weight_numel],
            bias=self.radial_bias,
            layer_norm=self.radial_layer_norm,
            act=self.radial_act,
        )
        
        if self.norm == 'density': 
            # this block from MACE
            self.edge_density = MLP(
                [self.radial_in_channel, 64, 1],
                bias=self.radial_bias,
                layer_norm=self.radial_layer_norm,
                act=self.radial_act,
            )
            self.alpha = torch.nn.Parameter(torch.tensor(self.avg_num_neighbors))
            self.beta = torch.nn.Parameter(torch.tensor(0.0))

        if 'z_ij' in self.conv_weights:
            self.source_embedding = Linear(
                '0e',
                '0e',
                self.num_elements,
                self.num_channel,
                bias=False,
            )
            self.target_embedding = Linear(
                '0e',
                '0e',
                self.num_elements,
                self.num_channel,
                bias=False,
            )
            torch.nn.init.uniform_(self.source_embedding.weight, a=-0.001, b=0.001)
            torch.nn.init.uniform_(self.target_embedding.weight, a=-0.001, b=0.001)


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
        
        # === MHA block ===
        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        resBA = None
        resBB = None
        resAB = None
        density = None
        
        if hasattr(self, "resnetBB"):
            resBB = self.resnetBB(node_feats, node_attrs_slice)
            resBB = self.handle_lammps(resBB, lmp_data, lmp_natoms, self.layer)

        if hasattr(self, "resnetBA"):
            resBA = self.resnetBA(node_feats, node_attrs_slice)
            resBA = self.handle_lammps(resBA, lmp_data, lmp_natoms, self.layer)

        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(node_feats, lmp_data, lmp_natoms, self.layer)
        
        edge_embedding = [edge_feats]
        if hasattr(self, 'source_embedding'):
            edge_embedding.append(
                self.source_embedding(node_attrs_total.unsqueeze(1)[edge_index[0]]).squeeze(1)
            )
        if hasattr(self, 'target_embedding'):
            edge_embedding.append(
                self.target_embedding(node_attrs_total.unsqueeze(1)[edge_index[1]]).squeeze(1)
            )
        full_edge_feats = torch.cat(edge_embedding, dim=-1)


        conv_weights = self.edge_info(full_edge_feats)

        if cutoff is not None:
            conv_weights = conv_weights * cutoff

        if self.use_oeq:
            m_i = self.rejector(node_feats, edge_attrs, edge_index, conv_weights)
        else:
            m_ij = self.rejector(node_feats[edge_index[0]], edge_attrs, conv_weights)
            m_i = scatter_sum(m_ij, edge_index[1], dim=0, dim_size=node_attrs_slice.size(0))

        m_i = self.linear_down(m_i)

        # norm
        if hasattr(self, "edge_density"):
            edge_density = self.edge_density(full_edge_feats)
            edge_density = torch.tanh(edge_density ** 2)
            if cutoff is not None:
                conv_weights = conv_weights * cutoff
            density = scatter_sum(edge_density, edge_index[1], dim=0, dim_size=node_attrs_total.size(0))
            density = density * self.beta + self.alpha
            density = density.masked_fill(density == 0, 1e-9)
        if density is not None:
            m_i = m_i / density.unsqueeze(-1)
        else:
            m_i = m_i / self.avg_num_neighbors

        if resBA is not None:
            m_i = m_i + resBA # layernorm here

        # === FFN / MoE Block === 
        if hasattr(self, "nonlinearity"):
            m_i = self.nonlinearity(m_i)               
            if self.has_linear_after_nonlinear:
                m_i = self.linear_nonlinearity(m_i)
        if hasattr(self, "resnetAB"):                
            resAB = self.resnetAB(m_i, node_attrs_total)

        if resAB is not None:
            sc = resAB
        elif resBB is not None:
            sc = resBB
        else:
            sc = None

        # === LAMMPS postprocessing ===
        if graph.lmp:
            m_i = self.truncate_ghosts(m_i, nlocal)
            sc = self.truncate_ghosts(sc, nlocal)
   
        return m_i, sc
    

INTERACTION: Dict[str, torch.nn.Module] = {
    "normal": SpectralInteraction,
    "spectral": SpectralInteraction,
}