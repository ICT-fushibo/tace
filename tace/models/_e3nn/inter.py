################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


from typing import Optional, Dict


import torch
from torch_scatter import scatter_sum
from e3nn import o3

from ..mlp import ACTIVATION
from .base import Interaction
from ..mlp import MLP
from .linear import Linear, ElementLinear
from .fused import ScatterTensorProduct
from .nonlinear import GateNonlinear, NormNonlinear, GridNonlinear


class SpectralInteraction(Interaction):
    def _setup(self) -> None:
            
        if self.layer > 0 and self.resnet in ['BB']:
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
        self.nonlinear_type = None
        if self.nonlinear is not None:
            self.nonlinear_act, self.nonlinear_type = self.nonlinear.split('_')
            if self.nonlinear_type == 'norm':
                self.nonlinearity = NormNonlinear(
                    self.irreps_out,
                    activation=ACTIVATION[self.nonlinear_act](),
                    bias=True,
                )
            elif self.nonlinear_type == 'grid':
                self.nonlinearity = GridNonlinear(
                    self.irreps_out,
                    activation=ACTIVATION[self.nonlinear_act](),
                    bias=False,
                )
            else:
                irreps_gated = self.irreps_out
                irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in self.irreps_out)
                self.nonlinearity = GateNonlinear(
                    irreps_gates=irreps_gates,
                    act_gates=[ACTIVATION[self.nonlinear_act]()] * len(irreps_gates),
                    irreps_gated=irreps_gated,
                )
                linear_down_irreps_out = self.nonlinearity.irreps_in
            if self.has_linear_after_nonlinear:
                self.linear_nonlinearity = Linear(
                    self.irreps_out, 
                    self.irreps_out,  
                    bias=self.use_bias,
                )

        if self.layer > 0 and self.resnet in ['BAB']:
            self.resnetBA = ElementLinear(
                self.irreps_in,
                linear_down_irreps_out,
                bias=self.use_bias,
                num_elements=self.num_elements,
            )      
        
        self.linear_down = Linear(
            self.rejector.irreps_out.simplify(),
            linear_down_irreps_out,
            bias=self.use_bias,
        )    

        if self.layer > 0 and self.resnet in ['AB', 'BAB']:
            self.resnetAB = ElementLinear(
                self.irreps_out,
                self.irreps_sc,
                bias=self.use_bias,
                num_elements=self.num_elements,
            )    

        self.edge_info = MLP(
            [self.edge_feats_channel] + self.radial_mlp + [self.rejector.weight_numel],
            bias=self.radial_bias,
            layer_norm=self.radial_layer_norm,
            act=self.radial_act,
        )

        if self.norm == 'density': # this block from MACE
            self.edge_density = MLP(
                [self.edge_feats_channel, 64, 1],
                bias=self.radial_bias,
                layer_norm=self.radial_layer_norm,
                act=self.radial_act,
            )
            self.alpha = torch.nn.Parameter(torch.tensor(self.avg_num_neighbors))
            self.beta = torch.nn.Parameter(torch.tensor(0.0))

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
            edge_density = torch.tanh(self.edge_density(edge_feats) ** 2)
            if cutoff is not None:
                conv_weights = conv_weights * cutoff
            density = scatter_sum(edge_density, edge_index[1], dim=0, dim_size=node_attrs_total.size(0))
            density  = self.truncate_ghosts(density , nlocal)
            density = density * self.beta + self.alpha
            density = density.masked_fill(density == 0, 1e-9)
            
        if density is not None:
            m_i = m_i / density
        else:
            m_i = m_i / self.avg_num_neighbors

        if resBA is not None:
            m_i = m_i + resBA

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
        sc = self.truncate_ghosts(sc, nlocal)

        return m_i, sc
    


INTERACTION: Dict[str, torch.nn.Module] = {
    "normal": SpectralInteraction,
    "spectral": SpectralInteraction,
}