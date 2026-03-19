################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Tuple


import torch
from torch import Tensor
from cartnn.o3 import expand_dims_to
from torch_scatter import scatter_sum


from ..mlp import ACTIVATION, MLP
from .linear import Linear, ElementLinear
from .ctr import Contraction
from .utils import add_dict_to_left
from .nonlinear import GateNonlinear
from .base import Interaction
from .paths import count_irreps, generate_combinations


class SpectralInteraction(Interaction):
    def _setup(self) -> None:

        combs = generate_combinations(
            max(self.irreps_in),
            max(self.irreps_out),
            max(self.irreps_out),
            l1l2=self.l1l2,
        )
        l3_count = count_irreps(combs, False, False, True, True)
        ls = []
        channels_in = []
        channels_out = []
        for l3, count in l3_count.items():
            channels_in.append(self.num_channel * count)
            channels_out.append(self.num_channel)
            ls.append(l3)

        if self.layer > 0 and self.resnet in ['BB']:
            self.resnetBB = ElementLinear(
                self.irreps_sc,
                self.num_channel,
                self.num_channel,
                bias=self.use_bias,
                num_elements=self.num_elements,
            )    

        self.linear_up = Linear(
            self.irreps_in,
            self.num_channel,
            self.num_channel,
            bias=self.use_bias,
        )    

        self.rejector = Contraction(
            lmax_in=max(self.irreps_in),
            lmax_out=max(self.irreps_out),
            num_channel=self.num_channel,
            combs=combs,
        )

        linear_down_irreps_out = self.irreps_out
        self.nonlinear_type = None
        if self.nonlinear is not None:
            self.nonlinear_act, self.nonlinear_type = self.nonlinear.split('_')
            if self.nonlinear_type == 'gate':
                self.nonlinearity = GateNonlinear(
                    ls=self.irreps_out,
                    num_channel=self.num_channel,
                    act=ACTIVATION[self.nonlinear_act](),
                )  
                self.produce_gate = Linear(
                    [0],
                    l3_count[0] * self.num_channel,
                    len(self.irreps_out) * self.num_channel,
                    bias=True,
                )

        if self.layer > 0 and self.resnet in ['BAB']:
            self.resnetBA = ElementLinear(
                linear_down_irreps_out,
                self.num_channel,
                self.num_channel,
                bias=self.use_bias,
                num_elements=self.num_elements,
            )    

        self.linear_down = Linear(
            ls,
            channels_in,
            channels_out,
            bias=True,
        )  

        if self.layer > 0 and self.resnet in ['AB', 'BAB']:
            self.resnetAB = ElementLinear(
                self.irreps_sc,
                self.num_channel,
                self.num_channel,
                bias=self.use_bias,
                num_elements=self.num_elements,
            )    

        self.edge_info = MLP(
            [self.radial_in_channel] + self.radial_mlp + [self.rejector.weight_numel],
            bias=self.radial_bias,
            layer_norm=self.radial_layer_norm,
            act=self.radial_act,
        )

        if self.norm == 'density': # this block from MACE
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
                [0],
                self.num_elements,
                self.num_channel,
                bias=False,
            )
            self.target_embedding = Linear(
                [0],
                self.num_elements,
                self.num_channel,
                bias=False,
            )
            torch.nn.init.uniform_(self.source_embedding.weight, a=-0.001, b=0.001)
            torch.nn.init.uniform_(self.target_embedding.weight, a=-0.001, b=0.001)
    
    def forward(
        self,
        node_feats: Dict[int, Tensor],
        node_attrs_total: Tensor,
        node_attrs_slice: Tensor,
        edge_feats: Tensor,
        edge_attrs: Dict[int, Tensor],
        edge_index: Tensor,
        cutoff: Tensor,
        graph,
    ) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:

        # === LAMMPS pretreatment ===
        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        # === residual === 
        resBA = {}
        resBB = {}
        resAB = {}
        density = None
        if hasattr(self, 'resnetBB'):
            resBB = self.resnetBB(node_feats, node_attrs_slice)
            resBB = self.handle_lammps(resBB, lmp_data, lmp_natoms, self.layer)

        if hasattr(self, "resnetBA"):
            resBA = self.resnetBA(node_feats, node_attrs_slice)
            resBA = self.handle_lammps(resBA, lmp_data, lmp_natoms, self.layer)
            
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(node_feats, lmp_data, lmp_natoms, self.layer)

        # === conv_weights === 
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

        # === ICTP and ICTC ===
        m_i = self.rejector(node_feats, edge_attrs, conv_weights, edge_index) # not include fused operation

        if hasattr(self, "produce_gate"):
            gate = self.produce_gate(m_i)

        m_i = self.linear_down(m_i)

        # === normalizer ===
        if hasattr(self, 'edge_density'):
            edge_density = torch.tanh(self.edge_density(full_edge_feats) ** 2)
            if cutoff is not None:
                conv_weights = conv_weights * cutoff
            density = scatter_sum(edge_density, edge_index[1], dim=0, dim_size=node_attrs_total.size(0))
            density = density * self.beta + self.alpha
            density = density.masked_fill(density == 0, 1e-9)

        if density is not None:
            for r in m_i.keys():
                m_i[r] = m_i[r] / expand_dims_to(density, r+2, dim=-1)
        else:
            for r in m_i.keys():
                m_i[r] = m_i[r] / self.avg_num_neighbors  

        m_i = add_dict_to_left(m_i, resBA)

        if hasattr(self, "nonlinearity"):
            m_i = self.nonlinearity(m_i, torch.split(gate[0], self.num_channel, dim=-1))

        if hasattr(self, "resnetAB"):
            resAB = self.resnetAB(m_i, node_attrs_total)

        if len(resAB) > 0:
            sc = resAB
        elif len(resBB) > 0:
            sc = resBB
        else:
            sc = {}

        # === LAMMPS postprocessing ===
        if graph.lmp:
            m_i = self.truncate_ghosts(m_i, nlocal)
            sc = self.truncate_ghosts(sc, nlocal)

        return m_i, sc

INTERACTION: Dict[str, torch.nn.Module] = {
    "normal": SpectralInteraction,
    "spectral": SpectralInteraction,
}