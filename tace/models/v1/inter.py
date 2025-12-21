################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, List, Tuple, Optional, Any

import torch
from torch import Tensor
from cartnn.o3 import ICTD, expand_dims_to
from cartnn.util import scatter_sum

from .act import ACT
from .mlp import MLP
from .paths import generate_combinations
from .linear import SelfInteraction, ElementLinear
from .ctr import Contraction
from .utils import Graph, LAMMPS_MP, dict2flatten, flatten2dict


class Interaction(torch.nn.Module):
    def __init__(
        self,
        atomic_numbers: int,
        num_channel: int,
        num_channel_hidden: int,
        max_r_in: int,
        r_sc: List[int],
        max_r_out: int,
        avg_num_neighbors,
        num_radial_basis,
        radial_mlp={},
        inter: Dict = {},
        bias: bool = False,
        layer: int = -1,
        num_layers: int = -1,
    ) -> None:
        super().__init__()

        # combs
        combs = generate_combinations(
            max_r_in,
            max_r_out,
            max_r_out,
            l1l2=inter["l1l2"][layer],
        )

        # === arguments ===
        enable_residual = inter.get('residual', False)
        enable_layer_norm = radial_mlp.get('enable_layer_norm', False)
        self.add_source_target_embedding = inter.get('add_source_target_embedding', False)
        normalizer = inter.get('normalizer', {})
        self.normalizer_type = normalizer.get('type', 'fixed')
        self.normalizer_scale_shift_trainable = normalizer.get('scale_shift_trainable', False)
        self.register_buffer(
            "avg_num_neighbors",
            torch.tensor(avg_num_neighbors, dtype=torch.get_default_dtype()),
        )
        self.linear_up = SelfInteraction(
            in_channel=num_channel,
            out_channel=num_channel,
            rs=list(range(max_r_in + 1)),
            bias=bias,
        )

        kernel = inter.get('kernel', 'scatter')
        assert kernel in ['scatter', 'torch_fusion']
        self.tc = Contraction(
            combs=combs,
            ictp_lw=inter.get('ictp_lw', False), 
            ictc_lw=inter.get('ictc_lw', False), 
            ictp_hw=inter.get('ictp_hw', True),
            ictc_hw=inter.get('ictc_hw', True),
            num_channel=num_channel,
            num_channel_hidden=num_channel_hidden,
            lmax_in=max_r_in,
            lmax_out=max_r_out,
            kernel=kernel,
        )

        self.enable_residual = enable_residual or layer > 0 or num_layers == 1
        if self.enable_residual:
            self.scs = torch.nn.ModuleDict()
            for r in r_sc:
                self.scs[str(r)] = ElementLinear(
                    num_channel_hidden,
                    num_channel,
                    bias=(r == 0 and bias),
                    atomic_numbers=atomic_numbers,
                    l=r,
                )

        # === ICT ===
        for r in range(max_r_out + 1):
            DS = ICTD(r, r)[1]
            self.register_buffer(f"D_{r}_{r}_1", DS[0].to(torch.get_default_dtype()))
            del DS

        if self.add_source_target_embedding:
            self.source_embedding = MLP(
                len(atomic_numbers),
                num_channel,
                hidden_dim=[],
                act=None,
                bias=False,
                forward_weight_init=True,
                enable_layer_norm=False,
            )
            self.target_embedding = MLP(
                len(atomic_numbers),
                num_channel,
                hidden_dim=[],
                act=None,
                bias=False,
                forward_weight_init=True,
                enable_layer_norm=False,
            )
            torch.nn.init.uniform_(self.source_embedding.mlp[0].weight, a=-0.001, b=0.001)
            torch.nn.init.uniform_(self.target_embedding.mlp[0].weight, a=-0.001, b=0.001)

        if self.normalizer_type == 'dynamic':
            # this normalizer_type is based on mace, for UMLIP
            if self.add_source_target_embedding:
                normalizer_in_dim = num_radial_basis + 2 * num_channel
            else:
                normalizer_in_dim = num_radial_basis
            self.density_normalizer = MLP(
                normalizer_in_dim,
                1,
                normalizer.get('hidden', [64]),
                act=normalizer.get('act_1', 'silu'),
                bias=normalizer.get('bias', False),
                forward_weight_init=True,
                enable_layer_norm=enable_layer_norm,
            )
            self.normalizer_act_2 = ACT[normalizer.get('act_2', 'tanh')]()
            if self.normalizer_scale_shift_trainable:
                self.alpha = torch.nn.Parameter(torch.tensor(20.0), requires_grad=True)
                self.beta = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.r_sc = r_sc
        self.max_r_in = max_r_in
        self.max_r_out = max_r_out
        self.layer = layer
        self.num_layers = num_layers
        self.num_channel = num_channel


        # ==== conv weight ==== TODO for xzm move this to interaction
        if inter.get('add_source_target_embedding', False):
            radial_in_dim = num_radial_basis + 2 * num_channel
        else:
            radial_in_dim = num_radial_basis
        self.radial_net = MLP(
            radial_in_dim,
            num_channel * len(combs),
            radial_mlp["hidden"][layer],
            act=radial_mlp["act"],
            bias=radial_mlp.get('bias', False),
            forward_weight_init=radial_mlp.get("forward_weight_init", True),
            enable_layer_norm=radial_mlp.get('enable_layer_norm', False),
        )

    def D(self, l: int):
        return dict(self.named_buffers())[f"D_{l}_{l}_1"]
    
    def forward(
        self,
        node_feats: Dict[int, Tensor],
        node_attrs: Tensor,
        node_attrs_lmp: Tensor,
        edge_feats: Tensor,
        edge_attrs: Dict[int, Tensor],
        edge_index: Tensor,
        cutoff: Tensor,
        graph: Graph,
    ) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:

        lmp = graph.lmp
        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lmp_data=lmp_data,
            lmp_natoms=lmp_natoms,
            layer=self.layer,
        )

        if self.add_source_target_embedding:
            source_embedding = self.source_embedding(node_attrs)
            target_embedding = self.target_embedding(node_attrs) # TODO BUG check which node_attrs in lmp
            edge_feats = torch.cat(
                [
                    edge_feats,
                    source_embedding[edge_index[0]],
                    target_embedding[edge_index[1]],
                ],
                dim=-1,
            )
        if hasattr(self, 'density_normalizer'):
            edge_density = self.normalizer_act_2(self.density_normalizer(edge_feats) ** 2)
            if cutoff is not None:
                edge_density = edge_density * cutoff
            density = scatter_sum(
                src=edge_density, index=edge_index[1], dim=0, dim_size=node_attrs_lmp.shape[0]
            )
            # if lmp:
            #     density = self.truncate_ghosts(density, nlocal) 
            if self.normalizer_scale_shift_trainable:
                density = density * self.beta + self.alpha
            else:
                density = density + 1
            density = density.masked_fill(density == 0, 1e-9)


        conv_weights = self.radial_net(edge_feats) # for compatiable
        if cutoff is not None:
            conv_weights = conv_weights * cutoff

        tmp_m_i = self.tc(node_feats, edge_attrs, conv_weights, edge_index)
        m_i = {}


        for r in tmp_m_i.keys():
            T = tmp_m_i[r]
            if self.normalizer_type == 'dynamic':
                normalizer = expand_dims_to(density, T.ndim, dim=-1)
            else:
                normalizer = self.avg_num_neighbors
            T = T / normalizer
            B = T.size(0)
            C = T.size(1)
            REST = (3,) * r
            m_i[r] = (
                T.reshape(B, C, -1) @ self.D(r)
            ).reshape((B, C) + REST)
            
        residual = {}
        if self.enable_residual:
            for nu, sc in self.scs.items():
                nu = int(nu)
                residual[nu] = sc(m_i[nu], node_attrs_lmp)

        if lmp:
            node_attrs_lmp = self.truncate_ghosts(node_attrs_lmp, nlocal)
            max_r = max(m_i.keys())
            m_i = dict2flatten(max_r, m_i)
            m_i = self.truncate_ghosts(m_i, nlocal)
            m_i = flatten2dict(max_r, m_i, self.num_channel)

            if len(residual) > 0:
                max_r = max(residual.keys())                
                residual = dict2flatten(max_r, residual)
                residual = self.truncate_ghosts(residual, nlocal)
                residual = flatten2dict(max_r, residual, self.num_channel)
        return m_i, residual


    def handle_lammps(
        self,
        node_feats: Dict[int, Tensor],
        lmp_data: Optional[Any],
        lmp_natoms: Tuple[int, int],
        layer: int,
    ) -> Tensor:  
        _, nghosts = lmp_natoms
        first_layer = (layer == 0)
        if lmp_data is None or first_layer or torch.jit.is_scripting():
            return node_feats
        max_r = max(node_feats.keys())
        node_feats = dict2flatten(max_r, node_feats)
        pad = torch.zeros(
            (nghosts, node_feats.shape[1]),
            dtype=node_feats.dtype,
            device=node_feats.device,
        )
        node_feats = torch.cat((node_feats, pad), dim=0)
        node_feats = LAMMPS_MP.apply(node_feats, lmp_data)
        return flatten2dict(max_r, node_feats, self.num_channel)
    
    def truncate_ghosts(
        self, t: Tensor, nlocal: Optional[int] = None
    ) -> Tensor:
        return t[:nlocal] if nlocal is not None else t

