################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
"""
So far, all the functions have been concentrated within one interaction module,
and LAMMPS utils are modified from MACE
"""
import abc
from typing import Dict, List, Tuple, Optional, Any


import torch
from torch import Tensor


from cartnn import ICTD, expand_dims_to
from .mlp import MLP
from .linear import SelfInteraction, LinearDict
from .ctr import Contraction
from .utils import Graph, LAMMPS_MP, dict2flatten, flatten2dict, add_dict_to_left
from .layers import NormNonlinearity, GatedNonlinearity
from ...utils.torch_scatter import scatter_sum


class InteractionBase(torch.nn.Module):
    def __init__(
        self,
        layer: int,
        num_layers: int,
        num_channel: int,
        num_channel_hidden: int,
        bias: bool,
        lmax_in: int,
        lmax_out: int,
        ls_sc: List[int],
        atomic_numbers: int,
        avg_num_neighbors: float,
        num_radial_basis: int,
        radial_mlp: Dict,
        inter: Dict,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.num_layers = num_layers
        self.num_channel = num_channel
        self.num_channel_hidden = num_channel_hidden
        self.bias = bias
        self.lmax_in = lmax_in
        self.lmax_out = lmax_out
        self.ls_sc = ls_sc
        self.atomic_numbers = atomic_numbers
        self.num_radial_basis = num_radial_basis
        self.radial_mlp = radial_mlp
        self.inter = inter
        self.register_buffer(
            "avg_num_neighbors",
            torch.tensor(avg_num_neighbors, dtype=torch.get_default_dtype()),
        )
        for l in range(lmax_out + 1):
            DS = ICTD(l, l)[1]
            self.register_buffer(f"D_{l}_{l}_1", DS[0].to(torch.get_default_dtype()))
            del DS
        self._setup()

    def D(self, l: int):
        return dict(self.named_buffers())[f"D_{l}_{l}_1"]
    
    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

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
        self, t: torch.Tensor, nlocal: Optional[int] = None
    ) -> torch.Tensor:
        return t[:nlocal] if nlocal is not None else t
    
    def truncate_ghosts_dict(
        self, t: Dict[int, torch.Tensor], nlocal: Optional[int] = None
    ) -> Dict[int, torch.Tensor]:
        lmax = max(t.keys())
        t = dict2flatten(lmax, t)
        t = self.truncate_ghosts(t, nlocal)
        t = flatten2dict(lmax, t, self.num_channel)
        return t
    
class Interaction(InteractionBase):
    def _setup(self) -> None:
        # === resnet === 
        if self.inter.get('use_resnet', False):
            self.resnet = SelfInteraction(
                self.num_channel,
                self.num_channel,
                ls=list(range(self.lmax_in + 1)),
                bias=self.bias,
                atomic_numbers=self.atomic_numbers,
            )    

        # === linear up ===  
        self.linear_up = SelfInteraction(
            self.num_channel,
            self.num_channel,
            ls=list(range(self.lmax_in + 1)),
            bias=self.bias,
        )

        # === ICTP and ICTC
        self.tc = Contraction(
            num_channel=self.num_channel,
            num_channel_hidden=self.num_channel_hidden,
            lmax_in=self.lmax_in,
            lmax_out=self.lmax_out,
            l1l2=self.inter["l1l2"][self.layer],
            filter_combs=self.inter.get("filter_combs", None)
        )

        # === self / skip connection ===
        self.sc_from = None
        if self.inter.get('sc', {}).get('use_first_sc', False) \
        or self.layer > 0 \
        or self.num_layers == 1:
            self.scs = torch.nn.ModuleDict()
            for l in self.ls_sc:
                    self.scs[str(l)] = LinearDict[True, False](
                    self.num_channel_hidden,
                    self.num_channel,
                    bias=(l == 0 and self.bias),
                    atomic_numbers=self.atomic_numbers,
                    l=l,
                )

        # ==== conv weights ====
        conv_weights_type = self.inter.get('conv_weights', ['edge_ij'])
        assert 'edge_ij' in conv_weights_type, 'edge_ij must be in conv_weights' 
        radial_in_dim = self.num_radial_basis + self.num_channel * (len(conv_weights_type)-1)
        need_layer_norm = radial_in_dim != self.num_radial_basis
        self.radial_net = MLP(
            radial_in_dim,
            self.tc.weight_numel,
            self.radial_mlp["hidden"][self.layer],
            act=self.radial_mlp["act"],
            bias=self.radial_mlp.get('bias', False),
            forward_weight_init=True,
            enable_layer_norm=need_layer_norm,
        )
        if 'node_j' in conv_weights_type:
            self.source_embedding = MLP(
                len(self.atomic_numbers),
                self.num_channel,
                hidden_dim=[],
                act=None,
                bias=False,
                forward_weight_init=True,
                enable_layer_norm=False,
            )
            torch.nn.init.uniform_(self.source_embedding.mlp[0].weight, a=-0.001, b=0.001)
        if 'node_i' in conv_weights_type:
            self.target_embedding = MLP(
                len(self.atomic_numbers),
                self.num_channel,
                hidden_dim=[],
                act=None,
                bias=False,
                forward_weight_init=True,
                enable_layer_norm=False,
            )
            torch.nn.init.uniform_(self.target_embedding.mlp[0].weight, a=-0.001, b=0.001)

        # ==== normalizer ====
        self.normalizer = self.inter.get('normalizer', 'avg_num_neighbors')
        assert self.normalizer in ["avg_num_neighbors", "density_v1"]
        if self.normalizer == 'density_v1': # density_v1 are from mace
            self.density_normalizer = MLP(
                radial_in_dim,
                1,
                [64],
                act='silu',
                bias=True,
                forward_weight_init=True,
                enable_layer_norm=need_layer_norm,
            )
            self.alpha = torch.nn.Parameter(torch.tensor(20.0), requires_grad=True)
            self.beta = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # === nonlinearity ===
        nonlinearity_type = self.inter.get('nonlinearity', {}).get('type', None)
        nonlinearity_gate = self.inter.get('nonlinearity', {}).get('gate', 'silu')
        if nonlinearity_type is not None:
            if nonlinearity_type in ['gated', 'gated_gate']:
                self.nonlinearity_gate = GatedNonlinearity(
                    rmax=self.lmax_out,
                    in_dim=self.num_channel_hidden,
                    gate=nonlinearity_gate,
                )  
            elif nonlinearity_type in ['norm', 'norm_gate']:
                self.nonlinearity_gate = NormNonlinearity(
                    rmax=self.lmax_out,
                    in_dim=self.num_channel_hidden,
                    gate=nonlinearity_gate,
                )
            else:
                raise
            self.nonlinearity_linear = SelfInteraction(
                in_channel=self.num_channel_hidden,
                out_channel=self.num_channel_hidden,
                ls=list(range(self.lmax_out + 1)),
                bias=self.bias,
            )
    
    def forward(
        self,
        node_feats: Dict[int, Tensor],
        node_attrs_total: Tensor,
        node_attrs_slice: Tensor,
        edge_feats: Tensor,
        edge_attrs: Dict[int, Tensor],
        edge_index: Tensor,
        cutoff: Tensor,
        graph: Graph,
    ) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:

        # === LAMMPS pretreatment ===
        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        # === residual === 
        residual = {}
        if hasattr(self, 'resnet'):
            residual = self.resnet(node_feats, node_attrs_slice)
            residual = self.handle_lammps(
                residual,
                lmp_data=lmp_data,
                lmp_natoms=lmp_natoms,
                layer=self.layer,
            )

        # === linear up === 
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps( 
            node_feats,
            lmp_data=lmp_data,
            lmp_natoms=lmp_natoms,
            layer=self.layer,
        )

        # === conv_weights === 
        edge_embedding: list[torch.Tensor] = [edge_feats]
        if hasattr(self, 'source_embedding'):
            edge_embedding.append(self.source_embedding(node_attrs_total)[edge_index[0]])
        if hasattr(self, 'target_embedding'):
            edge_embedding.append(self.target_embedding(node_attrs_total)[edge_index[1]])
        full_edge_feats = torch.cat(edge_embedding, dim=-1)
        conv_weights = self.radial_net(full_edge_feats)
        if cutoff is not None:
            conv_weights = conv_weights * cutoff

        # === normalizer ===
        if hasattr(self, 'density_normalizer'):
            edge_density = torch.tanh(self.density_normalizer(full_edge_feats) ** 2)
            if cutoff is not None:
                edge_density = edge_density * cutoff
            density = scatter_sum(edge_density, edge_index[1], dim=0, dim_size=node_attrs_total.size(0)) 
            density = density * self.beta + self.alpha
            density = density.masked_fill(density == 0, 1e-9)

        # === ICTP and ICTC ===
        r_m_i: Dict[int, torch.Tensor] = self.tc(node_feats, edge_attrs, conv_weights, edge_index)
        m_i = {}
        for r, t in r_m_i.items():
            B = t.size(0)
            C = t.size(1)
            REST = (3,) * r
            if self.normalizer == 'density_v1':
                normalizer = expand_dims_to(density, r+2, dim=-1)
            else:
                normalizer = self.avg_num_neighbors
            t = t / normalizer
            m_i[r] = (t.reshape(B, C, -1) @ self.D(r)).reshape((B, C) + REST)
        add_dict_to_left(m_i, residual)

        # === nonlinearity ===
        if hasattr(self, 'nonlinearity_gate'):
            m_i = self.nonlinearity_linear(self.nonlinearity_gate(m_i))

        scs = {}
        # === self connection ===
        if hasattr(self, 'scs'):
            for r in self.ls_sc:
                scs[r] = self.scs[str(r)](m_i[r], node_attrs_total)

        # === LAMMPS postprocessing ===
        if graph.lmp:
            m_i = self.truncate_ghosts_dict(m_i, nlocal)
            if len(scs) > 0:
                scs = self.truncate_ghosts_dict(scs, nlocal)
 
        return m_i, scs



