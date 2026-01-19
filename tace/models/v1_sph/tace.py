################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import sys 
from typing import Dict, List, Optional, Any


import torch
from e3nn import o3

from tace.models.v1.radial import ZBLBasis
from tace.models.v1.layers import OneHotToAtomicEnergy, ScaleShift

from tace.models.v1.utils import Graph
from .readout import build_scalar_readout
from .representation import SphTACEDescriptor
from .default import (
    RADIAL_BASIS,
    RADIAL_MLP,
    INTER,
    PROD,
    READOUT_EMLP,
    SCALE_SHIFT,
    SHORT_RANGE,
    LONG_RANGE,
    check_model_config
)
from ...dataset.statistics import Statistics
from ...dataset.quantity import get_target_irreps
from ...utils.torch_scatter import scatter_sum

class SphTACEV1(torch.nn.Module):
    def __init__(
        self,
        cutoff: float,
        statistics: List[Statistics],
        max_neighbors: Optional[int] = None,
        lmax: int | List[int] = 3,
        Lmax: int | List[int] = 2,
        group: str = "SO(3)",
        enable_oeq: bool = False,
        bias: bool = False,
        num_layers: int = 2,
        num_levels: int = 1,
        num_channel: int = 64,
        num_channel_hidden: int = 64,
        radial_basis: Dict = RADIAL_BASIS,
        radial_mlp: Dict = RADIAL_MLP,
        inter: Dict = INTER,
        prod: Dict = PROD,
        scale_shift: Dict = SCALE_SHIFT,
        short_range: Dict[str, bool] = SHORT_RANGE,
        long_range: Dict[str, bool] = LONG_RANGE,
        readout_emlp: Dict = READOUT_EMLP,
        target_property: List[str] = [],
        embedding_property: List[str] = [],
        **kwargs,
    ):
        cfg = {
            k: v for k, v in locals().items() 
            if k != "self" 
            and not k.startswith('_')
            and not k.startswith('__')
        }
        cfg = check_model_config(cfg)
        super().__init__()
        # === init ===
        if "model_config" in kwargs:
            self.model_config = kwargs["model_config"]
        self.statistics = cfg["statistics"]
        self.max_neighbors = cfg['max_neighbors']
        self.avg_num_neighbors = cfg['avg_num_neighbors']
        self.num_levels = cfg['num_levels']
        self.target_property = cfg['target_property']
        self.embedding_property = cfg['embedding_property']
        self.target_irreps = (
            get_target_irreps(self.target_property, cart=False) * cfg['num_channel']
        ).regroup()
        if max(cfg['Lmax']) < max(self.target_irreps.ls): # TODO, include parity
                raise ValueError(
                    f"cfg.model.config.Lmax {max(cfg['Lmax'])} should be greatet than"
                    f"the tensor property you want to predict."
                )
        self.register_buffer('num_layers', torch.tensor(cfg['num_layers'], dtype=torch.int64))
        self.register_buffer('atomic_numbers', torch.tensor(cfg['atomic_numbers'], dtype=torch.int64))
        self.register_buffer('cutoff', torch.tensor(cfg['cutoff'], dtype=torch.get_default_dtype()))


        # input, hiiden, output irreps
        ls_in = []      # in of inter
        ls_hidden = []  # out of inter and in of prod
        ls_out = []     # out of prod, out of sc
    
        p = 1 if cfg["group"] == "SO(3)" else -1
        for idx in range(cfg["num_layers"]):
            node_hidden_irreps = (
                o3.Irreps.spherical_harmonics(cfg["Lmax"][idx], p) * cfg["num_channel"]
            ) .regroup()
            sh_irreps = o3.Irreps.spherical_harmonics(cfg["lmax"][idx], p)   
            edge_hidden_irreps = (sh_irreps * cfg["num_channel"]).regroup()
            ls_hidden.append(edge_hidden_irreps)
            ls_in.append(o3.Irreps([(cfg["num_channel"], (0, 1))])) if idx == 0 else ls_in.append(node_hidden_irreps)
            ls_out.append(self.target_irreps) if idx == cfg["num_layers"] - 1 else ls_out.append(node_hidden_irreps)

        for_descriptor = {
            "cutoff": cfg['cutoff'],
            "num_layers": cfg['num_layers'],
            "num_channel": cfg['num_channel'],
            "radial_basis": cfg['radial_basis'],
            "radial_mlp": cfg['radial_mlp'],
            "inter": cfg['inter'],
            "prod": cfg['prod'],
            "atomic_numbers": cfg['atomic_numbers'],
            "avg_num_neighbors": cfg['avg_num_neighbors'],
            "group": cfg["group"],
            "enable_oeq": cfg["enable_oeq"],
            "ls_in": ls_in,
            "ls_hidden": ls_hidden,
            "ls_out": ls_out,
            "sh_irreps": sh_irreps,

        }
        self.descriptor = SphTACEDescriptor(**for_descriptor)

        # === Readout ===
        self.use_all_layer = cfg['readout_emlp'].get('use_all_layer', True)
        for_scalar_readout = {
            'irreps_in': ls_out,
            'irreps_hidden': o3.Irreps("16x0e"),
            'act': cfg['readout_emlp'].get('act', "silu"),
            'bias': cfg['readout_emlp'].get('bias', False),
            'num_levels': cfg.get('num_levels', 1),
            'use_multi_head': cfg.get('use_multi_head', False),
            'num_layers': cfg['num_layers'],
            'use_all_layer': cfg['readout_emlp'].get('use_all_layer', True)
        }

        # === Energy ===
        if "energy" in self.target_property:
            self.energy_readouts = build_scalar_readout(**for_scalar_readout)
            self.atomic_energy_layer = OneHotToAtomicEnergy(cfg['atomic_energies'])
            self.scale_shift = ScaleShift.build_from_config(cfg['statistics'], cfg['scale_shift'])

       # === Short range, Long range ===
        short_range_ = cfg.get('short_range', {})
        long_range_ = cfg.get('long_range', {})
        if short_range_.get('zbl', {}).get('enable', True) and 'energy' in self.target_property:
            self.zbl = ZBLBasis(
                cfg['radial_basis']["polynomial_cutoff"],
                short_range_.get('zbl', {}).get('trainable', False),
            )
        if long_range_.get('les', {}).get('enable', False):
            try:
                from les import Les
            except ImportError as e:
                raise ImportError(
                    "Can not import les. Please install the library from https://github.com/ChengUCB/les."
                ) from e
            les_arguments = long_range_.get('les', {}).get('les_arguments', None)
            if les_arguments is None:
                les_arguments = {"use_atomwise": False}
            self.compute_bec = les_arguments.get("compute_bec", False)
            self.bec_output_index = les_arguments.get("bec_output_index", None)
            self.les = Les(les_arguments=les_arguments)
            self.les_readouts = build_scalar_readout(**for_scalar_readout)

    def readout_fn(
        self,
        data: Dict[str, torch.Tensor],
        graph: Graph,
        from_representation: Dict[str, Optional[torch.Tensor]]
    ) -> Dict[str, Optional[torch.Tensor]]:

        batch = data["batch"]
        descriptors = from_representation['descriptors']


        nlocal, _ = graph.lmp_natoms
        num_graphs = graph.num_graphs
        node_level = graph.node_level
        num_atoms_arange = graph.num_atoms_arange

        # === Energy ===
        E = None
        e_node = None
        if "energy" in self.target_property:
            e_base_node = self.atomic_energy_layer(data['node_attrs'])[num_atoms_arange, node_level]
            e_base_graph = scatter_sum(e_base_node, batch, dim=-1, dim_size=num_graphs)
            e_list = []
            for ii, energy_readout in enumerate(self.energy_readouts):
                if not self.use_all_layer:
                    ii = -1
                e_list.append(energy_readout(descriptors[ii], node_level)[num_atoms_arange, node_level])   
            e_node = torch.sum(torch.stack(e_list, dim=0), dim=0)
            # === ZBL === 
            if hasattr(self, "zbl"):
                e_zbl_node = self.zbl(
                    graph.edge_length, 
                    data['node_attrs'], 
                    data["edge_index"],
                    self.atomic_numbers
                )[num_atoms_arange]
                e_node = e_node + e_zbl_node
            # === scale and shift ===
            e_node = self.scale_shift(
                e_node, 
                data['node_attrs'][num_atoms_arange], 
                data['ptr'], 
                data['edge_index'], 
                data['batch'],
                node_level,
            )    
            # === uie === not support lammps now
            if hasattr(self, "uie_readout"):
                e_uie_node = self.uie_readout(from_representation['uie_feats'])
                e_node = e_node + e_uie_node.squeeze(-1) 
            e_graph = scatter_sum(e_node, batch, dim=-1, dim_size=num_graphs)
            e_node = e_base_node + e_node
            E = e_base_graph + e_graph


        if hasattr(self, 'les'):
            les_lq_list = []
            for ii, les_readout in enumerate(self.les_readouts):
                if not self.use_all_layer:
                    ii = -1
                les_lq_list.append(les_readout(descriptors[ii][0])[num_atoms_arange, node_level])
            LES_LQ = torch.sum(torch.stack(les_lq_list, dim=0), dim=0) 
            les_results = self.les(
                latent_charges=LES_LQ,
                positions=graph.positions,
                cell=graph.lattice.view(-1, 3, 3),
                batch=batch,
                compute_energy=True,
                compute_bec=self.compute_bec,
                bec_output_index=self.bec_output_index,
            )
            LES_E = les_results["E_lr"]
            if LES_E is None:
                LES_E = torch.zeros_like(E)
            E += LES_E
            LES_BEC = les_results["BEC"]
        else:
            LES_E = None
            LES_LQ = None
            LES_BEC = None

        scalar_descriptor = None
        # if o3.Irrep("0e") in self.target_irreps:
        #     scalar_descriptor_list = []
        #     for descriptor in descriptors:
        #         scalar_descriptor_list.append(descriptor[0])
        #     scalar_descriptor = torch.cat(scalar_descriptor_list, dim=-1)

        return {
            "energy": E,
            "node_energy": e_node, # not include les
            "les_energy": LES_E,
            "les_latent_charges": LES_LQ,
            "les_born_effective_charges": LES_BEC,
            "scalar_descriptor": scalar_descriptor,
        }
    
    def forward(self, data: Dict[str, torch.Tensor], graph: Graph) -> Dict[str, Any]:
        outs = self.descriptor(data, graph)
        return self.readout_fn(data, graph, outs)
    
    def _reset_target_property(self, target_property: List[str]):
        self.target_property = target_property

