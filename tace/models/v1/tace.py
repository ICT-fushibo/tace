################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, List, Optional, Any, Type


import torch


from .radial import ZBLBasis
from .layers import OneHotToAtomicEnergy, ScaleShift
from .readout import build_scalar_readout, build_tensor_readout, ScalarReadOut
from .representation import TACEDescriptor
from .utils import Graph, compute_fixed_charge_dipole
from .default import (
    RADIAL_BASIS,
    ANGULAR_BASIS,
    RADIAL_MLP,
    INTER,
    PROD,
    READOUT_EMLP,
    SCALE_SHIFT,
    SHORT_RANGE,
    LONG_RANGE,
    check_model_config
)
from .basis_change import PropertyBasisChange
from ...dataset.statistics import Statistics
from ...dataset.quantity import get_target_irreps
from ...utils.torch_scatter import scatter_sum

class TACEV1(torch.nn.Module):
    def __init__(
        self,
        cutoff: float,
        statistics: List[Statistics],
        max_neighbors: Optional[int] = None,
        lmax: int | List[int] = 3,
        Lmax: int | List[int] = 2,
        bias: bool = False,
        num_layers: int = 2,
        num_levels: int = 1,
        num_channel: int = 64,
        num_channel_hidden: int = 64,
        use_multi_head: bool = False,
        use_multi_fidelity: bool = False,
        radial_basis: Dict = RADIAL_BASIS,
        radial_mlp: Dict = RADIAL_MLP,
        angular_basis: Dict = ANGULAR_BASIS,
        inter: Dict = INTER,
        prod: Dict = PROD,
        scale_shift: Dict = SCALE_SHIFT,
        short_range: Dict[str, bool] = SHORT_RANGE,
        long_range: Dict[str, bool] = LONG_RANGE,
        readout_emlp: Dict = READOUT_EMLP,
        target_property: List[str] = [],
        embedding_property: List[str] = [],
        conservation: Dict[str, Dict[str, bool | str]] = {},
        universal_embedding: Dict[str, Dict[str, str | bool | int | float]] = {},
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
        self.conservation = cfg['conservation']
        self.universal_embedding = cfg['universal_embedding']
        self.special = cfg['special']
        self.target_irreps = get_target_irreps(self.target_property)
        if max(cfg['Lmax']) < max(self.target_irreps):
                raise ValueError(
                    f"cfg.model.config.Lmax {max(cfg['Lmax'])} should be greatet than"
                    f"the tensor property you want to predict."
                )
        self.register_buffer('num_layers', torch.tensor(cfg['num_layers'], dtype=torch.int64))
        self.register_buffer('atomic_numbers', torch.tensor(cfg['atomic_numbers'], dtype=torch.int64))
        self.register_buffer('cutoff', torch.tensor(cfg['cutoff'], dtype=torch.get_default_dtype()))

        for_descriptor = {
            "cutoff": cfg['cutoff'],
            "num_layers": cfg['num_layers'],
            "Lmax": cfg['Lmax'],
            "lmax": cfg['lmax'],
            "num_channel": cfg['num_channel'],
            "num_channel_hidden": cfg['num_channel_hidden'],
            "radial_basis": cfg['radial_basis'],
            "radial_mlp": cfg['radial_mlp'],
            "angular_basis": cfg['angular_basis'],
            "inter": cfg['inter'],
            "prod": cfg['prod'],
            "bias": cfg['bias'],
            "target_irreps": self.target_irreps,
            "universal_embedding": cfg['universal_embedding'],
            "atomic_numbers": cfg['atomic_numbers'],
            "avg_num_neighbors": cfg['avg_num_neighbors'],
        }
        self.descriptor = TACEDescriptor(**for_descriptor)

        # === Readout ===
        self.use_all_layer = cfg['readout_emlp'].get('use_all_layer', True)
        self.use_multi_head = cfg['use_multi_head']
        for_scalar_readout = {
            'in_dim': cfg['num_channel'],
            'hidden_dim': cfg['readout_emlp'].get('hidden_dim', [16]),
            'act': cfg['readout_emlp'].get('act', "silu"),
            'bias': cfg['readout_emlp'].get('bias', False),
            'num_levels': cfg.get('num_levels', 1),
            'use_multi_head': cfg.get('use_multi_head', False),
            'num_layers': cfg['num_layers'],
            'use_all_layer': cfg['readout_emlp'].get('use_all_layer', True)
        }

        for_tensor_readout = {
            'in_dim': cfg['num_channel'],
            'hidden_dim': cfg['readout_emlp'].get('hidden_dim', [16]),
            'gate': cfg['readout_emlp'].get('gate', "silu"),
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

        # === Direct Dipolet === 
        if "direct_dipole" in self.target_property:
            self.dipole_readouts = build_tensor_readout(l=1, **for_tensor_readout)
            
        # === Direct Forces Readout ===
        if "direct_forces" in self.target_property:
            self.direct_forces_readouts = build_tensor_readout(l=1, **for_tensor_readout)
       
        # === Direct Polarizability ===
        if "direct_polarizability" in self.target_property:
            self.direct_polarizability_readout0s = build_scalar_readout(**for_scalar_readout)
            self.direct_polarizability_readout2s = build_tensor_readout(l=2, **for_tensor_readout)
            self.direct_polarizability_basis_change = PropertyBasisChange["direct_polarizability"]() 

        # === Direct Virials ===
        if 'direct_virials' in self.target_property or 'direct_stress' in self.target_property:
            self.direct_virials_readout0s = build_scalar_readout(**for_scalar_readout)
            self.direct_virials_readout2s = build_tensor_readout(l=2, **for_tensor_readout)
            self.direct_virials_basis_change = PropertyBasisChange["direct_virials"]() 

        # === Final Collinear Magmoms ===
        if 'final_collinear_magmoms' in self.target_property:
            self.final_collinear_magmoms_readouts = build_scalar_readout(**for_scalar_readout)

        # === Final Noncollinear Magmoms === not conform to time reversal ...
        if 'final_noncollinear_magmoms' in self.target_property:
            self.final_noncollinear_magmoms_readouts = build_tensor_readout(l=1, **for_tensor_readout)

        # === Charges ===
        if "charges" in self.target_property:
            self.predict_charges_method = self.conservation.get('charges', {}).get('method', 'lagrangian')
            if self.predict_charges_method == 'lagrangian':
                self.chi_readouts = build_scalar_readout(**for_scalar_readout)
                self.eta_readouts = build_scalar_readout(**for_scalar_readout)
            elif self.predict_charges_method == 'uniform_distribution':
                self.charges_readouts = build_scalar_readout(**for_scalar_readout)
            else:
                raise ValueError(
                    f"Unknown predict_charges_method: {self.predict_charges_method}. "
                    "Supported methods are ['lagrangian', 'uniform_distribution']."
                )
            
        # === Direct Hessians ===
        if 'direct_hessians' in self.target_property:
            self.direct_hessians_readout0s = build_scalar_readout(**for_scalar_readout)
            self.direct_hessians_readout2s = build_tensor_readout(l=2, **for_tensor_readout)
            self.direct_hessians_basis_change = PropertyBasisChange["direct_hessians"]() 

        # === Universal Invariant Embedding, not allow multi-head ===
        if cfg['readout_emlp'].get('enable_uie_readout', False) \
        and self.universal_embedding['invariant_embedding_property']:
            self.uie_readout = ScalarReadOut(
                cfg['num_channel'],
                1,
                hidden_dim=[],
                act=None,
                bias=False,
                forward_weight_init=True,
            )

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
        dtype = data['node_attrs'].dtype
        device = data['node_attrs'].device

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
                e_list.append(energy_readout(descriptors[ii][0], node_level)[num_atoms_arange, node_level])   
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

        # === Direct Forces ===
        D_F = None
        if 'direct_forces' in self.target_property:
            d_f_list = []
            for ii, direct_forces_readout in enumerate(self.direct_forces_readouts):
                if not self.use_all_layer:
                    ii = -1
                d_f_list.append(
                    direct_forces_readout(
                        descriptors[ii][1],
                        node_level,
                    )[num_atoms_arange, node_level, :]
                )
            D_F = torch.sum(torch.stack(d_f_list, dim=-1), dim=-1)

        # === Direct Dipole ===
        D = None
        if 'direct_dipole' in self.target_property:
            d_base = compute_fixed_charge_dipole(
                charges=data["charges"],
                positions=data["positions"],
                batch=data["batch"],
                num_graphs=num_graphs,
            )
            d_list = []
            for ii, dipole_readout in enumerate(self.dipole_readouts):
                if not self.use_all_layer:
                    ii = -1
                d_list.append(
                    dipole_readout(
                        descriptors[ii][1],
                        node_level,
                    )[num_atoms_arange, node_level, :]
                )
            d_node = torch.sum(torch.stack(d_list, dim=-1), dim=-1)
            d_graph = scatter_sum(d_node, batch, dim=0, dim_size=num_graphs)
            D = d_base + d_graph

        # === Direct Polarizability ===
        ALPHA = None
        if 'direct_polarizability' in self.target_property:
            alpha0_list = []; alpha2_list = []
            for ii, (polarizability_readout0, polarizability_readout2) in enumerate(
                zip(self.direct_polarizability_readout0s, self.direct_polarizability_readout2s)
            ):
                if not self.use_all_layer:
                    ii = -1
                alpha0_list.append(
                    polarizability_readout0(
                        descriptors[ii][0],
                    )[num_atoms_arange, node_level]
                )
                alpha2_list.append(
                    polarizability_readout2(
                        descriptors[ii][2],
                        node_level,
                    )[num_atoms_arange, node_level, :, :]
                )
            alpha0_node = torch.sum(torch.stack(alpha0_list, dim=-1), dim=-1)
            alpha2_node = torch.sum(torch.stack(alpha2_list, dim=-1), dim=-1)
            alpha0_graph = scatter_sum(alpha0_node, batch, dim=0, dim_size=num_graphs)
            alpha2_graph = scatter_sum(alpha2_node, batch, dim=0, dim_size=num_graphs)
            ALPHA = self.direct_polarizability_basis_change(alpha0_graph, alpha2_graph)

        # === Direct Virials and Stress ===
        D_V = None
        D_S = None
        if 'direct_virials' in self.target_property or 'direct_stress' in self.target_property:
            d_v0_list = []; d_v2_list = []
            for ii, (direct_virials_readout0, direct_virials_readout2) in enumerate(
                zip(self.direct_virials_readout0s, self.direct_virials_readout2s)
            ):
                if not self.use_all_layer:
                    ii = -1
                d_v0_list.append(
                    direct_virials_readout0(
                        descriptors[ii][0],
                    )[num_atoms_arange, node_level]
                )
                d_v2_list.append(
                    direct_virials_readout2(
                        descriptors[ii][2],
                    )[num_atoms_arange, node_level, :, :]
                )
            d_v0_node = torch.sum(torch.stack(d_v0_list, dim=-1), dim=-1)
            d_v2_node = torch.sum(torch.stack(d_v2_list, dim=-1), dim=-1)
            d_v0_graph = scatter_sum(d_v0_node, batch, dim=0, dim_size=num_graphs)
            d_v2_graph = scatter_sum(d_v2_node,batch, dim=0, dim_size=num_graphs)
            D_V = self.direct_virials_basis_change(d_v0_graph, d_v2_graph)
            VOLUME = torch.linalg.det(data["lattice"]).abs().unsqueeze(-1)
            D_S = -D_V / VOLUME.view(-1, 1, 1)
            D_S = torch.where(torch.abs(D_S) < 1e10, D_S, torch.zeros_like(D_S))

         # === Charges === 
        CHARGES = None
        if 'charges' in self.target_property:
            if self.predict_charges_method == 'lagrangian':
                chi_list = []; eta_list = []
                for ii, (chi_readout, eta_readout) in enumerate(
                    zip(self.chi_readouts, self.eta_readouts)
                ):
                    if not self.use_all_layer:
                        ii = -1
                    chi_list.append(chi_readout(descriptors[ii][0])[num_atoms_arange, node_level])
                    eta_list.append(eta_readout(descriptors[ii][0])[num_atoms_arange, node_level])
                chi_node = torch.sum(torch.stack(chi_list, dim=-1), dim=-1)
                eta_node = torch.sum(torch.stack(eta_node, dim=-1), dim=-1)
                eta_node = torch.hypot(eta_node, torch.tensor(1e-6, device=device, dtype=dtype))
                eta_node = torch.reciprocal(eta_node)
                lambda_graph = (
                    data["total_charge"]
                    + scatter_sum(
                        chi_node * eta_node, batch, dim=-1, dim_size=num_graphs
                    )
                ) / scatter_sum(eta_node, batch, dim=-1, dim_size=num_graphs)
                lambda_node = lambda_graph[batch]
                CHARGES = lambda_node * (eta_node) - (chi_node * eta_node)
            elif self.predict_charges_method == 'uniform_distribution':
                c_list = []
                for ii, charges_readout in enumerate(self.charges_readouts):
                    if not self.use_all_layer:
                        ii = -1
                    c_list.append(charges_readout(descriptors[ii][0])[num_atoms_arange, node_level])
                c_node = torch.sum(torch.stack(c_list, dim=-1), dim=-1)
                c_graph = scatter_sum(c_node, batch, dim=-1, dim_size=num_graphs)
                c_delta_node = (c_graph - data["total_charge"]) / (data["ptr"][1:] - data["ptr"][:-1])
                CHARGES = c_node + c_delta_node[batch]

        # === Direct Polarizability ===
        D_HESSIANS = None
        if 'direct_hessians' in self.target_property:
            d_hessians0_list = []; d_hessians2_list = []
            for ii, (direct_hessians_readout0, direct_hessians_readout2) in enumerate(
                zip(self.direct_hessians_readout0s, self.direct_hessians_readout2s)
            ):
                if not self.use_all_layer:
                    ii = -1
                d_hessians0_list.append(
                    direct_hessians_readout0(
                        descriptors[ii][0],
                    )[num_atoms_arange, node_level]
                )
                d_hessians2_list.append(
                    direct_hessians_readout2(
                        descriptors[ii][2],
                        node_level,
                    )[num_atoms_arange, node_level, :, :]
                )
            d_hessians0_node = torch.sum(torch.stack(d_hessians0_list, dim=-1), dim=-1)
            d_hessians2_node = torch.sum(torch.stack(d_hessians2_list, dim=-1), dim=-1)
            d_hessians_node = self.direct_hessians_basis_change(d_hessians0_node, d_hessians2_node)

            d_hessians_list = []
            for idx in range(num_graphs):
                start = data["ptr"][idx]
                end = data["ptr"][idx+1]
                this_d_hessians_node = d_hessians_node[start:end, :, :]
                d_hessians_list.append(
                    (
                        0.5 * (this_d_hessians_node[:, None, :, :] + this_d_hessians_node[None, :, :, :] )
                    ).reshape(-1, 3, 3)
                ) 
            D_HESSIANS = torch.cat(d_hessians_list, dim=0)
            
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
        if 0 in self.target_irreps:
            scalar_descriptor_list = []
            for descriptor in descriptors:
                scalar_descriptor_list.append(descriptor[0])
            scalar_descriptor = torch.cat(scalar_descriptor_list, dim=-1)

        # === Final Noncollinear Magmoms ===
        F_NC_MAG = None
        if 'final_noncollinear_magmoms' in self.target_property:
            f_nc_m_list = []
            for ii, final_noncollinear_magmoms_readout in enumerate(self.final_noncollinear_magmoms_readouts):
                if not self.use_all_layer:
                    ii = -1
                f_nc_m_list.append(
                    final_noncollinear_magmoms_readout(
                        descriptors[ii][1],
                        node_level,
                    )[num_atoms_arange, node_level, :]
                )
            F_NC_MAG = torch.sum(torch.stack(f_nc_m_list, dim=-1), dim=-1)

        # === Final Collinear Magmoms ===
        F_C_MAG = None
        if 'final_collinear_magmoms' in self.target_property:
            f_c_m_list = []
            for ii, final_collinear_magmoms_readout in enumerate(self.final_collinear_magmoms_readouts):
                if not self.use_all_layer:
                    ii = -1
                f_c_m_list.append(final_collinear_magmoms_readout(descriptors[ii][0])[num_atoms_arange, node_level])
            F_C_MAG = torch.abs(torch.sum(torch.stack(f_c_m_list, dim=0), dim=0))

        return {
            "energy": E,
            "node_energy": e_node, # not include les
            "direct_dipole": D,
            "direct_polarizability": ALPHA,
            "direct_forces": D_F,
            "direct_virials": D_V,
            "direct_stress": D_S,
            "direct_hessians": D_HESSIANS,
            "charges": CHARGES,
            "les_energy": LES_E,
            "les_latent_charges": LES_LQ,
            "les_born_effective_charges": LES_BEC,
            "final_collinear_magmoms": F_C_MAG,
            "final_noncollinear_magmoms": F_NC_MAG,
            "scalar_descriptor": scalar_descriptor,
        }
    
    def forward(self, data: Dict[str, torch.Tensor], graph: Graph) -> Dict[str, Any]:
        outs = self.descriptor(data, graph)
        return self.readout_fn(data, graph, outs)
    
    def _reset_target_property(self, target_property: List[str]):
        self.target_property = target_property

