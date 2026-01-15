################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import List, Dict, Optional


import torch
from torch import Tensor
from e3nn.util.jit import compile_mode

from .tace import TACEV1
from .wrapper_utils import (
    compute_symmetric_displacement, 
    compute_atomic_virials_stresses,
    compute_hessians_vmap,
    sample_force_jacobian,
)
from .utils import Graph
from ...dataset.quantity import PROPERTY, ComputeFlag
from ...utils.torch_scatter import scatter_sum


@compile_mode("script")
class WrapModelV1(torch.nn.Module):

    level: int
    spin_on: int

    def __init__(self, readout_fn: TACEV1):
        super().__init__()
        self.readout_fn = readout_fn
        target_property = readout_fn.target_property
        self.num_levels = readout_fn.num_levels
        self._set_target_property(target_property)
        self._set_lammps_mliap()
        self._set_special
        self._set_embedding_property()
        self._set_universal_embedding()
        self._set_spin_on()
        self._set_computing_level()
   
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, Optional[torch.Tensor | List[torch.Tensor]]]:
        # === pre processing ===
        graph = self.prepare_graph(data)

        # === predict ===
        RESULTS = self.readout_fn(data, graph)
        FIRST = self.first_derivative_fn(data, graph, RESULTS)
        SECOND = self.second_derivative_fn(data, graph, RESULTS, FIRST)

        return {**RESULTS, **FIRST, **SECOND}
    
    def _set_spin_on(self, enable: bool = False):
        self.spin_on = 1 if enable else 0

    def reset_spin_on(self, enable: bool = True):
        self._set_spin_on(enable)

    def get_spin_on(self) -> int:
        return self.spin_on

    def _set_embedding_property(self):
        self.embedding_property = getattr(
            self.readout_fn, "embedding_property", []
        )

    def get_embedding_property(self):
        return self.embedding_property

    def _set_universal_embedding(self):
        self.universal_embedding = getattr(
            self.readout_fn, "universal_embedding", {}
        )

    def _set_special(self):
        self.special = getattr(
            self.readout_fn, "special", {}
        )

    def _set_computing_level(self, level: int = 0):
        self.level = level
        
    def reset_computing_level(self, level: int = 0):
        level = int(level)
        assert level >= 0
        self.level = level

    def get_computing_level(self) -> int:
        return self.level

    def _set_target_property(self, target_property: List[str]):

        assert 'direct_hessians' not in target_property
        assert 'final_collinear_magmoms' not in target_property
        assert 'final_noncollinear_magmoms' not in target_property

        self.target_property = target_property

        self.flags = ComputeFlag()
        for k in self.target_property:
            setattr(self.flags, f"compute_{k}", k in self.target_property)

        self.compute_first_derivative = False
        for p in self.target_property:
            if PROPERTY[p]['first_derivative']:
                self.compute_first_derivative = True

        self.compute_second_derivative = False
        for p in self.target_property:
            if PROPERTY[p]['second_derivative']:
                self.compute_second_derivative = True

        self.retain_graph = self.compute_second_derivative
        self.create_graph = self.compute_second_derivative
        
    def reset_target_property(self, target_property: List[str]):
        assert isinstance(target_property, List)
        self._set_target_property(target_property)
        self.readout_fn._reset_target_property(target_property)

    def get_target_property(self) -> List[str]:
        return list(set(self.target_property))

    def _set_lammps_mliap(self, enable: bool = False):
        self.lmp = enable

    def reset_lammps_mliap(self, enable: bool = True):
        self._set_lammps_mliap(enable)

    def first_derivative_fn(
        self, data: Dict[str, Tensor], graph: Graph, results: Dict[str, Tensor]
    ) -> Dict[str, Optional[Tensor]]:

        E = results["energy"]
        F = None
        V = None
        S = None
        P = None
        M = None
        C_MAG_F = None
        NC_MAG_F = None
        EDGE_F = None
        A_V = None
        A_S = None

        inputs = []
        if self.flags.compute_forces:
            inputs.append(graph.positions)
        if self.flags.compute_stress or self.flags.compute_virials:
            inputs.append(graph.displacement)
        if self.flags.compute_polarization or self.flags.compute_conservative_dipole:
            inputs.append(data["electric_field"])
        if self.flags.compute_magnetization:
            inputs.append(data["magnetic_field"])
        if self.flags.compute_collinear_magnetic_forces:
            inputs.append(data["initial_collinear_magmoms"])
        if self.flags.compute_noncollinear_magnetic_forces:
            inputs.append(data["initial_noncollinear_magmoms"])
        if self.flags.compute_edge_forces or \
            self.flags.compute_atomic_virials or \
            self.flags.compute_atomic_stresses:
            inputs.append(graph.edge_vector)
            
        if self.compute_first_derivative:
            grad_outputs = torch.ones_like(E)
            grads = torch.autograd.grad(
                outputs=E,
                inputs=inputs,
                grad_outputs=grad_outputs,
                retain_graph=self.training or self.retain_graph,
                create_graph=self.training or self.create_graph,
                allow_unused=True,
            )
        idx = 0
        if self.flags.compute_forces:
            F = -grads[idx]
            idx += 1
        if self.flags.compute_stress or self.flags.compute_virials:
            V = -grads[idx]
            VOLUME = torch.linalg.det(data["lattice"]).abs().unsqueeze(-1)
            S = -V / VOLUME.view(-1, 1, 1)
            S = torch.where(torch.abs(S) < 1e10, S, torch.zeros_like(S))
            idx += 1
        if self.flags.compute_polarization or self.flags.compute_conservative_dipole:
            P = -grads[idx]
            idx += 1
        if self.flags.compute_magnetization:
            M = -grads[idx]
            idx += 1
        if self.flags.compute_collinear_magnetic_forces:
            C_MAG_F = -grads[idx]
            idx += 1
        if self.flags.compute_noncollinear_magnetic_forces:
            NC_MAG_F = -grads[idx]
            idx += 1
        if self.flags.compute_edge_forces or \
            self.flags.compute_atomic_virials or \
            self.flags.compute_atomic_stresses:
            EDGE_F = grads[idx] # consistency with LAMMPS
            idx += 1
            if not self.lmp:
                A_V, A_S = compute_atomic_virials_stresses(
                    graph.edge_vector,
                    EDGE_F,
                    data['edge_index'],
                    graph.lattice,
                    data['batch'],
                    data['node_attrs'].size(0),
                    True,
                    True,
                )
                
        return {
            "forces": F,
            "virials": V,
            "stress": S,
            "polarization": P,
            "magnetization": M,
            "collinear_magnetic_forces": C_MAG_F,
            "noncollinear_magnetic_forces": NC_MAG_F,
            "edge_forces": EDGE_F,
            "atomic_virials": A_V,
            "atomic_stresses": A_S,
        }

    def second_derivative_fn(
        self,
        data: Dict[str, torch.Tensor],
        graph: Graph,
        extra: Dict[str, torch.Tensor],
        first_derivative: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        forces = first_derivative["forces"]
        polarization = first_derivative["polarization"]
        magnetization = first_derivative["magnetization"]

        inputs_list: List[Tensor] = []
        if self.flags.compute_born_effective_charges:
            inputs_list.append(graph.positions)
        if self.flags.compute_conservative_polarizability:
            inputs_list.append(data["electric_field"])

        BEC = None
        ALPHA = None
        CHI_M = None
        HESSIANS = None
        jacs_per_graph = None
        samples_per_graph = None

        BECList = []
        ALPHAList = []
        CHI_MList = []
        if self.flags.compute_conservative_polarizability or self.flags.compute_born_effective_charges:
            for i in range(3):  # μ = 0,1,2

                polarization_i = polarization.sum(dim=0)[i]  # sum over batch dimension
                grads = torch.autograd.grad(
                    outputs=polarization_i,
                    inputs=inputs_list,
                    retain_graph=(
                        i != 2 
                        or self.training 
                        or self.flags.compute_hessians 
                        or self.flags.compute_magnetization
                    ),
                    create_graph=self.training,
                )
                idx = 0
                if self.flags.compute_born_effective_charges:
                    BEC = grads[idx]
                    idx += 1
                if self.flags.compute_conservative_polarizability:
                    ALPHA = grads[idx]

                if BEC is None:
                    BEC = torch.zeros(
                        graph.positions.shape[0],
                        3,
                        3,
                        device=graph.positions.device,
                        dtype=graph.positions.dtype,
                    )
                if ALPHA is None:
                    ALPHA = torch.zeros_like(data["electric_field"])

                BECList.append(BEC)  # [atoms, 3]
                ALPHAList.append(ALPHA)  # [B,3]

            BEC = torch.stack(BECList, dim=0)  # [3, atoms, 3]
            BEC = BEC.transpose(1, 0)  #        # [atoms, 3 (pol), 3 (pos)]
            ALPHA = torch.stack(ALPHAList, dim=1)  # [B,3,3]

        if self.flags.compute_magnetization:
            MF = data["magnetic_field"]
            for i in range(3):
                mag_i = magnetization.sum(dim=0)[i]
                CHI_M = torch.autograd.grad(
                    outputs=mag_i,
                    inputs=[MF],
                    retain_graph=(
                        i != 2 
                        or self.training 
                        or self.flags.compute_hessians 
                    ),
                    create_graph=self.training,
                    allow_unused=True,
                )[0]
                if CHI_M is None:
                    CHI_M = torch.zeros_like(MF)
                CHI_MList.append(CHI_M)  # [B,3]
            CHI_M = torch.stack(CHI_MList, dim=1)  # [B,3,3]


        if self.flags.compute_hessians:
            if self.training:
                jacs_per_graph, samples_per_graph = sample_force_jacobian(
                    forces, 
                    graph.positions, 
                    data["ptr"], 
                    num_samples=self.special.get("special", 2),
                    create_graph=self.training,
                )
            else:
                HESSIANS = compute_hessians_vmap(forces, graph.positions)

        return {
            "hessians": HESSIANS,
            "jacs_per_graph": jacs_per_graph,
            "samples_per_graph": samples_per_graph,
            "conservative_polarizability": ALPHA,
            "born_effective_charges": BEC,
            "magnetic_susceptibility": CHI_M,
        }

    def prepare_graph(self, data: Dict[str, torch.Tensor]) -> Graph:

        node_level = (
            data['level'][data['batch']]
            if ("level" in data) and (self.num_levels != 1)
            else torch.full_like(data['batch'], self.level, dtype=torch.int64)
        )  # used for multi-fidelity and multi-head

        if "spin_on" not in data: # used for uie
            data['spin_on'] = torch.tensor([self.spin_on], device=data["node_attrs"].device, dtype=torch.int64)

        if self.lmp:
            for p in self.target_property:
                for requires_grad_p in PROPERTY[p]['requires_grad_with']:
                    if p != 'forces':
                        data[requires_grad_p].requires_grad_(True)
            dtype = data["node_attrs"].dtype
            device =  data["node_attrs"].device 
            nlocal, nghosts = data["natoms"][0], data["natoms"][1]
            num_graphs = 2
            displacement = None
            positions = torch.zeros(
                (int(nlocal), 3),
                dtype=dtype,
                device=device,
            )
            lattice = torch.zeros(
                (num_graphs, 3, 3),
                dtype=dtype,
                device=device,
            )
            edge_vector = data["edge_vector"].requires_grad_(True)
            edge_length = (edge_vector**2).sum(dim=1, keepdim=True).sqrt() + 1e-9
            lmp_data = data["lmp_data"]
            lmp_natoms = (nlocal, nghosts)
            num_atoms_arange = torch.arange(nlocal, device=positions.device, dtype=torch.int64)
        else:
            requires_grad_p_list = []
            for p in self.target_property:
                for requires_grad_p in PROPERTY[p]['requires_grad_with']:
                    requires_grad_p_list.append(requires_grad_p)
                    if requires_grad_p != "edge_vector":
                        data[requires_grad_p].requires_grad_(True)
            dtype = data["node_attrs"].dtype
            device =  data["node_attrs"].device 
            positions = data["positions"]
            num_graphs = data["ptr"].numel() - 1
            displacement = torch.zeros(
                (num_graphs, 3, 3), dtype=dtype, device=device
            )
            if self.flags.compute_virials or self.flags.compute_stress:
                displacement = compute_symmetric_displacement(data, num_graphs)
            source = data["edge_index"][0]
            target = data["edge_index"][1]
            edge_batch = data["batch"][source]
            edge_vector = (
                data["positions"][target]
                - data["positions"][source]
                + torch.einsum(
                    "ni,nij->nj", data["edge_shifts"], data["lattice"][edge_batch]
                )
            )
            if set(self.target_property) & {"edge_vector", "atomic_stresses", "atomic_virials"}:
                    edge_vector.requires_grad_(True)
            edge_length = (edge_vector**2).sum(dim=1, keepdim=True).sqrt() + 1e-9
            lattice = data['lattice']
            lmp_data = None
            lmp_natoms = (positions.size(0), 0)
            num_atoms_arange = torch.arange(positions.shape[0], device=positions.device, dtype=torch.int64)

        return Graph(
            lmp=self.lmp,
            lmp_data=lmp_data,
            lmp_natoms=lmp_natoms, 
            num_graphs=num_graphs,
            displacement=displacement,
            positions=positions,
            edge_vector=edge_vector,
            edge_length=edge_length,
            lattice=lattice,
            node_level=node_level,
            num_atoms_arange=num_atoms_arange,
        )
    


