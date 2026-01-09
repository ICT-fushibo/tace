################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
"""Wrapper for TACE model in TorchSim.

Based on https://github.com/TorchSim/torch-sim/blob/main/torch_sim/models

This module provides a TorchSim wrapper of the TACE model for computing
energies, forces, and stresses for atomistic systems. It integrates the TACE model
with TorchSim's simulation framework, handling batched computations for multiple
systems simultaneously.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Optional

import torch
import torch_sim as ts
from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import vesin_nl_ts
from torch_sim.typing import StateDict
from tace.lightning import load_tace
from tace.dataset.element import TorchElement
from tace.utils._global import DTYPE, DEVICE

class TACETorchSimCalc(ModelInterface):
    def __init__(
        self,
        model: str | Path | torch.nn.Module | None = None,
        *,
        level: Optional[int] = None,
        spin_on: Optional[bool] = None,
        target_property: Optional[list[str]] = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64, 
        neighbor_list_fn: Callable = vesin_nl_ts,
        compute_forces: bool = True,
        compute_stress: bool = True,
        atomic_numbers: torch.Tensor | None = None,
        system_idx: torch.Tensor | None = None,
    ) -> None:
        """
        The model can be initialized with atomic numbers
        and system indices, or these can be provided during the forward pass.

        Args:
            level : int
                Specify which fidelity level to use. 
            spin_on : bool
                If your model uses spin_on uie embedding, you can control whether 
                your calculation enables spin polarization.
            target_property: list(str)
                Extra caculate hessians, atomic_virials, Conservative polarizability, etc,
                If you want to use this parameter, you must provide all the required physical quantities.
            atomic_numbers (torch.Tensor | None): Atomic numbers with shape [n_atoms].
                If provided at initialization, cannot be provided again during forward.
            system_idx (torch.Tensor | None): System indices with shape [n_atoms]
                indicating which system each atom belongs to. If not provided with
                atomic_numbers, all atoms are assumed to be in the same system.
        """
        super().__init__()
        self._device = DEVICE[device] or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self._dtype = DTYPE[dtype]
        self.neighbor_list_fn = neighbor_list_fn
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self._memory_scales_with = "n_atoms_x_density"

        # Load TACE model
        model = load_tace(
            model, 
            self._device, 
            strict=True, 
            use_ema=True, 
            target_property=target_property
        ) 
        if level is not None:
            self.level = level
            model.reset_computing_level(level) 
        else:
            self.level = model.get_computing_level()
        if spin_on is not None:
            self.spin_on = 1 if spin_on else 0
            model.reset_spin_on(self.spin_on) 
        else:
            self.spin_on = model.get_spin_on() 
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.model = model
        if self.dtype is not None:
            self.model = self.model.to(dtype=self.dtype)

        # Set model properties
        self.r_max = self.model.readout_fn.cutoff
        self.torch_element = TorchElement(
            self.model.readout_fn.atomic_numbers.cpu().tolist()
        )

        # Store flag to track if atomic numbers were provided at init
        self.atomic_numbers_in_init = atomic_numbers is not None

        # Set up system_idx information if atomic numbers are provided
        if atomic_numbers is not None:
            if system_idx is None:
                # If system_idx is not provided, assume all atoms belong to same system
                system_idx = torch.zeros(
                    len(atomic_numbers), dtype=torch.long, device=self.device
                )

            self.setup_from_system_idx(atomic_numbers, system_idx)

        if self._compute_forces:
            self.model.flags.compute_forces = True
            self.model.compute_first_derivative = True
        if self._compute_stress:
            self.model.flags.compute_stress = True
            self.model.compute_first_derivative = True

    def setup_from_system_idx(
        self, atomic_numbers: torch.Tensor, system_idx: torch.Tensor
    ) -> None:
        self.atomic_numbers = atomic_numbers
        self.system_idx = system_idx

        # Determine number of systems and atoms per system
        self.n_systems = system_idx.max().item() + 1

        # Create ptr tensor for system boundaries
        self.n_atoms_per_system = []
        ptr = [0]
        for sys_idx in range(self.n_systems):
            system_mask = system_idx == sys_idx
            n_atoms = system_mask.sum().item()
            self.n_atoms_per_system.append(n_atoms)
            ptr.append(ptr[-1] + n_atoms)

        self.ptr = torch.tensor(ptr, dtype=torch.long, device=self.device)
        self.total_atoms = atomic_numbers.shape[0]
        self.node_attrs = self.torch_element.z2onehot(atomic_numbers).to(self._dtype)

    # onehot = element.z2onehot(atomic_numbers).to(dtype=torch.get_default_dtype())

    def forward(self, state: ts.SimState | StateDict) -> dict[str, torch.Tensor]:
        sim_state = (
            state
            if isinstance(state, ts.SimState)
            else ts.SimState(**state, masses=torch.ones_like(state["positions"]))
        )

        # Handle input validation for atomic numbers
        if sim_state.atomic_numbers is None and not self.atomic_numbers_in_init:
            raise ValueError(
                "Atomic numbers must be provided in either the constructor or forward."
            )
        if sim_state.atomic_numbers is not None and self.atomic_numbers_in_init:
            raise ValueError(
                "Atomic numbers cannot be provided in both the constructor and forward."
            )

        # Use system_idx from init if not provided
        if sim_state.system_idx is None:
            if not hasattr(self, "system_idx"):
                raise ValueError(
                    "System indices must be provided if not set during initialization"
                )
            sim_state.system_idx = self.system_idx

        # Update system_idx information if new atomic numbers are provided
        if (
            sim_state.atomic_numbers is not None
            and not self.atomic_numbers_in_init
            and not torch.equal(
                sim_state.atomic_numbers,
                getattr(self, "atomic_numbers", torch.zeros(0, device=self.device)),
            )
        ):
            self.setup_from_system_idx(sim_state.atomic_numbers, sim_state.system_idx)

        # Process each system's neighbor list separately
        edge_indices = []
        edge_shifts_list = []
        offset = 0

        # TODO (AG): Currently doesn't work for batched neighbor lists
        for sys_idx in range(self.n_systems):
            system_mask = sim_state.system_idx == sys_idx
            # Calculate neighbor list for this system
            edge_idx, shifts_idx = self.neighbor_list_fn(
                positions=sim_state.positions[system_mask],
                cell=sim_state.row_vector_cell[sys_idx],
                pbc=sim_state.pbc,
                cutoff=self.r_max,
            )

            # Adjust indices for the system
            edge_idx = edge_idx + offset

            edge_indices.append(edge_idx)
            edge_shifts_list.append(shifts_idx)

            offset += len(sim_state.positions[system_mask])

        # Combine all neighbor lists
        edge_index = torch.cat(edge_indices, dim=1)
        edge_shifts = torch.cat(edge_shifts_list, dim=0)

        # Get model output
        out = self.model(
            dict(
                ptr=self.ptr,
                batch=sim_state.system_idx,
                node_attrs=self.node_attrs,
                # pbc=sim_state.pbc,
                lattice=sim_state.row_vector_cell,
                positions=sim_state.positions,
                edge_index=edge_index,
                edge_shifts=edge_shifts,
            ),
        )

        results: dict[str, torch.Tensor] = {}

        # Process energy
        energy = out["energy"]
        if energy is not None:
            results["energy"] = energy.detach()
        else:
            results["energy"] = torch.zeros(self.n_systems, device=self.device)

        # Process forces
        if self.compute_forces:
            forces = out["forces"]
            if forces is not None:
                results["forces"] = forces.detach()

        # Process stress
        if self.compute_stress:
            stress = out["stress"]
            if stress is not None:
                results["stress"] = stress.detach()

        return results