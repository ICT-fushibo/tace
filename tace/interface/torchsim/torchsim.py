################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
"""Wrapper for TACE model in TorchSim.

Based on https://github.com/TorchSim/torch-sim/blob/main/torch_sim/models
"""

import os
from typing import Union
from collections.abc import Callable
from pathlib import Path


import torch
try:
    import torch_sim as ts
    from torch_sim.models.interface import ModelInterface
    from torch_sim.neighbors import torchsim_nl
    TORCH_SIM_AVAILABLE = True
except ImportError as e:
    TORCH_SIM_AVAILABLE = False
    print(f"[warning] torch_sim import failed: {e}")


from tace.models.adapter import TensorModel
from tace.lightning import load_tace
from tace.utils._global import DTYPE, DEVICE

  
class TACETorchSimCalc(ModelInterface):
    """Computes energies for multiple systems using a TACE model.

    This class wraps a TACE model to compute energies, forces, and stresses for
    atomic systems within the TorchSim framework. It supports batched calculations
    for multiple systems and handles the necessary transformations between
    TorchSim's data structures and TACE's expected inputs.

    Attributes:
        r_max (float): Cutoff radius for neighbor interactions.
        z_table (utils.AtomicNumberTable): Table mapping atomic numbers to indices.
        model (torch.nn.Module): The underlying TACE neural network model.
        neighbor_list_fn (Callable): Function used to compute neighbor lists.
        atomic_numbers (torch.Tensor): Atomic numbers with shape [n_atoms].
        system_idx (torch.Tensor): System indices with shape [n_atoms].
        n_systems (int): Number of systems in the batch.
        n_atoms_per_system (list[int]): Number of atoms in each system.
        ptr (torch.Tensor): Pointers to the start of each system in the batch with
            shape [n_systems + 1].
        total_atoms (int): Total number of atoms across all systems.
        node_attrs (torch.Tensor): One-hot encoded atomic types with shape
            [n_atoms, n_elements].
    """

    def __init__(
        self,
        model: Union[str, Path, torch.nn.Module, None] = None,
        *,
        device: Union[torch.device, None] = None,
        dtype: torch.dtype = torch.float32,
        neighbor_list_fn: Callable = torchsim_nl,
        compute_forces: bool = True,
        compute_stress: bool = True,
        atomic_numbers: Union[torch.Tensor, None] = None,
        system_idx: Union[torch.Tensor, None] = None,
        fidelity_idx: Union[int, None] = None,
        target_property: Union[list[str], None] = None,
        enable_oeq: bool = False,
        enable_eqt: bool = False,
        enable_cue: bool = False,
    ) -> None:
        """Initialize the TACE model for energy, force, and stress calculations within
        the TorchSim framework. The model can be initialized with atomic numbers
        and system indices, or these can be provided during the forward pass.

        Args:
            model (str | Path | torch.nn.Module | None): The TACE neural network model,
                either as a path to a saved model or as a loaded torch.nn.Module instance.
            device (torch.device | None): The device to run computations on.
                Defaults to CUDA if available, otherwise CPU.
            dtype (torch.dtype): The data type for tensor operations.
                Defaults to torch.float32.
            atomic_numbers (torch.Tensor | None): Atomic numbers with shape [n_atoms].
                If provided at initialization, cannot be provided again during forward.
            system_idx (torch.Tensor | None): System indices with shape [n_atoms]
                indicating which system each atom belongs to. If not provided with
                atomic_numbers, all atoms are assumed to be in the same system.
            neighbor_list_fn (Callable): Function to compute neighbor lists.
                Defaults to torch_nl_linked_cell.
            compute_forces (bool): Whether to compute forces. Defaults to True.
            compute_stress (bool): Whether to compute stress. Defaults to True.
            fidelity_idx : int
                Specify which fidelity fidelity_idx to use. 
            target_property: list(str)
                Extra caculate hessian, atomic_virials, Conservative polarizability, etc,
                If you want to use this parameter, you must provide all the required physical quantities.
            enable_oeq (bool): Whether to enable Oeq acceleration. Defaults to False.
            enable_eqt (bool): Whether to enable Eqt acceleration. Defaults to False.
            enable_cue (bool): Whether to enable CuE acceleration. Defaults to False.
        """
        super().__init__()

        assert TORCH_SIM_AVAILABLE, "Please install package torch-sim-atomistic !"
        if enable_oeq:
            print("Using Oeq for acceleration")
            os.environ['TACE_USE_OEQ'] = '1'
        if enable_cue:
            print("Using CuEq for acceleration")
            os.environ['TACE_USE_CUE'] = '1'
        if enable_eqt:
            print("Using Eqt for acceleration")
            os.environ['TACE_USE_EQT'] = '1'

        self._device = DEVICE[device] or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = DTYPE[dtype]
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self.neighbor_list_fn = neighbor_list_fn
        self._memory_scales_with = "n_atoms_x_density"

        # Load TACE model
        model: TensorModel = load_tace(
            model, 
            'cpu', 
            strict=True, 
            use_ema=True, 
            target_property=target_property
        ) 
        model.flags.compute_forces = self._compute_forces
        model.flags.compute_stress = self._compute_stress
        model.compute_first_derivative = self._compute_forces or self._compute_stress
        model.reset_fidelity_idx(fidelity_idx) 
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.model = model
        self.model = self.model.to(device=self._device)
        if self.dtype is not None:
            self.model = self.model.to(device=self._device)

        # Set model properties
        self.r_max = self.model.readout_fn.cutoff
        atomic_nums = self.model.readout_fn.atomic_numbers
        if not isinstance(atomic_nums, torch.Tensor):
            raise TypeError("TACE model atomic_numbers must be a tensor")
        
        self.torch_element = model.get_torch_element()
        self.model.atomic_numbers = atomic_nums.detach().clone().to(device=self.device)

        self.atomic_numbers_in_init = atomic_numbers is not None
        self.system_idx_in_init = system_idx is not None

        if atomic_numbers is not None:
            self.atomic_numbers = atomic_numbers
            self._setup_node_attrs(atomic_numbers)

        if system_idx is not None:
            self.system_idx = system_idx
            self._setup_ptr(system_idx)

        if (
            atomic_numbers is not None
            and system_idx is not None
            and system_idx.shape[0] != atomic_numbers.shape[0]
        ):
            raise ValueError(
                f"system_idx length {system_idx.shape[0]} must match "
                f"atomic_numbers length {atomic_numbers.shape[0]}."
            )

    def _setup_ptr(self, system_idx: torch.Tensor) -> None:
        """Compute system boundary pointers from system indices.

        Args:
            system_idx (torch.Tensor): System indices tensor with shape [n_atoms].
        """
        counts = torch.bincount(system_idx)
        self.n_systems = len(counts)
        self.n_atoms_per_system = counts.tolist()
        self.ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

    def _setup_node_attrs(self, atomic_numbers: torch.Tensor) -> None:
        """Compute one-hot encoded node attributes from atomic numbers.

        Args:
            atomic_numbers (torch.Tensor): Atomic numbers tensor with shape [n_atoms].
        """
        self.node_attrs = self.torch_element.z2onehot(atomic_numbers).to(self._dtype)

    def forward(
        self, state: ts.SimState, **_kwargs: object
    ) -> dict[str, torch.Tensor]:
        """Compute energies, forces, and stresses for the given atomic systems.

        Processes the provided state information and computes energies, forces, and
        stresses using the underlying TACE model. Handles batched calculations for
        multiple systems and constructs the necessary neighbor lists.

        Args:
            state (SimState): State object containing positions, cell, and other
                system information.
            **_kwargs: Unused; accepted for interface compatibility.

        Returns:
            dict[str, torch.Tensor]: Computed properties:
                - 'energy': System energies with shape [n_systems]
                - 'forces': Atomic forces with shape [n_atoms, 3] if compute_forces=True
                - 'stress': System stresses with shape [n_systems, 3, 3] if
                    compute_stress=True

        Raises:
            ValueError: If atomic numbers are not provided either in the constructor
                or in the forward pass, or if provided in both places.
            ValueError: If system indices are not provided when needed.
        """
        if self.atomic_numbers_in_init:
            if state.positions.shape[0] != self.atomic_numbers.shape[0]:
                raise ValueError(
                    f"Expected {self.atomic_numbers.shape[0]} atoms, "
                    f"got {state.positions.shape[0]}."
                )
        elif not hasattr(self, "atomic_numbers") or not torch.equal(
            state.atomic_numbers, self.atomic_numbers
        ):
            self._setup_node_attrs(state.atomic_numbers)
            self.atomic_numbers = state.atomic_numbers

        if self.system_idx_in_init:
            if state.system_idx.shape[0] != self.system_idx.shape[0]:
                raise ValueError(
                    f"Expected system_idx of length {self.system_idx.shape[0]}, "
                    f"got {state.system_idx.shape[0]}."
                )
        elif not hasattr(self, "system_idx") or not torch.equal(
            state.system_idx, self.system_idx
        ):
            self._setup_ptr(state.system_idx)
            self.system_idx = state.system_idx

        edge_index, mapping_system, unit_shifts = self.neighbor_list_fn(
            state.positions,
            state.row_vector_cell,
            state.pbc,
            self.r_max,
            state.system_idx,
        )
        # shifts = ts.transforms.compute_cell_shifts(
        #     state.row_vector_cell, unit_shifts, mapping_system
        # )

        data_dict = dict(
            ptr=self.ptr,
            node_attrs=self.node_attrs,
            batch=state.system_idx,
            pbc=state.pbc,
            lattice=state.row_vector_cell,
            positions=state.positions,
            edge_index=edge_index,
            edge_shifts=unit_shifts,
            # initial_noncollinear_magmoms=getattr(state, "initial_noncollinear_magmoms", None),
        )

        # Get model output
        out = self.model(data_dict)

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

        # Propagate additional model outputs (e.g. dipole, charges, etc.)
        for key, val in out.items():
            if key not in ("energy", "forces", "stress") and isinstance(
                val, torch.Tensor
            ):
                results[key] = val.detach()

        return results