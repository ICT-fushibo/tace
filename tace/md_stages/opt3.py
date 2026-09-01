"""TACE Opt3: one whole-step CUDA Graph for GPU-resident NVT MD.

Each production replay contains the FP64 Berendsen or Nose-Hoover-chain
integrator, fixed-shape PBC neighbour construction, native TACE forward and
conservative-force autograd, and persistent state updates.  Capacity overflow
is recorded on device and fails closed at the next synchronization point.

Transaction rollback, graph buckets, compilation, model-specific fusion and
eager fallback are intentionally outside this stage.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from md_benchmark.md_route import (
    MDRunRequest,
    MDRunResult,
    configure_torch_baseline,
    validate_result,
)
from torch import Tensor

from tace.md_route import _set_exact_acceleration_environment, _validate_model_contract
from tace.md_stages.fixed_neighbor import (
    FixedShapeTACENeighborBuilder,
    maximum_neighbors_in_graph,
    neighbor_capacity_from_probe,
)
from tace.md_stages.opt1 import (
    BerendsenIntegrator,
    GPUMDState,
    NoseHooverChainIntegrator,
    _build_integrator,
    _distribution_version,
    _record_observation,
    _validate_final_state,
)
from tace.md_stages.opt2 import (
    _DEFAULT_GRAPH_ENERGY_ATOL,
    _DEFAULT_GRAPH_FORCE_ATOL,
    _NeutralPaddingRadialBasis,
    _assert_close,
    _enable_padding_density_masks_,
    _positive_float,
    _positive_int,
)


_BACKEND = "whole-step-cuda-graph"
_DEFAULT_CAPTURE_WARMUP = 3
_DEFAULT_NEIGHBOR_MARGIN = 0.10
_DEFAULT_NEIGHBOR_STEP = 8
_DEFAULT_SINK_COUNT = 32


def _slots_from_total_edge_capacity(
    total_capacity: int,
    num_atoms: int,
    *,
    slot_step: int,
    guard_slots: int = 1,
) -> int:
    """Convert an Opt2 total-edge CAP into guarded per-centre slots."""

    if total_capacity < 1 or num_atoms < 1:
        raise ValueError("total_capacity and num_atoms must be positive")
    if slot_step < 1 or guard_slots < 0:
        raise ValueError("slot_step must be positive and guard_slots non-negative")
    per_atom = math.ceil(total_capacity / num_atoms)
    aligned = math.ceil(per_atom / slot_step) * slot_step
    return aligned + guard_slots * slot_step


def _integrate_nhc_pure(
    momenta: Tensor,
    eta: Tensor,
    p_eta: Tensor,
    integrator: NoseHooverChainIntegrator,
    delta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply one NHC half-step without mutating persistent thermostat state."""

    def update_one(
        current_momenta: Tensor,
        values: list[Tensor],
        index: int,
        delta2: float,
        delta4: float,
    ) -> None:
        if index < len(values) - 1:
            values[index] = values[index] * torch.exp(
                -delta4 * values[index + 1] / integrator.Q[index + 1]
            )
        if index == 0:
            g_j = (
                (current_momenta.square() / integrator.masses).sum()
                - 3.0 * integrator.num_atoms * integrator.kT
            )
        else:
            g_j = (
                values[index - 1].square() / integrator.Q[index - 1]
                - integrator.kT
            )
        values[index] = values[index] + delta2 * g_j
        if index < len(values) - 1:
            values[index] = values[index] * torch.exp(
                -delta4 * values[index + 1] / integrator.Q[index + 1]
            )

    output_momenta = momenta
    output_eta = eta
    output_p_eta = p_eta
    for _ in range(integrator.chain_loops):
        for coefficient in (
            1.3512071919596578,
            -1.7024143839193153,
            1.3512071919596578,
        ):
            sub_delta = coefficient * delta / integrator.chain_loops
            delta2, delta4 = sub_delta / 2.0, sub_delta / 4.0
            values = [output_p_eta[index] for index in range(integrator.chain_length)]
            for index in range(integrator.chain_length - 1, -1, -1):
                update_one(output_momenta, values, index, delta2, delta4)
            output_p_eta = torch.stack(values)
            output_eta = output_eta + sub_delta * output_p_eta / integrator.Q
            output_momenta = output_momenta * torch.exp(
                -sub_delta * output_p_eta[0] / integrator.Q[0]
            )
            values = [output_p_eta[index] for index in range(integrator.chain_length)]
            for index in range(integrator.chain_length):
                update_one(output_momenta, values, index, delta2, delta4)
            output_p_eta = torch.stack(values)
    return output_momenta, output_eta, output_p_eta


class TACEWholeStepPotential:
    """Fixed-address native TACE input and capture-safe topology builder."""

    def __init__(
        self,
        atoms,
        model_path: str,
        *,
        device: torch.device,
        options: dict[str, Any],
    ) -> None:
        try:
            import torch_sim as ts
            from torch_sim.neighbors import torchsim_nl
        except ImportError as exc:
            raise ImportError(
                "TACE Opt3 requires torch-sim-atomistic>=0.6.1"
            ) from exc
        from tace.interface.torchsim import TACETorchSimCalc

        self.device = device
        self.num_atoms = len(atoms)
        atomic_numbers = torch.as_tensor(
            atoms.get_atomic_numbers(), dtype=torch.int64, device=device
        )
        system_idx = torch.zeros(self.num_atoms, dtype=torch.int64, device=device)
        calculator = TACETorchSimCalc(
            model_path,
            device=device,
            dtype=None,
            neighbor_list_fn=torchsim_nl,
            compute_forces=True,
            compute_stress=False,
            atomic_numbers=atomic_numbers,
            system_idx=system_idx,
            enable_oeq=False,
            enable_cue=False,
            enable_eqt=False,
            enable_compile=False,
            enable_triton=False,
        )
        self.model_metadata = _validate_model_contract(
            calculator, requested_accelerators=set()
        )
        self.model = calculator.model
        self.model_dtype = self.model.get_model_dtype()
        self.neighbor_list_fn = torchsim_nl
        self.system_idx = calculator.system_idx
        sim_state = ts.io.atoms_to_state(
            [atoms], device=device, dtype=self.model_dtype
        )
        self.cell = sim_state.row_vector_cell.reshape(3, 3).contiguous()
        self.pbc = sim_state.pbc.reshape(3).contiguous()
        initial_positions = sim_state.positions.detach().clone()
        initial_positions.requires_grad_(True)
        self.static_positions = initial_positions
        self.static_data: dict[str, Tensor] = {
            "ptr": calculator.ptr,
            "node_attrs": calculator.node_attrs,
            "batch": calculator.system_idx,
            "pbc": sim_state.pbc,
            "lattice": sim_state.row_vector_cell,
            "positions": self.static_positions,
        }

        initial_edge_index, _, initial_edge_shifts = self.neighbor_list_fn(
            self.static_positions,
            sim_state.row_vector_cell,
            sim_state.pbc,
            self.model_metadata["cutoff_a"],
            self.system_idx,
        )
        if initial_edge_index.shape[1] < 1:
            raise RuntimeError("TACE Opt3 initial neighbour graph contains no edges")
        self.initial_edge_count = int(initial_edge_index.shape[1])
        initial_maximum = maximum_neighbors_in_graph(
            initial_edge_index, self.num_atoms
        )
        margin = _nonnegative_float(
            options, "graph_neighbor_margin", _DEFAULT_NEIGHBOR_MARGIN
        )
        slot_step = _positive_int(
            options, "graph_neighbor_step", _DEFAULT_NEIGHBOR_STEP
        )
        inferred_slots = neighbor_capacity_from_probe(
            max(1, initial_maximum), margin=margin, slot_step=slot_step
        )
        explicit_slots = options.get("neighbors_per_atom")
        if explicit_slots is not None:
            if (
                isinstance(explicit_slots, bool)
                or not isinstance(explicit_slots, int)
                or explicit_slots < 1
            ):
                raise ValueError("TACE Opt3 neighbors_per_atom must be positive")
            slots = explicit_slots
        else:
            total_capacity = options.get("edge_capacity")
            if total_capacity is not None:
                if (
                    isinstance(total_capacity, bool)
                    or not isinstance(total_capacity, int)
                    or total_capacity < 1
                ):
                    raise ValueError("TACE Opt3 edge_capacity must be positive")
                total_floor = _slots_from_total_edge_capacity(
                    total_capacity,
                    self.num_atoms,
                    slot_step=slot_step,
                    guard_slots=1,
                )
            else:
                total_floor = 0
            slots = max(inferred_slots, total_floor)
        if slots < initial_maximum:
            raise RuntimeError(
                "TACE Opt3 per-centre capacity is smaller than the initial graph: "
                f"required={initial_maximum}, capacity={slots}"
            )
        self.initial_max_neighbors = initial_maximum
        self.neighbors_per_atom = int(slots)
        self.edge_capacity = self.num_atoms * self.neighbors_per_atom
        sink_count = _positive_int(
            options, "graph_sink_count", _DEFAULT_SINK_COUNT
        )
        self.builder = FixedShapeTACENeighborBuilder(
            num_atoms=self.num_atoms,
            cell=self.cell,
            pbc=self.pbc,
            cutoff=self.model_metadata["cutoff_a"],
            neighbors_per_atom=self.neighbors_per_atom,
            sink_count=sink_count,
        )
        self.static_data["edge_index"] = self.builder.edge_index
        self.static_data["edge_shifts"] = self.builder.edge_shifts

        exact_data = dict(self.static_data)
        exact_data["edge_index"] = initial_edge_index
        exact_data["edge_shifts"] = initial_edge_shifts
        with torch.enable_grad():
            official_outputs = self.model(exact_data)
        official_energy, official_forces = self._extract(official_outputs)
        self.reference_energy = official_energy.detach().clone()
        self.reference_forces = official_forces.detach().clone()

        representation = self.model.readout_fn.representation
        representation.radial_basis = _NeutralPaddingRadialBasis(
            representation.radial_basis,
            self.model_metadata["cutoff_a"],
        ).to(device=device, dtype=self.model_dtype)
        self.padding_density_masks_enabled = _enable_padding_density_masks_(
            self.model
        )
        self.builder.build(self.static_positions)
        with torch.enable_grad():
            fixed_outputs = self.model(self.static_data)
        fixed_energy, fixed_forces = self._extract(fixed_outputs)
        self.fixed_initial_energy = fixed_energy.detach().clone()
        self.fixed_initial_forces = fixed_forces.detach().clone()
        energy_atol = _positive_float(
            options, "graph_energy_atol", _DEFAULT_GRAPH_ENERGY_ATOL
        )
        force_atol = _positive_float(
            options, "graph_force_atol", _DEFAULT_GRAPH_FORCE_ATOL
        )
        rtol = _positive_float(options, "graph_rtol", 2.0e-5)
        self.validation = {
            "rtol": rtol,
            "energy_atol": energy_atol,
            "force_atol": force_atol,
            "fixed_builder_initial_energy_max_abs": _assert_close(
                "Opt3 initial energy",
                self.reference_energy,
                fixed_energy.detach(),
                rtol=rtol,
                atol=energy_atol,
            ),
            "fixed_builder_initial_force_max_abs": _assert_close(
                "Opt3 initial forces",
                self.reference_forces,
                fixed_forces.detach(),
                rtol=rtol,
                atol=force_atol,
            ),
            "numerical_validation_failure_policy": "report_only",
        }
        self.builder.reset_stats()

    def _extract(self, outputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        if (
            outputs.get("energy") is None
            or outputs.get("forces") is None
        ):
            raise RuntimeError(f"TACE model omitted energy/forces: {sorted(outputs)}")
        return (
            outputs["energy"].reshape(-1)[0],
            outputs["forces"].reshape(self.num_atoms, 3),
        )

    def evaluate(
        self, positions: Tensor, *, step: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Build topology and run TACE; this entire method is captured."""

        with torch.no_grad():
            self.static_positions.copy_(positions)
        self.builder.build(self.static_positions, step=step)
        with torch.enable_grad():
            outputs = self.model(self.static_data)
        energy, forces = self._extract(outputs)
        return forces.to(torch.float64), energy.to(torch.float64)


class TACEWholeStepGraph:
    """Own persistent MD state and exactly one whole-step CUDA Graph."""

    def __init__(
        self,
        potential: TACEWholeStepPotential,
        state: GPUMDState,
        integrator: BerendsenIntegrator | NoseHooverChainIntegrator,
        *,
        capture_warmup: int,
    ) -> None:
        if state.forces is None or state.potential_energy is None:
            raise ValueError("TACE Opt3 requires an evaluated initial state")
        self.potential = potential
        self.state = state
        self.integrator = integrator
        self.device = state.positions.device
        self.capture_warmup = int(capture_warmup)
        self.initial_positions = state.positions.clone()
        self.initial_momenta = state.momenta.clone()
        self.initial_forces = state.forces.clone()
        self.initial_energy = state.potential_energy.clone()
        self.initial_eta = (
            integrator.eta.clone()
            if isinstance(integrator, NoseHooverChainIntegrator)
            else None
        )
        self.initial_p_eta = (
            integrator.p_eta.clone()
            if isinstance(integrator, NoseHooverChainIntegrator)
            else None
        )
        self.step_counter = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.advance = torch.zeros((), device=self.device, dtype=torch.float64)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.capture_stream: torch.cuda.Stream | None = None
        self.capture_wall_time_s = 0.0
        self.capture_count = 0
        self.total_replays = 0
        self.production_replays = 0
        self.output_addresses_stable = False
        self.validation: dict[str, Any] = {}

    @torch.no_grad()
    def restore_initial_(self) -> None:
        assert self.state.forces is not None
        assert self.state.potential_energy is not None
        self.state.positions.copy_(self.initial_positions)
        self.state.momenta.copy_(self.initial_momenta)
        self.state.forces.copy_(self.initial_forces)
        self.state.potential_energy.copy_(self.initial_energy)
        self.step_counter.zero_()
        self.advance.zero_()
        if isinstance(self.integrator, NoseHooverChainIntegrator):
            assert self.initial_eta is not None and self.initial_p_eta is not None
            self.integrator.eta.copy_(self.initial_eta)
            self.integrator.p_eta.copy_(self.initial_p_eta)

    def _step_values(
        self,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        state = self.state
        assert state.forces is not None and state.potential_energy is not None
        if isinstance(self.integrator, BerendsenIntegrator):
            temperature = (
                2.0
                * self.integrator.kinetic_energy(state.momenta)
                / (self.integrator.degrees_of_freedom * units.kB)
            ).clamp_min(1.0e-12)
            scale = torch.sqrt(
                1.0
                + (
                    self.integrator.target_temperature / temperature - 1.0
                )
                * (self.integrator.dt / self.integrator.taut)
            ).clamp(min=0.9, max=1.1)
            half_momenta = state.momenta * scale
            half_momenta = (
                half_momenta + 0.5 * self.integrator.dt * state.forces
            )
            half_momenta = half_momenta - half_momenta.sum(
                dim=0, keepdim=True
            ) / float(half_momenta.shape[0])
            advanced_positions = (
                state.positions
                + self.integrator.dt
                * half_momenta
                / self.integrator.masses
            )
            eta_final = p_eta_final = None
        elif isinstance(self.integrator, NoseHooverChainIntegrator):
            dt2 = self.integrator.dt / 2.0
            half_momenta, eta_half, p_eta_half = _integrate_nhc_pure(
                state.momenta,
                self.integrator.eta,
                self.integrator.p_eta,
                self.integrator,
                dt2,
            )
            half_momenta = half_momenta + dt2 * state.forces
            advanced_positions = (
                state.positions
                + self.integrator.dt
                * half_momenta
                / self.integrator.masses
            )
        else:
            raise TypeError(
                f"Unsupported TACE Opt3 integrator {type(self.integrator).__name__}"
            )

        positions = state.positions + self.advance * (
            advanced_positions - state.positions
        )
        forces, energy = self.potential.evaluate(
            positions,
            step=self.step_counter + self.advance.to(torch.long),
        )
        if isinstance(self.integrator, BerendsenIntegrator):
            momenta = half_momenta + 0.5 * self.integrator.dt * forces
        else:
            assert isinstance(self.integrator, NoseHooverChainIntegrator)
            dt2 = self.integrator.dt / 2.0
            half_momenta = half_momenta + dt2 * forces
            momenta, eta_final, p_eta_final = _integrate_nhc_pure(
                half_momenta,
                eta_half,
                p_eta_half,
                self.integrator,
                dt2,
            )
        momenta = state.momenta + self.advance * (momenta - state.momenta)
        return positions, momenta, forces, energy, eta_final, p_eta_final

    def _graph_body(self) -> None:
        state = self.state
        assert state.forces is not None and state.potential_energy is not None
        positions, momenta, forces, energy, eta_final, p_eta_final = (
            self._step_values()
        )
        with torch.no_grad():
            state.positions.copy_(positions)
            state.momenta.copy_(momenta)
            state.forces.copy_(forces)
            state.potential_energy.copy_(energy)
            if isinstance(self.integrator, NoseHooverChainIntegrator):
                assert eta_final is not None and p_eta_final is not None
                self.integrator.eta.copy_(
                    self.integrator.eta
                    + self.advance * (eta_final - self.integrator.eta)
                )
                self.integrator.p_eta.copy_(
                    self.integrator.p_eta
                    + self.advance * (p_eta_final - self.integrator.p_eta)
                )
            self.step_counter.add_(self.advance.to(torch.long))

    def _persistent_tensors(self) -> tuple[Tensor, ...]:
        assert self.state.forces is not None
        assert self.state.potential_energy is not None
        tensors = [
            self.state.positions,
            self.state.momenta,
            self.state.forces,
            self.state.potential_energy,
            self.step_counter,
            self.advance,
            self.potential.static_positions,
            self.potential.builder.edge_index,
            self.potential.builder.edge_shifts,
        ]
        if isinstance(self.integrator, NoseHooverChainIntegrator):
            tensors.extend((self.integrator.eta, self.integrator.p_eta))
        return tuple(tensors)

    def _eager_one_step_reference(self) -> dict[str, Tensor]:
        """Evaluate initial forces, then advance one fixed-builder eager step."""

        self.restore_initial_()
        self.advance.zero_()
        self._graph_body()
        self.advance.fill_(1.0)
        positions, momenta, forces, energy, eta, p_eta = self._step_values()
        reference = {
            "positions": positions.detach().clone(),
            "momenta": momenta.detach().clone(),
            "forces": forces.detach().clone(),
            "energy": energy.detach().clone(),
        }
        if eta is not None and p_eta is not None:
            reference["eta"] = eta.detach().clone()
            reference["p_eta"] = p_eta.detach().clone()
        self.restore_initial_()
        self.potential.builder.reset_stats()
        return reference

    def _validate_graph_step(self, reference: dict[str, Tensor]) -> None:
        assert self.graph is not None
        self.restore_initial_()
        self.potential.builder.reset_stats()
        # Replay zero evaluates the fixed-builder initial force without moving
        # the MD or thermostat state. Replay one then advances one physical step.
        self.advance.zero_()
        self.graph.replay()
        self.advance.fill_(1.0)
        self.graph.replay()
        torch.cuda.synchronize(self.device)
        self.total_replays += 2
        assert self.state.forces is not None
        assert self.state.potential_energy is not None
        candidate = {
            "positions": self.state.positions,
            "momenta": self.state.momenta,
            "forces": self.state.forces,
            "energy": self.state.potential_energy,
        }
        if isinstance(self.integrator, NoseHooverChainIntegrator):
            candidate["eta"] = self.integrator.eta
            candidate["p_eta"] = self.integrator.p_eta
        nonfinite = [
            name
            for name, value in candidate.items()
            if not bool(torch.isfinite(value).all().item())
        ]
        if nonfinite:
            raise FloatingPointError(
                f"TACE Opt3 graph validation produced non-finite {nonfinite}"
            )
        errors = {
            name: float((candidate[name] - expected).abs().max().item())
            for name, expected in reference.items()
        }
        energy_atol = float(self.potential.validation["energy_atol"])
        force_atol = float(self.potential.validation["force_atol"])
        state_atol = 1.0e-10
        limits = {
            "positions": state_atol,
            "momenta": state_atol,
            "forces": force_atol,
            "energy": energy_atol,
            "eta": state_atol,
            "p_eta": state_atol,
        }
        self.validation = {
            "reference": "eager_fixed_builder_integrator_one_step",
            "max_abs_error": errors,
            "tolerances": {name: limits[name] for name in errors},
            "within_tolerance": all(
                error <= limits[name] for name, error in errors.items()
            ),
            "failure_policy": "nonfinite_or_address_error; tolerance_report_only",
        }
        self.restore_initial_()
        self.potential.builder.reset_stats()

    def capture(self) -> None:
        if self.graph is not None:
            raise RuntimeError("TACE whole-step CUDA Graph is already captured")
        eager_reference = self._eager_one_step_reference()
        current_stream = torch.cuda.current_stream(self.device)
        side_stream = torch.cuda.Stream(device=self.device)
        self.capture_stream = side_stream
        side_stream.wait_stream(current_stream)
        with torch.cuda.stream(side_stream):
            self.restore_initial_()
            self.advance.fill_(1.0)
            for _ in range(self.capture_warmup):
                self._graph_body()
            self.restore_initial_()
            self.advance.fill_(1.0)
            self.potential.builder.reset_stats()
        current_stream.wait_stream(side_stream)
        torch.cuda.synchronize(self.device)

        addresses = tuple(tensor.data_ptr() for tensor in self._persistent_tensors())
        started = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=side_stream), torch.enable_grad():
                self._graph_body()
        except Exception as exc:
            raise RuntimeError(
                "TACE Opt3 whole-step CUDA Graph capture failed; eager fallback "
                "is forbidden"
            ) from exc
        torch.cuda.synchronize(self.device)
        self.capture_wall_time_s = time.perf_counter() - started
        self.graph = graph
        self.capture_count = 1
        self.restore_initial_()
        self.potential.builder.reset_stats()
        self._validate_graph_step(eager_reference)
        self.output_addresses_stable = addresses == tuple(
            tensor.data_ptr() for tensor in self._persistent_tensors()
        )
        if not self.output_addresses_stable:
            raise RuntimeError("TACE Opt3 persistent state addresses changed")

    def reset_production(self) -> None:
        if self.graph is None:
            raise RuntimeError("Capture must complete before production")
        self.restore_initial_()
        self.potential.builder.reset_stats()
        self.production_replays = 0

    def evaluate_initial(self) -> None:
        """Replay the measured fixed-builder initial energy/force evaluation."""

        self.advance.zero_()
        self.step()
        self.advance.fill_(1.0)

    def step(self) -> None:
        if self.graph is None:
            raise RuntimeError("Capture must complete before replay")
        try:
            self.graph.replay()
        except Exception as exc:
            raise RuntimeError(
                "TACE Opt3 whole-step CUDA Graph replay failed; eager fallback "
                "is forbidden"
            ) from exc
        self.total_replays += 1
        self.production_replays += 1

    def raise_if_overflow(self) -> None:
        stats = self.potential.builder.stats()
        if int(stats["fixed_builder_capacity_misses"]):
            raise RuntimeError(
                "TACE Opt3 per-centre neighbor capacity overflow: "
                f"required={stats['fixed_builder_max_overflow_required']}, "
                f"capacity={self.potential.neighbors_per_atom}; no topology was "
                "silently accepted and eager fallback is forbidden"
            )

    def stats(self) -> dict[str, Any]:
        return {
            **self.potential.builder.stats(),
            "cuda_graph_capture_count": self.capture_count,
            "cuda_graph_capture_wall_time_s": self.capture_wall_time_s,
            "cuda_graph_total_replays": self.total_replays,
            "cuda_graph_production_replays": self.production_replays,
            "cuda_graph_replay_output_addresses_stable": (
                self.output_addresses_stable
            ),
            "whole_step_eager_graph_validation": self.validation,
        }


def _nonnegative_float(
    options: dict[str, Any], key: str, default: float
) -> float:
    value = options.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"TACE Opt3 {key} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"TACE Opt3 {key} must be finite and non-negative")
    return value


def _validate_request(request: MDRunRequest) -> None:
    if request.model != "tace" or request.stage != "opt3":
        raise ValueError("tace.md_stages.opt3 owns only model='tace', stage='opt3'")
    if request.backend != _BACKEND:
        raise ValueError("TACE Opt3 backend must be 'whole-step-cuda-graph'")
    if request.config.device.split(":", maxsplit=1)[0] != "cuda":
        raise ValueError("TACE Opt3 requires a CUDA device")
    if request.config.dtype != "float64":
        raise ValueError("TACE Opt3 requires --dtype float64 for the MD state")
    if request.atoms.constraints:
        raise NotImplementedError("TACE Opt3 does not silently ignore constraints")
    if len(request.atoms) < 2:
        raise ValueError("NVT MD requires at least two atoms")
    if not bool(np.all(request.atoms.pbc)):
        raise ValueError("TACE Opt3 currently requires full periodic boundaries")
    if request.config.collect_trajectory or request.output_path is not None:
        raise NotImplementedError(
            "TACE Opt3 captures energy/forces but not stress; use --no-statistics "
            "for the short Matbench timing/observation protocol"
        )
    if request.options.get("compute_stress", False) is not False:
        raise ValueError("TACE Opt3 does not capture stress")
    if request.options.get("model_dtype", "checkpoint") != "checkpoint":
        raise ValueError("TACE Opt3 fixes model_dtype='checkpoint'")
    _positive_int(request.options, "graph_warmup_steps", _DEFAULT_CAPTURE_WARMUP)
    _positive_float(request.options, "graph_rtol", 2.0e-5)
    _positive_float(
        request.options, "graph_energy_atol", _DEFAULT_GRAPH_ENERGY_ATOL
    )
    _positive_float(
        request.options, "graph_force_atol", _DEFAULT_GRAPH_FORCE_ATOL
    )
    _nonnegative_float(
        request.options, "graph_neighbor_margin", _DEFAULT_NEIGHBOR_MARGIN
    )
    _positive_int(
        request.options, "graph_neighbor_step", _DEFAULT_NEIGHBOR_STEP
    )
    _positive_int(request.options, "graph_sink_count", _DEFAULT_SINK_COUNT)


def _configure_opt3_runtime() -> None:
    _set_exact_acceleration_environment(set())
    configure_torch_baseline()


def run_md(request: MDRunRequest) -> MDRunResult:
    """Run native TACE with one whole-step CUDA Graph per request."""

    _validate_request(request)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; TACE Opt3 never falls back to eager")
    if not hasattr(torch.cuda, "CUDAGraph"):
        raise RuntimeError("This PyTorch build does not provide CUDA Graph support")
    _configure_opt3_runtime()
    device = torch.device(request.config.device)
    config = request.config
    atoms = request.atoms.copy()
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=config.temperature_k,
        rng=np.random.default_rng(config.seed),
    )
    positions = torch.as_tensor(
        np.asarray(atoms.positions), dtype=torch.float64, device=device
    ).clone()
    momenta = torch.as_tensor(
        np.asarray(atoms.get_momenta()), dtype=torch.float64, device=device
    ).clone()
    masses = torch.as_tensor(
        np.asarray(atoms.get_masses()), dtype=torch.float64, device=device
    ).clone()
    potential = TACEWholeStepPotential(
        atoms, request.model_path, device=device, options=request.options
    )
    state = GPUMDState(
        positions=positions,
        momenta=momenta,
        forces=potential.fixed_initial_forces.to(torch.float64).clone(),
        potential_energy=potential.fixed_initial_energy.to(torch.float64).clone(),
    )
    integrator = _build_integrator(request, masses)
    capture_warmup = _positive_int(
        request.options, "graph_warmup_steps", _DEFAULT_CAPTURE_WARMUP
    )
    runner = TACEWholeStepGraph(
        potential,
        state,
        integrator,
        capture_warmup=capture_warmup,
    )
    runner.capture()
    if config.warmup_steps:
        runner.advance.fill_(1.0)
        for _ in range(config.warmup_steps):
            runner.step()
        torch.cuda.synchronize(device)
        runner.raise_if_overflow()
    runner.reset_production()

    observation_steps = set(config.observation_steps)
    observations = []
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    runner.evaluate_initial()
    runner.raise_if_overflow()
    if config.collect_statistics and 0 in observation_steps:
        observations.append(_record_observation(state, 0, masses))
    for step in range(1, config.steps + 1):
        runner.step()
        if config.collect_statistics and step in observation_steps:
            runner.raise_if_overflow()
            observations.append(_record_observation(state, step, masses))
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    runner.raise_if_overflow()
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1.0e9
    expected_replays = config.steps + 1
    if runner.production_replays != expected_replays:
        raise RuntimeError(
            "TACE Opt3 replay count mismatch: "
            f"observed={runner.production_replays}, expected={expected_replays}"
        )
    _validate_final_state(state)
    graph_stats = runner.stats()

    final_atoms = atoms.copy()
    final_atoms.set_positions(state.positions.detach().cpu().numpy())
    final_atoms.set_momenta(state.momenta.detach().cpu().numpy())
    result = MDRunResult(
        model=request.model,
        stage=request.stage,
        completed_steps=config.steps,
        elapsed_s=elapsed,
        peak_cuda_memory_gb=peak_memory_gb,
        final_atoms=final_atoms,
        observations=observations,
        metadata={
            "engine": "torch-sim-tace-gpu-resident-whole-step-cuda-graph",
            "backend": request.backend,
            "model_path": str(Path(request.model_path).resolve()),
            "torch_sim_version": _distribution_version("torch-sim-atomistic"),
            "model_dtype_policy": "checkpoint",
            "model_dtype": str(potential.model_dtype),
            "md_state_dtype": "float64",
            "md_state_device": str(device),
            "positions_momenta_forces_cuda_resident": True,
            "neighborlist_backend": "fixed_shape_pbc_candidate_builder",
            "neighborlist_device": "cuda",
            "neighborlist_in_cuda_graph": True,
            "integrator_in_cuda_graph": True,
            "state_update_in_cuda_graph": True,
            "model_in_cuda_graph": True,
            "force_autograd_in_cuda_graph": True,
            "cuda_graph_scope": "whole_step",
            "graph_capture_scope": "whole-md-step",
            "capture_count": runner.capture_count,
            "production_replays": runner.production_replays,
            "expected_production_replays": expected_replays,
            "production_graph_replay_count": runner.production_replays,
            "expected_production_graph_replay_count": expected_replays,
            "production_graph_replay_count_verified": True,
            "initial_force_evaluation_in_measured_region": True,
            "capacity_overflow_count": graph_stats[
                "fixed_builder_capacity_misses"
            ],
            "neighbor_list_inside_cuda_graph": True,
            "cuda_graph_neighbor_build_inside": True,
            "fixed_address_model_inputs": True,
            "fixed_edge_capacity": potential.edge_capacity,
            "initial_edge_count": potential.initial_edge_count,
            "initial_max_neighbors": potential.initial_max_neighbors,
            "neighbors_per_atom": potential.neighbors_per_atom,
            "capacity_policy": "esen_uniform_per_centre_cap",
            "capacity_total_to_per_atom_guard_slots": 1,
            "edge_padding": "distributed_far_shifted_self_edge_sink",
            "sink_padding": "distributed_far_shifted_self_edges",
            "padding_node_policy": "real_nodes_only_no_dummy_readout_atoms",
            "padding_density_masks_enabled": (
                potential.padding_density_masks_enabled
            ),
            "edge_overflow_policy": "device_detect_error_no_fallback",
            "transaction_rollback": False,
            "graph_buckets": False,
            "capture_failure_policy": "error_no_fallback",
            "validation_failure_policy": "report_only_energy_force",
            "eager_fixed_builder_validation": potential.validation,
            "model_implementation": "native_e3nn_eager",
            "tace_accelerators": [],
            "detected_acceleration_modules": potential.model_metadata[
                "detected_acceleration_modules"
            ],
            "openequivariance": False,
            "cuequivariance": False,
            "equitorch": False,
            "triton": False,
            "compile": False,
            "aoti": False,
            "amp": False,
            "tf32": False,
            "model_specific_fusion": False,
            "compute_stress": False,
            "integrator": config.integrator,
            "integrator_implementation": "tace.md_stages.opt3",
            "warmup_steps": config.warmup_steps,
            "warmup_full_state_restored": True,
            "cutoff_a": potential.model_metadata["cutoff_a"],
            "target_properties": potential.model_metadata["target_properties"],
            **graph_stats,
        },
    )
    validate_result(request, result)
    return result


__all__ = [
    "TACEWholeStepGraph",
    "TACEWholeStepPotential",
    "_validate_request",
    "run_md",
]
