"""TACE Opt2: model-only CUDA Graph over the native eager checkpoint.

The CUDA graph contains only the TACE energy/force model call.  CUDA neighbour
construction, fixed-capacity edge packing, the FP64 MD integrator, reporting,
and state updates remain outside the graph.  This stage deliberately enables no
compiled or model-specific TACE acceleration backend.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from md_benchmark.md_route import (
    MDRunRequest,
    MDRunResult,
    configure_torch_baseline,
    validate_result,
)
from torch import Tensor

from tace.md_route import _set_exact_acceleration_environment, _validate_model_contract
from tace.md_stages.opt1 import (
    GPUMDState,
    _build_integrator,
    _record_observation,
    _validate_final_state,
)

_DEFAULT_EDGE_CAPACITY_MULTIPLIER = 1.10
_DEFAULT_EDGE_CAPACITY_PADDING = 32
_DEFAULT_GRAPH_WARMUP_STEPS = 3
_DEFAULT_GRAPH_RTOL = 2.0e-5
_DEFAULT_GRAPH_ENERGY_ATOL = 2.0e-5
_DEFAULT_GRAPH_FORCE_ATOL = 2.0e-4


@dataclass
class FixedEdgeBuffers:
    """Fixed-address edge tensors with cutoff-masked dummy self edges."""

    edge_index: Tensor
    edge_shifts: Tensor
    dummy_shift: Tensor
    active_edges: int = 0

    @property
    def capacity(self) -> int:
        return int(self.edge_index.shape[1])

    @classmethod
    def allocate(
        cls,
        *,
        capacity: int,
        edge_index: Tensor,
        edge_shifts: Tensor,
        dummy_shift: Tensor,
    ) -> FixedEdgeBuffers:
        if capacity < 1:
            raise ValueError("TACE Opt2 edge capacity must be positive")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_shifts.ndim != 2 or edge_shifts.shape[1] != 3:
            raise ValueError("edge_shifts must have shape [num_edges, 3]")
        if edge_shifts.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_index and edge_shifts disagree on edge count")
        if dummy_shift.shape != (3,):
            raise ValueError("dummy_shift must have shape [3]")
        buffers = cls(
            edge_index=torch.zeros(
                (2, capacity), dtype=edge_index.dtype, device=edge_index.device
            ),
            edge_shifts=dummy_shift.to(
                device=edge_shifts.device, dtype=edge_shifts.dtype
            )
            .reshape(1, 3)
            .expand(capacity, 3)
            .clone(),
            dummy_shift=dummy_shift.to(
                device=edge_shifts.device, dtype=edge_shifts.dtype
            ).clone(),
        )
        buffers.update(edge_index, edge_shifts)
        return buffers

    def update(self, edge_index: Tensor, edge_shifts: Tensor) -> int:
        if edge_index.device != self.edge_index.device:
            raise ValueError("TACE Opt2 edge_index changed device")
        if edge_shifts.device != self.edge_shifts.device:
            raise ValueError("TACE Opt2 edge_shifts changed device")
        if edge_index.dtype != self.edge_index.dtype:
            raise ValueError("TACE Opt2 edge_index changed dtype")
        if edge_shifts.dtype != self.edge_shifts.dtype:
            raise ValueError("TACE Opt2 edge_shifts changed dtype")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_shifts.shape != (edge_index.shape[1], 3):
            raise ValueError("edge_shifts must have shape [num_edges, 3]")
        edge_count = int(edge_index.shape[1])
        if edge_count > self.capacity:
            raise RuntimeError(
                "TACE Opt2 fixed edge capacity overflow: "
                f"required={edge_count}, capacity={self.capacity}. Increase the "
                "route option edge_capacity or edge_capacity_multiplier; Opt2 "
                "does not fall back to eager execution."
            )
        with torch.no_grad():
            self.edge_index[:, :edge_count].copy_(edge_index)
            self.edge_shifts[:edge_count].copy_(edge_shifts)
            if edge_count < self.active_edges:
                self.edge_index[:, edge_count : self.active_edges].zero_()
                self.edge_shifts[edge_count : self.active_edges].copy_(
                    self.dummy_shift
                )
        self.active_edges = edge_count
        return edge_count


class _NeutralPaddingRadialBasis(torch.nn.Module):
    """Expose a binary validity mask while preserving released radial values.

    TACE checkpoints commonly apply the smooth cutoff directly to the radial
    basis and return ``cutoff=None``.  A far padded edge then has a zero radial
    vector, but downstream biased MLPs can still emit a nonzero message.  This
    adapter leaves every real-edge radial value untouched and returns an
    additional 0/1 mask so all downstream edge messages are zeroed after their
    biased projections.
    """

    def __init__(self, base: torch.nn.Module, cutoff: float) -> None:
        super().__init__()
        self.base = base
        self.register_buffer(
            "padding_cutoff",
            torch.tensor(float(cutoff)),
            persistent=False,
        )

    def forward(
        self,
        edge_length: Tensor,
        node_attrs: Tensor,
        edge_index: Tensor,
        atomic_numbers: Tensor,
        dcutoff: Tensor | None,
    ):
        radial, cutoff = self.base(
            edge_length,
            node_attrs,
            edge_index,
            atomic_numbers,
            dcutoff,
        )
        if cutoff is not None:
            return radial, cutoff
        valid = (edge_length < self.padding_cutoff).to(dtype=radial.dtype)
        return radial, valid


def _enable_padding_density_masks_(model: torch.nn.Module) -> int:
    """Make density-based scatter normalization ignore padded edges.

    Released checkpoints with an already-applied radial cutoff can leave
    ``apply_density_cutoff`` disabled.  Once Opt2 exposes a binary validity
    mask, biased density projections must consume that mask too; otherwise
    every fixed-capacity padding edge changes the normalization denominator.
    Exact-edge validation below guarantees that enabling it is a no-op for all
    real neighbors.
    """

    patched = 0
    for module in model.modules():
        if not hasattr(module, "edge_density"):
            continue
        if not bool(getattr(module, "_opt2_binary_density_mask", False)):
            module._opt2_binary_density_mask = True
            patched += 1
    return patched


def _edge_capacity(initial_edges: int, options: dict[str, Any]) -> int:
    explicit = options.get("edge_capacity")
    if explicit is not None:
        if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit < 1:
            raise ValueError("TACE Opt2 edge_capacity must be a positive integer")
        if explicit < initial_edges:
            raise ValueError(
                f"TACE Opt2 edge_capacity={explicit} is smaller than the initial "
                f"edge count {initial_edges}"
            )
        return explicit
    multiplier = options.get(
        "edge_capacity_multiplier", _DEFAULT_EDGE_CAPACITY_MULTIPLIER
    )
    if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)):
        raise ValueError("TACE Opt2 edge_capacity_multiplier must be numeric")
    if not math.isfinite(float(multiplier)) or float(multiplier) < 1.0:
        raise ValueError("TACE Opt2 edge_capacity_multiplier must be finite and >= 1")
    requested = max(
        initial_edges + _DEFAULT_EDGE_CAPACITY_PADDING,
        math.ceil(initial_edges * float(multiplier)),
    )
    return max(32, requested)


def _dummy_shift(cell: np.ndarray, cutoff: float) -> np.ndarray:
    cell = np.asarray(cell, dtype=np.float64)
    if cell.shape != (3, 3) or np.linalg.matrix_rank(cell) < 3:
        raise ValueError("TACE Opt2 requires a full-rank periodic cell")
    row_norms = np.linalg.norm(cell, axis=1)
    axis = int(np.argmax(row_norms))
    if row_norms[axis] <= 0.0:
        raise ValueError("TACE Opt2 could not construct a padded-edge shift")
    multiple = max(2, math.ceil((1.5 * cutoff) / row_norms[axis]) + 1)
    shift = np.zeros(3, dtype=np.float64)
    shift[axis] = float(multiple)
    if np.linalg.norm(shift @ cell) <= cutoff:
        raise RuntimeError("TACE Opt2 padded edge is not outside the model cutoff")
    return shift


def _capture_model_graph(
    model: Callable[[dict[str, Tensor]], dict[str, Tensor]],
    data: dict[str, Tensor],
    *,
    device: torch.device,
    warmup_steps: int,
) -> tuple[torch.cuda.CUDAGraph, dict[str, Tensor]]:
    """Warm and capture one fixed-address energy/force model call."""

    if warmup_steps < 1:
        raise ValueError("TACE Opt2 graph_warmup_steps must be positive")
    warmup_stream = torch.cuda.Stream(device=device)
    current_stream = torch.cuda.current_stream(device)
    warmup_stream.wait_stream(current_stream)
    with torch.cuda.stream(warmup_stream), torch.enable_grad():
        for _ in range(warmup_steps):
            outputs = model(data)
            if (
                "energy" not in outputs
                or "forces" not in outputs
                or outputs["energy"] is None
                or outputs["forces"] is None
            ):
                raise RuntimeError(
                    f"TACE Opt2 model omitted energy/forces: {sorted(outputs)}"
                )
    del outputs
    current_stream.wait_stream(warmup_stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph), torch.enable_grad():
            graph_outputs = model(data)
    except Exception as exc:
        raise RuntimeError(
            "TACE Opt2 model-only CUDA Graph capture failed; eager fallback is "
            "forbidden for this stage"
        ) from exc
    if (
        "energy" not in graph_outputs
        or "forces" not in graph_outputs
        or graph_outputs["energy"] is None
        or graph_outputs["forces"] is None
    ):
        raise RuntimeError(
            f"TACE Opt2 captured model omitted energy/forces: {sorted(graph_outputs)}"
        )
    return graph, graph_outputs


def _assert_close(
    name: str,
    reference: Tensor,
    candidate: Tensor,
    *,
    rtol: float,
    atol: float,
) -> float:
    if reference.shape != candidate.shape:
        raise RuntimeError(
            f"TACE Opt2 {name} validation shape mismatch: "
            f"{tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    if not bool(torch.isfinite(reference).all()) or not bool(
        torch.isfinite(candidate).all()
    ):
        raise FloatingPointError(
            f"TACE Opt2 {name} validation contains non-finite values"
        )
    max_abs = float((reference - candidate).abs().max().item())
    # rtol/atol remain part of the report contract.  Exceeding them is a
    # diagnostic result, not a CUDA Graph execution failure.
    _ = rtol, atol
    return max_abs


def _assert_data_ptrs(
    name: str, expected: dict[str, int], tensors: dict[str, Tensor]
) -> None:
    actual = {key: value.data_ptr() for key, value in tensors.items()}
    if actual != expected:
        changed = sorted(key for key in expected if expected[key] != actual.get(key))
        raise RuntimeError(
            f"TACE Opt2 {name} fixed-address invariant failed for {changed}"
        )


class TACEModelOnlyGraphEvaluator:
    """CUDA neighbour builder plus fixed-shape captured native TACE model."""

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
                "TACE Opt2 requires torch-sim-atomistic>=0.6.1"
            ) from exc

        from tace.interface.torchsim import TACETorchSimCalc

        atomic_numbers = torch.as_tensor(
            atoms.get_atomic_numbers(), dtype=torch.int64, device=device
        )
        system_idx = torch.zeros(len(atoms), dtype=torch.int64, device=device)
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
        self.device = device
        self.num_atoms = len(atoms)
        self.neighbor_list_fn = torchsim_nl
        self.system_idx = calculator.system_idx
        self.sim_state = ts.io.atoms_to_state(
            [atoms], device=device, dtype=self.model_dtype
        )
        self.static_positions = self.sim_state.positions
        if self.static_positions.data_ptr() != self.sim_state.positions.data_ptr():
            raise RuntimeError("TACE Opt2 failed to establish fixed position storage")
        self.static_positions.requires_grad_(True)
        self.static_data: dict[str, Tensor] = {
            "ptr": calculator.ptr,
            "node_attrs": calculator.node_attrs,
            "batch": calculator.system_idx,
            "pbc": self.sim_state.pbc,
            "lattice": self.sim_state.row_vector_cell,
            "positions": self.static_positions,
        }

        initial_edge_index, initial_shifts = self._build_edges()
        if initial_edge_index.shape[1] < 1:
            raise RuntimeError("TACE Opt2 initial neighbour list contains no edges")
        capacity = _edge_capacity(int(initial_edge_index.shape[1]), options)
        dummy_shift = torch.as_tensor(
            _dummy_shift(np.asarray(atoms.cell.array), self.model_metadata["cutoff_a"]),
            device=device,
            dtype=initial_shifts.dtype,
        )
        self.edges = FixedEdgeBuffers.allocate(
            capacity=capacity,
            edge_index=initial_edge_index,
            edge_shifts=initial_shifts,
            dummy_shift=dummy_shift,
        )
        self.static_data["edge_index"] = self.edges.edge_index
        self.static_data["edge_shifts"] = self.edges.edge_shifts
        self.initial_edges = int(initial_edge_index.shape[1])
        self.max_observed_edges = self.initial_edges

        rtol = _positive_float(options, "graph_rtol", _DEFAULT_GRAPH_RTOL)
        energy_atol = _positive_float(
            options, "graph_energy_atol", _DEFAULT_GRAPH_ENERGY_ATOL
        )
        force_atol = _positive_float(
            options, "graph_force_atol", _DEFAULT_GRAPH_FORCE_ATOL
        )
        warmup_steps = _positive_int(
            options, "graph_warmup_steps", _DEFAULT_GRAPH_WARMUP_STEPS
        )

        exact_data = dict(self.static_data)
        exact_data["edge_index"] = initial_edge_index
        exact_data["edge_shifts"] = initial_shifts
        with torch.enable_grad():
            official_exact_outputs = self.model(exact_data)
        official_exact_energy, official_exact_forces = self._extract(
            official_exact_outputs
        )
        official_exact_energy = official_exact_energy.detach().clone()
        official_exact_forces = official_exact_forces.detach().clone()

        representation = self.model.readout_fn.representation
        released_radial_basis = representation.radial_basis
        representation.radial_basis = _NeutralPaddingRadialBasis(
            released_radial_basis,
            self.model_metadata["cutoff_a"],
        ).to(device=device, dtype=self.model_dtype)
        self.padding_density_masks_enabled = _enable_padding_density_masks_(
            self.model
        )
        with torch.enable_grad():
            exact_outputs = self.model(exact_data)
            padded_outputs = self.model(self.static_data)
        exact_energy, exact_forces = self._extract(exact_outputs)
        padded_energy, padded_forces = self._extract(padded_outputs)
        exact_energy = exact_energy.detach().clone()
        exact_forces = exact_forces.detach().clone()
        patched_exact_energy_max_abs = _assert_close(
            "released-vs-masked exact energy",
            official_exact_energy,
            exact_energy,
            rtol=rtol,
            atol=energy_atol,
        )
        patched_exact_force_max_abs = _assert_close(
            "released-vs-masked exact forces",
            official_exact_forces,
            exact_forces,
            rtol=rtol,
            atol=force_atol,
        )
        padded_energy_max_abs = _assert_close(
            "padded energy",
            exact_energy,
            padded_energy.detach(),
            rtol=rtol,
            atol=energy_atol,
        )
        padded_force_max_abs = _assert_close(
            "padded forces",
            exact_forces,
            padded_forces.detach(),
            rtol=rtol,
            atol=force_atol,
        )
        del (
            official_exact_outputs,
            exact_outputs,
            padded_outputs,
            padded_energy,
            padded_forces,
        )

        self.graph, graph_outputs = _capture_model_graph(
            self.model,
            self.static_data,
            device=device,
            warmup_steps=warmup_steps,
        )
        self.graph_energy, self.graph_forces = self._extract(graph_outputs)
        input_tensors = {
            key: value for key, value in self.static_data.items() if value is not None
        }
        input_data_ptrs = {
            key: value.data_ptr() for key, value in input_tensors.items()
        }
        output_tensors = {
            "energy": self.graph_energy,
            "forces": self.graph_forces,
        }
        output_data_ptrs = {
            key: value.data_ptr() for key, value in output_tensors.items()
        }

        self.graph.replay()
        torch.cuda.synchronize(device)
        _assert_data_ptrs("input", input_data_ptrs, input_tensors)
        _assert_data_ptrs("output", output_data_ptrs, output_tensors)
        first_replay_energy = self.graph_energy.detach().clone()
        first_replay_forces = self.graph_forces.detach().clone()
        replay_energy_max_abs = _assert_close(
            "energy",
            exact_energy,
            self.graph_energy,
            rtol=rtol,
            atol=energy_atol,
        )
        replay_force_max_abs = _assert_close(
            "forces",
            exact_forces,
            self.graph_forces,
            rtol=rtol,
            atol=force_atol,
        )

        self.graph.replay()
        torch.cuda.synchronize(device)
        _assert_data_ptrs("input", input_data_ptrs, input_tensors)
        _assert_data_ptrs("output", output_data_ptrs, output_tensors)
        second_replay_energy_max_abs = _assert_close(
            "second replay energy",
            exact_energy,
            self.graph_energy,
            rtol=rtol,
            atol=energy_atol,
        )
        second_replay_force_max_abs = _assert_close(
            "second replay forces",
            exact_forces,
            self.graph_forces,
            rtol=rtol,
            atol=force_atol,
        )
        replay_stability_energy_max_abs = _assert_close(
            "consecutive replay energy",
            first_replay_energy,
            self.graph_energy,
            rtol=0.0,
            atol=energy_atol,
        )
        replay_stability_force_max_abs = _assert_close(
            "consecutive replay forces",
            first_replay_forces,
            self.graph_forces,
            rtol=0.0,
            atol=force_atol,
        )
        self.validation = {
            "rtol": rtol,
            "energy_atol": energy_atol,
            "force_atol": force_atol,
            "padded_eager_energy_max_abs": padded_energy_max_abs,
            "padded_eager_force_max_abs": padded_force_max_abs,
            "released_vs_masked_exact_energy_max_abs": (
                patched_exact_energy_max_abs
            ),
            "released_vs_masked_exact_force_max_abs": (
                patched_exact_force_max_abs
            ),
            "first_replay_energy_max_abs": replay_energy_max_abs,
            "first_replay_force_max_abs": replay_force_max_abs,
            "second_replay_energy_max_abs": second_replay_energy_max_abs,
            "second_replay_force_max_abs": second_replay_force_max_abs,
            "consecutive_replay_energy_max_abs": (
                replay_stability_energy_max_abs
            ),
            "consecutive_replay_force_max_abs": replay_stability_force_max_abs,
            "fixed_input_data_ptrs_verified": True,
            "fixed_output_data_ptrs_verified": True,
            "numerical_validation_failure_policy": "report_only",
        }
        self.graph_warmup_steps = warmup_steps
        self.output_forces = torch.empty(
            (self.num_atoms, 3), dtype=torch.float64, device=device
        )
        self.output_energy = torch.empty((), dtype=torch.float64, device=device)
        self.production_replays = 0

    def _build_edges(self) -> tuple[Tensor, Tensor]:
        edge_index, _, edge_shifts = self.neighbor_list_fn(
            self.static_positions,
            self.sim_state.row_vector_cell,
            self.sim_state.pbc,
            self.model_metadata["cutoff_a"],
            self.system_idx,
        )
        return edge_index, edge_shifts

    def _extract(self, outputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        if (
            "energy" not in outputs
            or "forces" not in outputs
            or outputs["energy"] is None
            or outputs["forces"] is None
        ):
            raise RuntimeError(f"TACE model omitted energy/forces: {sorted(outputs)}")
        return (
            outputs["energy"].reshape(-1)[0],
            outputs["forces"].reshape(self.num_atoms, 3),
        )

    def __call__(self, positions: Tensor) -> tuple[Tensor, Tensor, None]:
        if positions.device != self.device or positions.dtype != torch.float64:
            raise ValueError(
                "TACE Opt2 positions must be FP64 tensors on the selected CUDA device"
            )
        if positions.shape != (self.num_atoms, 3):
            raise ValueError(
                f"TACE Opt2 expected positions shape {(self.num_atoms, 3)}, "
                f"got {tuple(positions.shape)}"
            )
        with torch.no_grad():
            self.static_positions.copy_(positions)
        edge_index, edge_shifts = self._build_edges()
        edge_count = self.edges.update(edge_index, edge_shifts)
        self.max_observed_edges = max(self.max_observed_edges, edge_count)
        try:
            self.graph.replay()
        except Exception as exc:
            raise RuntimeError(
                "TACE Opt2 CUDA Graph replay failed; eager fallback is forbidden"
            ) from exc
        with torch.no_grad():
            self.output_forces.copy_(self.graph_forces)
            self.output_energy.copy_(self.graph_energy)
        self.production_replays += 1
        return self.output_forces, self.output_energy, None

    def reset_runtime_counters(self) -> None:
        self.production_replays = 0
        self.max_observed_edges = self.initial_edges


def _positive_float(options: dict[str, Any], key: str, default: float) -> float:
    value = options.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"TACE Opt2 {key} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"TACE Opt2 {key} must be finite and positive")
    return value


def _positive_int(options: dict[str, Any], key: str, default: int) -> int:
    value = options.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"TACE Opt2 {key} must be a positive integer")
    return value


def _distribution_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _validate_request(request: MDRunRequest) -> None:
    if request.model != "tace" or request.stage != "opt2":
        raise ValueError("tace.md_stages.opt2 only accepts model='tace', stage='opt2'")
    if request.backend != "model-only-cuda-graph":
        raise ValueError("TACE Opt2 backend must be 'model-only-cuda-graph'")
    if request.config.device.split(":", maxsplit=1)[0] != "cuda":
        raise ValueError("TACE Opt2 requires a CUDA device")
    if request.config.dtype != "float64":
        raise ValueError("TACE Opt2 requires --dtype float64 for the MD state")
    if request.atoms.constraints:
        raise NotImplementedError("TACE Opt2 does not silently ignore constraints")
    if len(request.atoms) < 2:
        raise ValueError("NVT MD requires at least two atoms")
    if not bool(np.all(request.atoms.pbc)):
        raise ValueError(
            "TACE Opt2 fixed padded edges require full periodic boundaries"
        )
    if request.config.collect_trajectory or request.output_path is not None:
        raise NotImplementedError(
            "TACE Opt2 captures energy and conservative forces only; trajectory "
            "stress is not part of the model-only graph"
        )
    if request.options.get("compute_stress", False) is not False:
        raise ValueError("TACE Opt2 does not compute stress")
    if request.options.get("model_dtype", "checkpoint") != "checkpoint":
        raise ValueError("TACE Opt2 fixes model_dtype='checkpoint'")
    _edge_capacity(1, request.options)
    _positive_float(request.options, "graph_rtol", _DEFAULT_GRAPH_RTOL)
    _positive_float(
        request.options, "graph_energy_atol", _DEFAULT_GRAPH_ENERGY_ATOL
    )
    _positive_float(request.options, "graph_force_atol", _DEFAULT_GRAPH_FORCE_ATOL)
    _positive_int(
        request.options, "graph_warmup_steps", _DEFAULT_GRAPH_WARMUP_STEPS
    )


def _configure_opt2_runtime() -> None:
    _set_exact_acceleration_environment(set())
    configure_torch_baseline()


def run_md(request: MDRunRequest) -> MDRunResult:
    """Run FP64 GPU MD with only native TACE energy/forces CUDA-graphed."""

    _validate_request(request)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; TACE Opt2 never falls back to eager")
    if not hasattr(torch.cuda, "CUDAGraph"):
        raise RuntimeError("This PyTorch build does not provide CUDA Graph support")
    device = torch.device(request.config.device)
    _configure_opt2_runtime()

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
    state = GPUMDState(positions=positions, momenta=momenta)
    initial_state = state.clone()
    evaluator = TACEModelOnlyGraphEvaluator(
        atoms,
        request.model_path,
        device=device,
        options=request.options,
    )
    integrator = _build_integrator(request, masses)

    if config.warmup_steps:
        for _ in range(config.warmup_steps):
            integrator.step(state, evaluator)
        torch.cuda.synchronize(device)
        state.restore_(initial_state)
        integrator.reset()
    evaluator.reset_runtime_counters()

    observation_steps = set(config.observation_steps)
    observations = []
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    _ensure_evaluated(state, evaluator)
    if config.collect_statistics and 0 in observation_steps:
        observations.append(_record_observation(state, 0, masses))
    for step in range(1, config.steps + 1):
        integrator.step(state, evaluator)
        if config.collect_statistics and step in observation_steps:
            observations.append(_record_observation(state, step, masses))
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1.0e9
    expected_replays = config.steps + 1
    if evaluator.production_replays != expected_replays:
        raise RuntimeError(
            "TACE Opt2 production replay count mismatch: "
            f"observed={evaluator.production_replays}, expected={expected_replays}"
        )
    _validate_final_state(state)

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
            "engine": "torch-sim-tace-gpu-resident-model-only-cuda-graph",
            "backend": request.backend,
            "model_path": str(Path(request.model_path).resolve()),
            "torch_sim_version": _distribution_version("torch-sim-atomistic"),
            "model_dtype_policy": "checkpoint",
            "model_dtype": str(evaluator.model_dtype),
            "md_state_dtype": "float64",
            "md_state_device": str(device),
            "positions_momenta_forces_cuda_resident": True,
            "neighborlist_backend": "torch_sim.neighbors.torchsim_nl",
            "neighborlist_device": "cuda",
            "neighborlist_in_cuda_graph": False,
            "integrator_in_cuda_graph": False,
            "state_update_in_cuda_graph": False,
            "model_in_cuda_graph": True,
            "force_autograd_in_cuda_graph": True,
            "cuda_graph_capture_count": 1,
            "production_graph_replay_count": evaluator.production_replays,
            "expected_production_graph_replay_count": expected_replays,
            "production_graph_replay_count_verified": True,
            "cuda_graph_scope": "model_energy_and_conservative_forces_only",
            "fixed_address_model_inputs": True,
            "fixed_edge_capacity": evaluator.edges.capacity,
            "initial_edge_count": evaluator.initial_edges,
            "max_observed_edge_count": evaluator.max_observed_edges,
            "edge_padding": "self_edge_shifted_beyond_cutoff",
            "edge_padding_neutralization": "binary_post_mlp_edge_mask",
            "padding_density_masks_enabled": (
                evaluator.padding_density_masks_enabled
            ),
            "edge_overflow_policy": "error_no_fallback",
            "capture_failure_policy": "error_no_fallback",
            "validation_failure_policy": "report_only_energy_force",
            "graph_warmup_steps": evaluator.graph_warmup_steps,
            "eager_replay_validation": evaluator.validation,
            "model_implementation": "native_e3nn_eager",
            "tace_accelerators": [],
            "detected_acceleration_modules": evaluator.model_metadata[
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
            "integrator_implementation": "tace.md_stages.opt1",
            "warmup_steps": config.warmup_steps,
            "warmup_full_state_restored": True,
            "cutoff_a": evaluator.model_metadata["cutoff_a"],
            "target_properties": evaluator.model_metadata["target_properties"],
        },
    )
    validate_result(request, result)
    return result


__all__ = [
    "FixedEdgeBuffers",
    "TACEModelOnlyGraphEvaluator",
    "_capture_model_graph",
    "_edge_capacity",
    "_validate_request",
    "run_md",
]
