"""Capture-safe fixed-shape PBC neighbour construction for TACE Opt3.

The production builder contains no ragged output and no host scalar reads.  It
enumerates the single-system PBC candidate universe once and writes a fixed
number of slots per centre.  Empty slots are far-shifted self edges distributed
over a sink bank.  TACE keeps only real nodes, so padding can never enter the
atomic-energy or force readout as an additional atom.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from md_benchmark.neighbor_utils import (
    displacement_exceeds_skin,
    make_slot_layout,
    normalize_neighbor_capacities,
    select_skin_candidates,
)


def neighbor_capacity_from_probe(
    maximum_neighbors: int,
    *,
    margin: float = 0.10,
    slot_step: int = 8,
) -> int:
    """Apply the eSEN CAP headroom and rounding policy."""

    if maximum_neighbors < 1:
        raise ValueError("maximum_neighbors must be positive")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and non-negative")
    if slot_step < 1:
        raise ValueError("slot_step must be positive")
    required = max(
        maximum_neighbors + 1,
        math.ceil(maximum_neighbors * (1.0 + margin)),
    )
    return int(math.ceil(required / slot_step) * slot_step)


def maximum_neighbors_in_graph(edge_index: Tensor, num_atoms: int) -> int:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if num_atoms < 1:
        raise ValueError("num_atoms must be positive")
    if edge_index.shape[1] == 0:
        return 0
    counts = torch.bincount(edge_index[1], minlength=num_atoms)[:num_atoms]
    return int(counts.max().item())


def _pbc_repetitions(
    cell: Tensor, cutoff: float, pbc: Tensor
) -> tuple[int, int, int]:
    cell64 = cell.detach().to(device="cpu", dtype=torch.float64).reshape(3, 3)
    pbc_cpu = pbc.detach().to(device="cpu", dtype=torch.bool).reshape(3)
    cross_a2a3 = torch.cross(cell64[1], cell64[2], dim=0)
    volume = torch.dot(cell64[0], cross_a2a3)
    if not bool(torch.isfinite(volume)) or float(volume.abs()) == 0.0:
        raise ValueError("Cannot enumerate PBC images for a singular cell")
    reciprocal = (
        cross_a2a3,
        torch.cross(cell64[2], cell64[0], dim=0),
        torch.cross(cell64[0], cell64[1], dim=0),
    )
    repetitions: list[int] = []
    for axis in range(3):
        if bool(pbc_cpu[axis]):
            inverse_plane_distance = torch.linalg.vector_norm(
                reciprocal[axis] / volume
            )
            repetitions.append(
                int(torch.ceil(cutoff * inverse_plane_distance).item())
            )
        else:
            repetitions.append(0)
    return tuple(repetitions)  # type: ignore[return-value]


class FixedShapeTACENeighborBuilder:
    """Write a fixed ``N * C`` TACE edge list using CUDA tensors only."""

    def __init__(
        self,
        *,
        num_atoms: int,
        cell: Tensor,
        pbc: Tensor,
        cutoff: float,
        neighbors_per_atom: int,
        neighbor_capacities: list[int] | Tensor | None = None,
        verlet_skin: float = 0.0,
        verlet_candidate_capacity: int | None = None,
        sink_count: int = 32,
        output_edge_index: Tensor | None = None,
        output_edge_shifts: Tensor | None = None,
    ) -> None:
        if num_atoms < 1:
            raise ValueError("num_atoms must be positive")
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        if neighbors_per_atom < 1:
            raise ValueError("neighbors_per_atom must be positive")
        if sink_count < 1:
            raise ValueError("sink_count must be positive")
        self.num_atoms = int(num_atoms)
        self.cutoff = float(cutoff)
        capacities = normalize_neighbor_capacities(
            neighbor_capacities,
            num_atoms=num_atoms,
            default=int(neighbors_per_atom),
        )
        (
            self.slot_centres,
            self.slot_ranks,
            self.selection_indices,
            self.neighbors_per_atom,
            self.edge_capacity,
        ) = make_slot_layout(capacities, device=cell.device)
        self.neighbor_capacities = torch.as_tensor(
            capacities, dtype=torch.long, device=cell.device
        )
        if verlet_skin < 0:
            raise ValueError("verlet_skin must be non-negative")
        self.verlet_skin = float(verlet_skin)
        self.verlet_candidate_capacity = verlet_candidate_capacity
        self.skin_candidate_ids: Tensor | None = None
        self.skin_candidate_mask: Tensor | None = None
        self.skin_reference_positions: Tensor | None = None
        self.skin_misses = torch.zeros((), dtype=torch.long, device=cell.device)
        self.skin_rebuilds = 0
        self.sink_count = min(int(sink_count), self.num_atoms)
        self.device = cell.device
        self.position_dtype = cell.dtype
        self.cell = cell.detach().reshape(3, 3).contiguous()
        self.inverse_cell = torch.linalg.inv(self.cell).contiguous()
        self.pbc = pbc.detach().to(device=cell.device, dtype=torch.bool).reshape(3)
        self.repetitions = _pbc_repetitions(cell, cutoff + self.verlet_skin, pbc)

        axes = [
            torch.arange(
                -repetition,
                repetition + 1,
                device=self.device,
                dtype=self.position_dtype,
            )
            for repetition in self.repetitions
        ]
        self.unit_cell_offsets = torch.cartesian_prod(*axes).reshape(-1, 3)
        self.num_cells = int(self.unit_cell_offsets.shape[0])
        self.candidates_per_atom = self.num_atoms * self.num_cells
        if self.neighbors_per_atom > self.candidates_per_atom:
            raise ValueError(
                "neighbors_per_atom exceeds the fixed candidate universe: "
                f"{self.neighbors_per_atom} > {self.candidates_per_atom}"
            )
        self.candidate_sources = torch.arange(
            self.num_atoms, device=self.device, dtype=torch.long
        ).repeat_interleave(self.num_cells)
        self.candidate_cell_offsets = self.unit_cell_offsets.repeat(
            self.num_atoms, 1
        )
        self.candidate_ids = torch.arange(
            self.candidates_per_atom, device=self.device, dtype=torch.long
        ).view(1, -1)
        self.slot_centres = self.slot_centres

        if output_edge_index is None:
            output_edge_index = torch.empty(
                (2, self.edge_capacity), device=self.device, dtype=torch.long
            )
        if output_edge_shifts is None:
            output_edge_shifts = torch.empty(
                (self.edge_capacity, 3),
                device=self.device,
                dtype=self.position_dtype,
            )
        if output_edge_index.shape != (2, self.edge_capacity):
            raise ValueError("output_edge_index has the wrong shape")
        if output_edge_shifts.shape != (self.edge_capacity, 3):
            raise ValueError("output_edge_shifts has the wrong shape")
        self.edge_index = output_edge_index
        self.edge_shifts = output_edge_shifts

        slots = torch.arange(
            self.edge_capacity, device=self.device, dtype=torch.long
        )
        self.sink_ids = slots.remainder(self.sink_count)
        cell_norms = torch.linalg.vector_norm(self.cell, dim=1)
        axis = int(cell_norms.argmax().item())
        axis_norm = float(cell_norms[axis].item())
        if not math.isfinite(axis_norm) or axis_norm <= 0:
            raise ValueError("Cannot construct a padding shift from the cell")
        far_shift = max(2, math.ceil((self.cutoff + 1.0) / axis_norm) + 1)
        self.padding_edge_shifts = self.edge_shifts.new_zeros(
            self.edge_capacity, 3
        )
        self.padding_edge_shifts[:, axis] = far_shift

        self.build_calls = torch.zeros((), device=self.device, dtype=torch.long)
        self.capacity_misses = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.first_overflow_step = torch.full(
            (), -1, device=self.device, dtype=torch.long
        )
        self.minimum_real_edges = torch.full(
            (), self.edge_capacity, device=self.device, dtype=torch.long
        )
        self.maximum_real_edges = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.maximum_neighbors = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.maximum_overflow_required = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.maximum_neighbors_by_atom = torch.zeros(
            self.num_atoms, device=self.device, dtype=torch.long
        )

    @torch.no_grad()
    def initialize_skin(self, positions: Tensor) -> None:
        if self.verlet_skin <= 0:
            return
        requested = self.verlet_candidate_capacity
        slots = max(self.neighbors_per_atom, int(requested)) if requested is not None else max(
            self.neighbors_per_atom * 2, self.neighbors_per_atom + 32
        )
        slots = min(slots, self.candidates_per_atom)
        selected, counts, selected_valid = select_skin_candidates(
            positions,
            self.candidate_sources,
            self.candidate_cell_offsets,
            self.cell,
            cutoff=self.cutoff + self.verlet_skin,
            slots_per_atom=slots,
        )
        torch._assert_async(
            (counts <= slots).all(),
            "TACE Opt3 Verlet candidate capacity is smaller than the "
            "cutoff+skin candidate count",
        )
        if self.skin_candidate_ids is None:
            self.skin_candidate_ids = selected
            self.skin_candidate_mask = selected_valid
            self.skin_reference_positions = positions.detach().clone()
        else:
            if self.skin_candidate_ids.shape != selected.shape:
                raise RuntimeError("Verlet candidate shape changed during rebuild")
            self.skin_candidate_ids.copy_(selected)
            assert self.skin_candidate_mask is not None
            self.skin_candidate_mask.copy_(selected_valid)
            assert self.skin_reference_positions is not None
            self.skin_reference_positions.copy_(positions)
        self.verlet_candidate_capacity = slots
        self.skin_rebuilds += 1

    def reset_stats(self) -> None:
        self.build_calls.zero_()
        self.capacity_misses.zero_()
        self.first_overflow_step.fill_(-1)
        self.minimum_real_edges.fill_(self.edge_capacity)
        self.maximum_real_edges.zero_()
        self.maximum_neighbors.zero_()
        self.maximum_overflow_required.zero_()
        self.maximum_neighbors_by_atom.zero_()
        self.skin_misses.zero_()
        self.skin_rebuilds = 0

    def build(
        self, positions: Tensor, *, step: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if positions.shape != (self.num_atoms, 3):
            raise ValueError(
                f"Expected positions {(self.num_atoms, 3)}, got {positions.shape}"
            )
        if positions.device != self.device:
            raise ValueError(
                f"Positions must be on {self.device}, got {positions.device}"
            )
        with torch.no_grad():
            if self.skin_candidate_ids is not None:
                assert self.skin_reference_positions is not None
                assert self.skin_candidate_mask is not None
                skin_miss = displacement_exceeds_skin(
                    positions,
                    self.skin_reference_positions,
                    self.verlet_skin,
                    self.cell,
                    self.pbc,
                    self.inverse_cell,
                )
                self.skin_misses.add_(skin_miss.to(torch.long))
                torch._assert_async(
                    ~skin_miss,
                    "TACE Opt3 Verlet skin exhausted; rebuild the candidate list",
                )
                cached = self.skin_candidate_ids.reshape(-1)
                candidate_sources = self.candidate_sources.index_select(
                    0, cached
                ).reshape(self.num_atoms, -1)
                candidate_cell_offsets = self.candidate_cell_offsets.index_select(
                    0, cached
                ).reshape(self.num_atoms, -1, 3)
                candidate_width = int(candidate_sources.shape[1])
                candidates = torch.arange(
                    candidate_width, device=self.device, dtype=torch.long
                ).reshape(1, -1).expand(self.num_atoms, -1)
                shifted_sources = positions.index_select(
                    0, candidate_sources.reshape(-1)
                ).reshape(self.num_atoms, candidate_width, 3) + torch.mm(
                    candidate_cell_offsets.reshape(-1, 3).to(dtype=positions.dtype),
                    self.cell.to(dtype=positions.dtype),
                ).reshape(self.num_atoms, candidate_width, 3)
                delta = shifted_sources - positions.unsqueeze(1)
                valid_candidates = self.skin_candidate_mask
            else:
                candidate_sources = self.candidate_sources
                candidate_cell_offsets = self.candidate_cell_offsets
                candidates = self.candidate_ids.expand(self.num_atoms, -1)
                shifted_sources = (
                    positions.index_select(0, candidate_sources)
                    + torch.mm(
                        candidate_cell_offsets.to(dtype=positions.dtype),
                        self.cell.to(dtype=positions.dtype),
                    )
                )
                delta = shifted_sources.unsqueeze(0) - positions.unsqueeze(1)
                valid_candidates = torch.ones_like(candidates, dtype=torch.bool)
            candidate_width = int(candidates.shape[1])
            distance_squared = delta.square().sum(dim=-1)
            valid = valid_candidates & (distance_squared <= self.cutoff * self.cutoff) & (
                distance_squared > 1.0e-8
            )
            counts = valid.sum(dim=1)
            candidate_order = torch.where(
                valid,
                candidates,
                torch.full_like(candidates, candidate_width),
            )
            selected_matrix = torch.topk(
                candidate_order,
                k=self.neighbors_per_atom,
                dim=1,
                largest=False,
                sorted=True,
            ).values
            selected_valid_matrix = selected_matrix < candidate_width
            safe_selected = selected_matrix.clamp_max(candidate_width - 1)
            if self.skin_candidate_ids is not None:
                selected_sources = torch.gather(
                    candidate_sources, 1, safe_selected
                )
                selected_offsets = torch.gather(
                    candidate_cell_offsets,
                    1,
                    safe_selected.unsqueeze(-1).expand(-1, -1, 3),
                )
            else:
                selected_sources = self.candidate_sources.index_select(
                    0, safe_selected.reshape(-1)
                ).reshape(self.num_atoms, -1)
                selected_offsets = self.candidate_cell_offsets.index_select(
                    0, safe_selected.reshape(-1)
                ).reshape(self.num_atoms, -1, 3)
            selected_valid = selected_valid_matrix.reshape(-1).index_select(
                0, self.selection_indices
            )
            sources = selected_sources.reshape(-1).index_select(
                0, self.selection_indices
            )
            offsets = selected_offsets.reshape(-1, 3).index_select(
                0, self.selection_indices
            )
            self.edge_index[0].copy_(
                torch.where(selected_valid, sources, self.sink_ids)
            )
            self.edge_index[1].copy_(
                torch.where(selected_valid, self.slot_centres, self.sink_ids)
            )
            # TACE forms r_target - r_source + shift @ cell.  The candidate
            # universe above uses r_source + offset @ cell - r_target.
            self.edge_shifts.copy_(
                torch.where(
                    selected_valid.unsqueeze(1),
                    -offsets.to(dtype=self.edge_shifts.dtype),
                    self.padding_edge_shifts,
                )
            )

            real_edges = selected_valid.sum()
            maximum = counts.max()
            maximum_excess = torch.clamp_min(
                counts - self.neighbor_capacities, 0
            ).max()
            overflow = maximum_excess > 0
            call_step = self.build_calls if step is None else step
            self.minimum_real_edges.copy_(
                torch.minimum(self.minimum_real_edges, real_edges)
            )
            self.maximum_real_edges.copy_(
                torch.maximum(self.maximum_real_edges, real_edges)
            )
            self.maximum_neighbors.copy_(
                torch.maximum(self.maximum_neighbors, maximum)
            )
            self.maximum_neighbors_by_atom.copy_(
                torch.maximum(self.maximum_neighbors_by_atom, counts)
            )
            self.maximum_overflow_required.copy_(
                torch.maximum(self.maximum_overflow_required, maximum)
            )
            self.capacity_misses.add_(overflow.to(dtype=torch.long))
            first = (self.first_overflow_step < 0) & overflow
            self.first_overflow_step.copy_(
                torch.where(first, call_step, self.first_overflow_step)
            )
            self.build_calls.add_(1)
        return self.edge_index, self.edge_shifts

    def stats(self) -> dict[str, Any]:
        calls = int(self.build_calls.item())
        minimum = int(self.minimum_real_edges.item()) if calls else None
        maximum = int(self.maximum_real_edges.item()) if calls else None
        first = int(self.first_overflow_step.item())
        return {
            "fixed_builder_build_calls": calls,
            "fixed_builder_capacity_misses": int(self.capacity_misses.item()),
            "fixed_builder_first_overflow_step": first if first >= 0 else None,
            "fixed_builder_edge_capacity": self.edge_capacity,
            "fixed_builder_neighbors_per_atom": self.neighbors_per_atom,
            "fixed_builder_neighbor_capacities": self.neighbor_capacities.detach()
            .to(device="cpu")
            .tolist(),
            "fixed_builder_capacity_policy": (
                "esen_per_atom_cap"
                if self.neighbor_capacities.unique().numel() > 1
                else "esen_uniform_cap"
            ),
            "fixed_builder_min_real_edges": minimum,
            "fixed_builder_max_real_edges": maximum,
            "fixed_builder_max_padding_fraction": (
                None
                if minimum is None
                else (self.edge_capacity - minimum) / self.edge_capacity
            ),
            "fixed_builder_max_neighbors": int(self.maximum_neighbors.item()),
            "fixed_builder_maximum_neighbors_by_atom": self.maximum_neighbors_by_atom.detach()
            .to(device="cpu")
            .tolist(),
            "fixed_builder_max_overflow_required": int(
                self.maximum_overflow_required.item()
            ),
            "fixed_builder_candidate_universe_size": (
                self.num_atoms * self.candidates_per_atom
            ),
            "fixed_builder_candidates_per_atom": self.candidates_per_atom,
            "fixed_builder_num_pbc_cells": self.num_cells,
            "fixed_builder_pbc_repetitions": list(self.repetitions),
            "fixed_builder_sink_count": self.sink_count,
            "fixed_builder_verlet_skin": self.verlet_skin,
            "fixed_builder_verlet_candidate_capacity": self.verlet_candidate_capacity,
            "fixed_builder_verlet_skin_misses": int(self.skin_misses.item()),
            "fixed_builder_verlet_rebuilds": self.skin_rebuilds,
            "fixed_builder_verlet_enabled": self.skin_candidate_ids is not None,
            "fixed_builder_active_candidate_slots": self.num_atoms
            * (
                int(self.skin_candidate_ids.shape[1])
                if self.skin_candidate_ids is not None
                else self.candidates_per_atom
            ),
            "fixed_builder_candidate_reduction_fraction": (
                0.0
                if self.skin_candidate_ids is None
                else 1.0
                - int(self.skin_candidate_ids.shape[1]) / self.candidates_per_atom
            ),
        }


__all__ = [
    "FixedShapeTACENeighborBuilder",
    "maximum_neighbors_in_graph",
    "neighbor_capacity_from_probe",
]
