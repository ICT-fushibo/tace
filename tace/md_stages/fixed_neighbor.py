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
        self.neighbors_per_atom = int(neighbors_per_atom)
        self.sink_count = min(int(sink_count), self.num_atoms)
        self.device = cell.device
        self.position_dtype = cell.dtype
        self.cell = cell.detach().reshape(3, 3)
        self.repetitions = _pbc_repetitions(cell, cutoff, pbc)

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
        self.edge_capacity = self.num_atoms * self.neighbors_per_atom
        self.candidate_sources = torch.arange(
            self.num_atoms, device=self.device, dtype=torch.long
        ).repeat_interleave(self.num_cells)
        self.candidate_cell_offsets = self.unit_cell_offsets.repeat(
            self.num_atoms, 1
        )
        self.candidate_ids = torch.arange(
            self.candidates_per_atom, device=self.device, dtype=torch.long
        ).view(1, -1)
        self.slot_centres = torch.arange(
            self.num_atoms, device=self.device, dtype=torch.long
        ).repeat_interleave(self.neighbors_per_atom)

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

    def reset_stats(self) -> None:
        self.build_calls.zero_()
        self.capacity_misses.zero_()
        self.first_overflow_step.fill_(-1)
        self.minimum_real_edges.fill_(self.edge_capacity)
        self.maximum_real_edges.zero_()
        self.maximum_neighbors.zero_()
        self.maximum_overflow_required.zero_()

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
            shifted_sources = (
                positions.index_select(0, self.candidate_sources)
                + torch.mm(
                    self.candidate_cell_offsets.to(dtype=positions.dtype),
                    self.cell.to(dtype=positions.dtype),
                )
            )
            delta = shifted_sources.unsqueeze(0) - positions.unsqueeze(1)
            distance_squared = delta.square().sum(dim=-1)
            valid = (distance_squared <= self.cutoff * self.cutoff) & (
                distance_squared > 1.0e-8
            )
            counts = valid.sum(dim=1)
            candidate_order = torch.where(
                valid,
                self.candidate_ids.expand(self.num_atoms, -1),
                torch.full_like(
                    self.candidate_ids.expand(self.num_atoms, -1),
                    self.candidates_per_atom,
                ),
            )
            selected = torch.topk(
                candidate_order,
                k=self.neighbors_per_atom,
                dim=1,
                largest=False,
                sorted=True,
            ).values.reshape(-1)
            selected_valid = selected < self.candidates_per_atom
            safe_selected = selected.clamp_max(self.candidates_per_atom - 1)
            sources = self.candidate_sources.index_select(0, safe_selected)
            offsets = self.candidate_cell_offsets.index_select(0, safe_selected)
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
            overflow = maximum > self.neighbors_per_atom
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
            "fixed_builder_capacity_policy": "esen_uniform_cap",
            "fixed_builder_min_real_edges": minimum,
            "fixed_builder_max_real_edges": maximum,
            "fixed_builder_max_padding_fraction": (
                None
                if minimum is None
                else (self.edge_capacity - minimum) / self.edge_capacity
            ),
            "fixed_builder_max_neighbors": int(self.maximum_neighbors.item()),
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
        }


__all__ = [
    "FixedShapeTACENeighborBuilder",
    "maximum_neighbors_in_graph",
    "neighbor_capacity_from_probe",
]
