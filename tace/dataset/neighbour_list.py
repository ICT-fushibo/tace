################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional, Tuple

from ase.geometry import complete_cell

import numpy as np
from matscipy.neighbours import neighbour_list
try:
    from ase.neighborlist import primitive_neighbor_list
    from vesin import NeighborList as vesin_nl
except ImportError:
    pass


# self-interaction: ase
# 1D, 2D: ase, matscipy
NL_BACKEND = ["ase", "vesin", "matscipy"]


def filter_max_neighbors(source, target, shifts, distances, max_neighbors="inf"):

    if max_neighbors is None or max_neighbors == "inf":
        return source, target, shifts
    order = np.lexsort((distances, source))
    src_sorted = source[order]
    dst_sorted = target[order]
    shifts_sorted = shifts[order]

    unique_src, counts = np.unique(src_sorted, return_counts=True)
    cum_counts = np.cumsum(counts)  # [3, 2, 1] => [3, 5, 6]

    mask = np.zeros(len(src_sorted), dtype=bool)
    start_idx = 0
    for end_idx in cum_counts:
        count = end_idx - start_idx
        keep = min(max_neighbors, count)
        mask[start_idx : start_idx + keep] = True
        start_idx = end_idx

    return (
        src_sorted[mask],
        dst_sorted[mask],
        shifts_sorted[mask],
    )

def get_neighborhood(
    positions: np.ndarray,
    cutoff: float,
    pbc: Optional[bool | Tuple[bool, bool, bool]] = None,
    lattice: Optional[np.ndarray] = None,  # [3, 3]
    max_neighbors: Optional[int] = None,
    backend: str = "vesin" # "matscipy",
) -> Tuple[np.ndarray, np.ndarray]:
    
    assert backend in NL_BACKEND, f"Neighborlist backend should be in {NL_BACKEND}"

    # === PBC ===
    if pbc is None:
        pbc = (False, False, False)
    elif isinstance(pbc, bool):
        pbc = (pbc,) * 3
    else:
        pbc = tuple(bool(i) for i in pbc)

    # === Lattiace ===
    if any(pbc):
        if lattice is None or np.allclose(lattice, 0.0):
            raise ValueError(
                "At least one direction is periodic, but lattice is None or zero."
            )
    if not any(pbc):
        lattice = None
    else:
        if lattice is None:
            raise ValueError(
                "At least one direction is periodic, but lattice is None."
            )
        
    # === Neighborlist ===
    if backend == "matscipy":
        edges = neighbour_list(
            quantities="ijSd",
            pbc=pbc,
            cell=lattice,
            positions=positions,
            cutoff=cutoff,
        )
    elif backend == "vesin":
        # https://github.com/Luthaf/vesin/blob/main/python/vesin/src/vesin/_ase.py
        if all(pbc):
            is_3D = True
        elif not any(pbc):
            is_3D = False
        else:
            raise ValueError("vesin only support pbc=(F, F, F) or (T, T, T)")
        if lattice is None:
            box = np.zeros((3, 3), dtype=positions.dtype)
        else:
            box = lattice
        edges = vesin_nl(
            cutoff=cutoff, 
            full_list=True
        ).compute(points=positions, box=box, periodic=is_3D, quantities="ijSd")
        edges = list(edges)
        edges[0] = edges[0].astype(np.int64)
        edges[1] = edges[1].astype(np.int64)
        edges = tuple(edges)
    elif backend == "ase":
        if lattice is None:
            cell = np.zeros((3, 3), dtype=positions.dtype)
        else:
            cell = lattice
        edges = primitive_neighbor_list(
            "ijSd",
            pbc,
            cell,
            positions,
            cutoff=cutoff,
            self_interaction=False,
            use_scaled_positions=False,
        )

    source, target, shifts = filter_max_neighbors(*edges, max_neighbors=max_neighbors)

    real_self_loop = source == target
    real_self_loop &= np.all(shifts == 0, axis=1)
    keep_edge = ~real_self_loop

    source = source[keep_edge]
    target = target[keep_edge]

    edge_shifts = shifts[keep_edge]
    edge_index = np.stack((source, target))

    if lattice is None:
        lattice = np.zeros((3, 3), dtype=positions.dtype)
 
    return edge_index, edge_shifts, pbc, lattice


def get_neighborhood_for_calculator(
    positions: np.ndarray,
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    lattice: Optional[np.ndarray] = None,  # [3, 3]
    max_neighbors: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    edges = neighbour_list(
        quantities="ijSd",
        pbc=pbc,
        cell=lattice,
        positions=positions,
        cutoff=cutoff,
    )
    source, target, shifts = filter_max_neighbors(*edges, max_neighbors=max_neighbors)

    real_self_loop = source == target
    real_self_loop &= np.all(shifts == 0, axis=1)
    keep_edge = ~real_self_loop

    source = source[keep_edge]
    target = target[keep_edge]

    edge_shifts = shifts[keep_edge]
    edge_index = np.stack((source, target))

    return edge_index, edge_shifts



