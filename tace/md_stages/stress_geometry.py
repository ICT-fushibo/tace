"""Opt4 fixed-cell local-potential strain in relative periodic coordinates."""

import torch


def strained_edge_vectors(positions, lattice, edge_index, edge_shifts, batch, strain):
    """Strain the complete periodic bond, not separate large absolute terms.

    dE/dstrain = sym(sum_e r_e outer dE/dr_e), including shift @ cell.
    This is the same joint position/strain derivative as the native affine
    path, but avoids separately accumulating origin-dependent position and
    cell virials before their cancellation. All operations retain model dtype.
    The zero strain leaf is allocated once by the Opt4 potential.
    """
    source, target = edge_index
    edge_batch = batch[source]
    vectors = positions[target] - positions[source] + torch.einsum(
        "ni,nij->nj", edge_shifts, lattice[edge_batch]
    )
    symmetric = 0.5 * (strain + strain.transpose(-1, -2))
    return vectors + torch.einsum("ni,nij->nj", vectors, symmetric[edge_batch])
