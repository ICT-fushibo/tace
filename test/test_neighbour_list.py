import importlib.util
from itertools import product

import numpy as np
import pytest

from tace.dataset.neighbour_list import get_neighborhood
from tace.models._e3nn.default import check_model_config


BACKENDS = ["ase", "matscipy", "vesin", "alchemiops"]
CPU_BACKENDS = ["ase", "matscipy"]
if importlib.util.find_spec("vesin") is not None:
    CPU_BACKENDS.append("vesin")

NEIGHBOR_CASES = [
    pytest.param(
        np.array([[0.0, 0.0, 0.0], [0.4, 0.1, 0.0], [2.0, 0.0, 0.0]]),
        0.75,
        False,
        None,
        id="nonperiodic-no-cell",
    ),
    pytest.param(
        np.array(
            [[0.1, 1.9, 2.9], [0.4, 2.1, 3.1], [3.9, 1.9, 2.9]]
        ),
        0.5,
        False,
        np.diag([4.0, 4.0, 4.0]),
        id="nonperiodic-with-cell",
    ),
    pytest.param(
        np.array([[0.1, 0.9, 1.9], [3.9, 1.1, 2.1]]),
        0.5,
        (True, False, False),
        np.diag([4.0, 0.0, 0.0]),
        id="1d-periodic",
    ),
    pytest.param(
        np.array(
            [[0.1, 0.1, 0.9], [3.9, 0.1, 1.1], [0.1, 3.9, 1.0]]
        ),
        0.5,
        (True, True, False),
        np.diag([4.0, 4.0, 0.0]),
        id="2d-periodic",
    ),
    pytest.param(
        np.array(
            [
                [0.1, 0.1, 0.1],
                [3.9, 0.1, 0.1],
                [0.1, 3.9, 0.1],
                [0.1, 0.1, 3.9],
            ]
        ),
        0.5,
        (True, True, True),
        np.diag([4.0, 4.0, 4.0]),
        id="3d-periodic",
    ),
]


@pytest.fixture(scope="session")
def alchemiops_backend(tmp_path_factory):
    warp = pytest.importorskip("warp")
    warp.config.kernel_cache_dir = str(tmp_path_factory.mktemp("warp-cache"))
    pytest.importorskip("nvalchemiops")


def _canonical_edges(edge_index, shifts):
    return {
        (int(source), int(target), tuple(int(value) for value in shift))
        for source, target, shift in zip(edge_index[0], edge_index[1], shifts)
    }


def _reference_edges(positions, cutoff, pbc, lattice):
    if isinstance(pbc, bool):
        pbc = (pbc,) * 3
    physical_lattice = (
        np.zeros((3, 3), dtype=positions.dtype) if lattice is None else lattice
    )
    shift_ranges = [(-1, 0, 1) if periodic else (0,) for periodic in pbc]
    expected = set()

    for source, target in product(range(len(positions)), repeat=2):
        for shift in product(*shift_ranges):
            if source == target and shift == (0, 0, 0):
                continue
            displacement = (
                positions[target]
                - positions[source]
                + np.asarray(shift) @ physical_lattice
            )
            if np.linalg.norm(displacement) < cutoff:
                expected.add((source, target, shift))

    return expected


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("positions", "cutoff", "pbc", "lattice"), NEIGHBOR_CASES)
def test_backend_geometry_matrix(backend, positions, cutoff, pbc, lattice, request):
    if backend == "vesin":
        pytest.importorskip("vesin")
    elif backend == "alchemiops":
        request.getfixturevalue("alchemiops_backend")

    edge_index, shifts, returned_pbc, returned_lattice = get_neighborhood(
        positions,
        cutoff=cutoff,
        pbc=pbc,
        lattice=lattice,
        backend=backend,
    )

    expected_pbc = (pbc,) * 3 if isinstance(pbc, bool) else pbc
    expected_lattice = (
        np.zeros((3, 3), dtype=positions.dtype) if lattice is None else lattice
    )
    assert returned_pbc == expected_pbc
    np.testing.assert_array_equal(returned_lattice, expected_lattice)
    assert np.all(shifts[:, np.logical_not(expected_pbc)] == 0)
    assert _canonical_edges(edge_index, shifts) == _reference_edges(
        positions, cutoff, expected_pbc, expected_lattice
    )


def test_model_config_rejects_max_neighbors():
    with pytest.raises(ValueError, match="does not truncate neighbor lists"):
        check_model_config({"max_neighbors": 64})


def test_periodic_structure_requires_cell():
    with pytest.raises(ValueError, match="lattice is None or zero"):
        get_neighborhood(
            np.array([[0.0, 0.0, 0.0]]),
            cutoff=1.5,
            pbc=True,
            lattice=None,
            backend="matscipy",
        )


def test_legacy_nvidia_backend_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown neighborlist backend"):
        get_neighborhood(
            np.array([[0.0, 0.0, 0.0]]),
            cutoff=1.5,
            backend="nvidia",
        )


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_max_neighbors_is_retained_but_not_applied(backend):
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )

    full_edges, full_shifts, _, _ = get_neighborhood(
        positions,
        cutoff=2.0,
        pbc=False,
        lattice=None,
        max_neighbors=None,
        backend=backend,
    )
    compatibility_edges, compatibility_shifts, _, _ = get_neighborhood(
        positions,
        cutoff=2.0,
        pbc=False,
        lattice=None,
        max_neighbors=1,
        backend=backend,
    )

    assert full_edges.shape[1] == 12
    assert _canonical_edges(compatibility_edges, compatibility_shifts) == (
        _canonical_edges(full_edges, full_shifts)
    )
