import importlib.util

import numpy as np
import pytest

from tace.dataset.neighbour_list import get_neighborhood
from tace.models._e3nn.default import check_model_config


CPU_BACKENDS = ["ase", "matscipy"]
if importlib.util.find_spec("vesin") is not None:
    CPU_BACKENDS.append("vesin")


def test_model_config_rejects_max_neighbors():
    with pytest.raises(ValueError, match="does not truncate neighbor lists"):
        check_model_config({"max_neighbors": 64})


def _edge_set(edge_index):
    return set(zip(edge_index[0].tolist(), edge_index[1].tolist()))


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_nonperiodic_cell_is_preserved(backend):
    positions = np.array([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]])
    lattice = np.eye(3) * 10.0

    edge_index, _, pbc, returned_lattice = get_neighborhood(
        positions,
        cutoff=2.0,
        pbc=(False, False, False),
        lattice=lattice,
        backend=backend,
    )

    assert pbc == (False, False, False)
    np.testing.assert_array_equal(returned_lattice, lattice)
    assert _edge_set(edge_index) == {(0, 1), (1, 0)}


@pytest.mark.parametrize("backend", CPU_BACKENDS)
def test_cellless_nonperiodic_structure(backend):
    positions = np.array([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]])

    edge_index, _, pbc, lattice = get_neighborhood(
        positions,
        cutoff=2.0,
        pbc=False,
        lattice=None,
        backend=backend,
    )

    assert pbc == (False, False, False)
    np.testing.assert_array_equal(lattice, np.zeros((3, 3)))
    assert _edge_set(edge_index) == {(0, 1), (1, 0)}


@pytest.mark.parametrize("backend", ["ase", "matscipy"])
def test_partial_pbc_is_preserved(backend):
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    lattice = np.diag([3.0, 3.0, 0.0])

    _, _, pbc, returned_lattice = get_neighborhood(
        positions,
        cutoff=1.5,
        pbc=(True, True, False),
        lattice=lattice,
        backend=backend,
    )

    assert pbc == (True, True, False)
    np.testing.assert_array_equal(returned_lattice, lattice)


def test_vesin_rejects_partial_pbc():
    pytest.importorskip("vesin")

    with pytest.raises(ValueError, match="vesin only support"):
        get_neighborhood(
            np.array([[0.0, 0.0, 0.0]]),
            cutoff=1.5,
            pbc=(True, True, False),
            lattice=np.eye(3),
            backend="vesin",
        )


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

    full_edges, _, _, _ = get_neighborhood(
        positions,
        cutoff=2.0,
        pbc=False,
        lattice=None,
        max_neighbors=None,
        backend=backend,
    )
    compatibility_edges, _, _, _ = get_neighborhood(
        positions,
        cutoff=2.0,
        pbc=False,
        lattice=None,
        max_neighbors=1,
        backend=backend,
    )

    assert full_edges.shape[1] == 12
    assert _edge_set(compatibility_edges) == _edge_set(full_edges)


@pytest.mark.parametrize(
    ("pbc", "lattice"),
    [
        (False, None),
        ((True, True, False), np.diag([3.0, 3.0, 0.0])),
    ],
)
def test_alchemiops_matches_matscipy(tmp_path, monkeypatch, pbc, lattice):
    pytest.importorskip("nvalchemiops")
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "warp"))
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.75, 0.0, 0.0], [0.0, 0.75, 0.0]]
    )

    reference, _, _, _ = get_neighborhood(
        positions,
        cutoff=2.0,
        pbc=pbc,
        lattice=lattice,
        backend="matscipy",
    )
    edge_index, _, returned_pbc, returned_lattice = get_neighborhood(
        positions,
        cutoff=2.0,
        pbc=pbc,
        lattice=lattice,
        max_neighbors=1,
        backend="alchemiops",
    )

    expected_pbc = (False, False, False) if pbc is False else pbc
    expected_lattice = np.zeros((3, 3)) if lattice is None else lattice
    assert returned_pbc == expected_pbc
    np.testing.assert_array_equal(returned_lattice, expected_lattice)
    assert _edge_set(edge_index) == _edge_set(reference)
