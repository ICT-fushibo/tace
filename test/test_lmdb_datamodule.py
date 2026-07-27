import pickle

import numpy as np
import pytest
from ase import Atoms
from torch_geometric.loader import DataLoader

from tace.dataset.datamodule import create_graphs
from tace.dataset.element import build_element_lookup
from tace.dataset.quantity import KEYS, KeySpecification, update_keyspec_from_kwargs


def _dataset_config():
    keyspec = KeySpecification()
    update_keyspec_from_kwargs(keyspec, KEYS)
    return {
        "cutoff": 3.0,
        "max_neighbors": None,
        "keyspec": keyspec,
        "target_property": ["energy", "forces"],
        "embedding_property": [],
        "neighborlist_backend": "matscipy",
    }


def _atoms(shift):
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.75 + shift, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
    )
    atoms.info["energy"] = -1.0 + shift
    atoms.arrays["forces"] = np.zeros((2, 3))
    return atoms


def test_lmdb_cache_is_reusable_and_worker_safe(tmp_path):
    kwargs = {
        "element": build_element_lookup([1]),
        "for_dataset": _dataset_config(),
        "stage": "train",
        "shard_dirs": [tmp_path],
        "storage_mode": "lmdb",
        "shard_size": 1,
        "avg_graph_size_in_KB": 1,
        "cache_size": 2,
    }
    dataset = create_graphs(atoms_list=[_atoms(0.0), _atoms(0.1)], **kwargs)
    assert len(dataset) == 2
    assert dataset[0].energy.item() == -1.0
    assert len(list(tmp_path.glob("train_shard*.lmdb"))) == 2

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored[1].energy.item() == pytest.approx(-0.9)
    dataset.close()
    restored.close()

    cached = create_graphs(atoms_list=None, **kwargs)
    batches = list(
        DataLoader(
            cached,
            batch_size=1,
            num_workers=2,
            multiprocessing_context="spawn",
        )
    )
    assert len(batches) == 2


def test_existing_lmdb_cache_does_not_add_distributed_barrier(
    tmp_path, monkeypatch
):
    common = {
        "element": build_element_lookup([1]),
        "for_dataset": _dataset_config(),
        "stage": "train",
        "shard_dirs": [tmp_path],
        "storage_mode": "lmdb",
    }
    original = create_graphs(atoms_list=[_atoms(0.0)], **common)
    original.close()

    monkeypatch.setattr(
        "tace.dataset.datamodule.dist.is_initialized", lambda: True
    )
    monkeypatch.setattr(
        "tace.dataset.datamodule.dist.get_rank", lambda: 1
    )
    monkeypatch.setattr(
        "tace.dataset.datamodule.dist.get_world_size", lambda: 2
    )

    def fail_if_called():
        raise AssertionError("Complete cache loading called dist.barrier()")

    monkeypatch.setattr(
        "tace.dataset.datamodule.dist.barrier", fail_if_called
    )
    cached = create_graphs(atoms_list=None, **common)
    assert len(cached) == 1
