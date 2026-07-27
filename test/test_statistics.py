import math

import torch
from ase import Atoms
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tace.dataset.element import build_element_lookup
from tace.dataset.quantity import KEYS, KeySpecification, update_keyspec_from_kwargs
from tace.dataset.statistics import (
    _compute_statistics,
    _finite_scale,
    compute_atomic_energy,
)
from tace.models.blocks import OneHotToAtomicEnergy, ScaleShift


def _single_graph(fidelity_idx=0):
    return Data(
        node_attrs=torch.ones(2, 1),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
        energy=torch.tensor([2.0]),
        forces=torch.zeros(2, 3),
        fidelity_idx=torch.tensor(fidelity_idx),
    )


def test_empty_fidelity_uses_neutral_statistics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    statistics = _compute_statistics(
        DataLoader([_single_graph()], batch_size=1),
        atomic_numbers=[1],
        atomic_energies=[{1: 0.0}, {1: 0.0}],
        target_property=["energy", "forces"],
        num_fidelities=2,
    )

    assert statistics[0]["__std_energy"] == 0.0
    assert math.isfinite(statistics[0]["__std_energy"])
    assert statistics[0]["rms_forces"] == {1: 0.0}

    empty = statistics[1]
    assert empty["available"] is False
    assert empty["num_graphs"] == 0
    assert empty["atomic_energy"] == {1: 0.0}
    assert empty["rms_forces"] == {1: 0.0}
    assert empty["mean_delta_energy_per_atom"] == {1: 0.0}

    scale_shift = ScaleShift.build_from_config(
        statistics,
        {
            "scale_type": "std_energy",
            "shift_type": "mean_energy",
            "scale_trainable": False,
            "shift_trainable": False,
            "all_atoms": True,
        },
        atomic_numbers=[1],
    )
    assert scale_shift.scale[1].item() == 1.0
    assert scale_shift.shift[1].item() == 0.0


def test_missing_fidelity_label_belongs_only_to_head_zero():
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.info["energy"] = -1.5
    element = build_element_lookup([1])
    keyspec = KeySpecification()
    update_keyspec_from_kwargs(keyspec, KEYS)

    assert compute_atomic_energy([atoms], element, keyspec, 0) == {1: -1.5}
    assert compute_atomic_energy([atoms], element, keyspec, 1) == {1: 0.0}


def test_atomic_energy_mapping_uses_atomic_number_order():
    layer = OneHotToAtomicEnergy([{8: -8.0, 1: -1.0}], [1, 8])
    values = layer(torch.eye(2))
    torch.testing.assert_close(values, torch.tensor([[-1.0], [-8.0]]))


def test_scale_shift_preserves_finite_nonpositive_scale():
    layer = ScaleShift.build_from_config(
        [
            {
                "atomic_numbers": [1, 8],
                "custom_scale": {1: -2.0, 8: 0.0},
            }
        ],
        {
            "scale_type": "custom_scale",
            "shift_type": None,
            "scale_trainable": False,
            "shift_trainable": False,
            "all_atoms": True,
        },
        atomic_numbers=[1, 8],
    )
    torch.testing.assert_close(layer.scale, torch.tensor([[-2.0, 0.0]]))


def test_statistics_scale_only_replaces_nonfinite_values():
    assert _finite_scale(-2.0) == -2.0
    assert _finite_scale(0.0) == 0.0
    assert _finite_scale(1.0e-30) == 1.0e-30
    assert _finite_scale(float("nan")) == 1.0
    assert _finite_scale(float("inf")) == 1.0
    assert _finite_scale(float("-inf")) == 1.0


def test_scale_shift_uses_explicit_atomic_number_order():
    layer = ScaleShift.build_from_config(
        [
            {
                "atomic_numbers": [1, 8],
                "custom_scale": {1: 1.0, 8: 8.0},
                "custom_shift": {1: -1.0, 8: -8.0},
            }
        ],
        {
            "scale_type": "custom_scale",
            "shift_type": "custom_shift",
            "scale_trainable": False,
            "shift_trainable": False,
            "all_atoms": True,
        },
        atomic_numbers=[8, 1],
    )
    torch.testing.assert_close(layer.atomic_numbers, torch.tensor([8, 1]))
    torch.testing.assert_close(layer.scale, torch.tensor([[8.0, 1.0]]))
    torch.testing.assert_close(layer.shift, torch.tensor([[-8.0, -1.0]]))


def test_missing_energy_labels_do_not_change_statistics(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    labeled = _single_graph()
    labeled.energy_weight = torch.tensor([1.0])
    missing = _single_graph()
    missing.energy = torch.tensor([100.0])
    missing.energy_weight = torch.tensor([0.0])

    statistics = _compute_statistics(
        DataLoader([labeled, missing], batch_size=2),
        atomic_numbers=[1],
        atomic_energies=[{1: 0.0}],
        target_property=["energy"],
    )
    assert statistics[0]["__mean_energy"] == 2.0
    assert statistics[0]["__std_energy"] == 0.0


def test_invalid_fidelity_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loader = DataLoader([_single_graph(fidelity_idx=2)], batch_size=1)
    try:
        _compute_statistics(
            loader,
            atomic_numbers=[1],
            atomic_energies=[{1: 0.0}, {1: 0.0}],
            target_property=["energy", "forces"],
            num_fidelities=2,
        )
    except ValueError as error:
        assert "fidelity_idx values must be in [0, 1]" in str(error)
    else:
        raise AssertionError("Out-of-range fidelity_idx was accepted")
