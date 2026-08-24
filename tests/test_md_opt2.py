from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest
import torch
from ase import Atoms
from md_benchmark.md_route import MDConfig, MDRunRequest

from tace import md_route
from tace.md_stages.opt2 import (
    FixedEdgeBuffers,
    TACEModelOnlyGraphEvaluator,
    _capture_model_graph,
    _edge_capacity,
    _enable_padding_density_masks_,
    _NeutralPaddingRadialBasis,
    _validate_request,
)


def _request(
    *,
    backend: str = "model-only-cuda-graph",
    collect_trajectory: bool = False,
    options: dict | None = None,
) -> MDRunRequest:
    atoms = Atoms(
        "H2", positions=[[0, 0, 0], [0, 0, 0.7]], cell=[5, 5, 5], pbc=True
    )
    return MDRunRequest(
        model="tace",
        stage="opt2",
        model_path="TACE-OAM-L.pt",
        atoms=atoms,
        config=MDConfig(
            device="cuda:0",
            dtype="float64",
            steps=1,
            observation_steps=(1,),
            collect_trajectory=collect_trajectory,
            record_interval=1 if collect_trajectory else 0,
        ),
        backend=backend,
        options={"model_dtype": "checkpoint", **(options or {})},
    )


def test_opt2_contract_requires_exact_backend_and_energy_force_only():
    _validate_request(_request())
    with pytest.raises(ValueError, match="model-only-cuda-graph"):
        _validate_request(_request(backend="gpu-resident"))
    with pytest.raises(NotImplementedError, match="trajectory stress"):
        _validate_request(_request(collect_trajectory=True))
    with pytest.raises(ValueError, match="does not compute stress"):
        _validate_request(_request(options={"compute_stress": True}))


def test_model_route_dispatches_opt2_without_touching_baseline(monkeypatch):
    request = _request()
    sentinel = object()

    def fake_optimized_stage(actual_request, *, module_prefix):
        assert actual_request is request
        assert module_prefix == "tace.md_stages"
        return sentinel

    monkeypatch.setattr(md_route, "run_optimized_stage", fake_optimized_stage)
    assert md_route.run_md(request) is sentinel


def test_edge_capacity_validation_and_rounding():
    assert _edge_capacity(100, {}) == 132
    assert _edge_capacity(100, {"edge_capacity": 150}) == 150
    with pytest.raises(ValueError, match="smaller than the initial"):
        _edge_capacity(100, {"edge_capacity": 99})
    with pytest.raises(ValueError, match=">= 1"):
        _edge_capacity(100, {"edge_capacity_multiplier": 0.9})


def test_fixed_edge_buffers_pad_shrink_and_fail_on_overflow():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.int64)
    edge_shifts = torch.zeros(3, 3, dtype=torch.float32)
    dummy_shift = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32)
    buffers = FixedEdgeBuffers.allocate(
        capacity=4,
        edge_index=edge_index,
        edge_shifts=edge_shifts,
        dummy_shift=dummy_shift,
    )
    assert buffers.active_edges == 3
    torch.testing.assert_close(buffers.edge_shifts[3], dummy_shift)

    buffers.update(edge_index[:, :1], edge_shifts[:1])
    assert buffers.active_edges == 1
    torch.testing.assert_close(
        buffers.edge_shifts[1:3], dummy_shift.reshape(1, 3).expand(2, 3)
    )
    assert torch.count_nonzero(buffers.edge_index[:, 1:3]) == 0

    with pytest.raises(RuntimeError, match="capacity overflow"):
        buffers.update(
            torch.zeros(2, 5, dtype=torch.int64),
            torch.zeros(5, 3, dtype=torch.float32),
        )


def test_evaluator_hot_path_has_no_host_transfer_or_eager_fallback():
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(TACEModelOnlyGraphEvaluator.__call__))
    )
    attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert attributes.isdisjoint({"cpu", "numpy", "item", "tolist", "to"})
    source = inspect.getsource(TACEModelOnlyGraphEvaluator.__call__)
    assert "graph.replay()" in source
    assert "self.model(" not in source


def test_runtime_counter_reset_excludes_capture_and_md_warmup_replays():
    evaluator = object.__new__(TACEModelOnlyGraphEvaluator)
    evaluator.production_replays = 9
    evaluator.initial_edges = 12
    evaluator.max_observed_edges = 18
    evaluator.reset_runtime_counters()
    assert evaluator.production_replays == 0
    assert evaluator.max_observed_edges == evaluator.initial_edges


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_graph_helper_captures_conservative_force_replay():
    device = torch.device("cuda:0")
    positions = torch.tensor(
        [[1.0, -2.0, 0.5]], device=device, dtype=torch.float32, requires_grad=True
    )

    class HarmonicModel:
        def __call__(self, data):
            energy = data["positions"].square().sum().reshape(1)
            forces = -torch.autograd.grad(energy.sum(), data["positions"])[0]
            return {"energy": energy, "forces": forces}

    graph, outputs = _capture_model_graph(
        HarmonicModel(), {"positions": positions}, device=device, warmup_steps=3
    )
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(outputs["forces"], -2.0 * positions)

    with torch.no_grad():
        positions.copy_(torch.tensor([[3.0, 1.0, -1.0]], device=device))
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(outputs["energy"], positions.square().sum().reshape(1))
    torch.testing.assert_close(outputs["forces"], -2.0 * positions)


def test_dummy_padding_distance_is_beyond_cutoff():
    from tace.md_stages.opt2 import _dummy_shift

    cell = np.diag([5.0, 6.0, 7.0])
    shift = _dummy_shift(cell, cutoff=6.0)
    assert np.linalg.norm(shift @ cell) > 6.0


def test_neutral_padding_radial_basis_preserves_real_and_masks_far_edges():
    class AppliedCutoffRadial(torch.nn.Module):
        def forward(self, edge_length, *_args):
            radial = torch.where(
                edge_length < 6.0,
                torch.full_like(edge_length, 2.0),
                torch.zeros_like(edge_length),
            )
            return radial, None

    wrapped = _NeutralPaddingRadialBasis(AppliedCutoffRadial(), cutoff=6.0)
    lengths = torch.tensor([[2.0], [12.0]])
    radial, valid = wrapped(
        lengths,
        torch.ones(1, 1),
        torch.zeros(2, 2, dtype=torch.long),
        torch.ones(1, dtype=torch.long),
        None,
    )
    torch.testing.assert_close(radial, torch.tensor([[2.0], [0.0]]))
    torch.testing.assert_close(valid, torch.tensor([[1.0], [0.0]]))


def test_padding_mask_is_enabled_for_density_normalization_modules():
    class Interaction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.edge_density = torch.nn.Linear(2, 1)
            self.apply_density_cutoff = False

    model = torch.nn.Sequential(Interaction(), torch.nn.Linear(1, 1))
    assert _enable_padding_density_masks_(model) == 1
    assert model[0].apply_density_cutoff is True
    assert _enable_padding_density_masks_(model) == 0
