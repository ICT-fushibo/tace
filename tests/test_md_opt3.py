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
from tace.md_stages.fixed_neighbor import (
    FixedShapeTACENeighborBuilder,
    neighbor_capacity_from_probe,
)
from tace.md_stages.opt3 import (
    TACEWholeStepGraph,
    TACEWholeStepPotential,
    _integrate_nhc_pure,
    _slots_from_total_edge_capacity,
    _validate_request,
)


def _request(*, backend: str = "whole-step-cuda-graph") -> MDRunRequest:
    return MDRunRequest(
        model="tace",
        stage="opt3",
        model_path="TACE-OAM-L.pt",
        atoms=Atoms(
            "H2", positions=[[0, 0, 0], [0, 0, 0.7]], cell=[5, 5, 5], pbc=True
        ),
        config=MDConfig(
            device="cuda:0",
            dtype="float64",
            steps=1,
            observation_steps=(0, 1),
            integrator="nose_hoover_chain",
        ),
        backend=backend,
        options={"model_dtype": "checkpoint"},
    )


def test_opt3_contract_accepts_matbench_nhc_and_rejects_wrong_backend():
    _validate_request(_request())
    with pytest.raises(ValueError, match="whole-step-cuda-graph"):
        _validate_request(_request(backend="gpu-resident"))


def test_public_route_dispatches_opt3_without_touching_baseline(monkeypatch):
    request = _request()
    sentinel = object()

    def fake_optimized_stage(actual_request, *, module_prefix):
        assert actual_request is request
        assert module_prefix == "tace.md_stages"
        return sentinel

    monkeypatch.setattr(md_route, "run_optimized_stage", fake_optimized_stage)
    assert md_route.run_md(request) is sentinel


def test_esen_capacity_policy_rounds_with_headroom():
    assert neighbor_capacity_from_probe(16, margin=0.10, slot_step=8) == 24
    assert neighbor_capacity_from_probe(80, margin=0.10, slot_step=8) == 88
    with pytest.raises(ValueError, match="positive"):
        neighbor_capacity_from_probe(0)


def test_guarded_total_edge_capacity_is_aligned_without_second_guard():
    assert _slots_from_total_edge_capacity(
        8960, 108, slot_step=8, guard_slots=0
    ) == 88
    assert _slots_from_total_edge_capacity(
        256, 32, slot_step=8, guard_slots=0
    ) == 8


def test_fixed_builder_distributes_padding_without_dummy_nodes():
    cell = torch.eye(3, dtype=torch.float64) * 5.0
    builder = FixedShapeTACENeighborBuilder(
        num_atoms=2,
        cell=cell,
        pbc=torch.ones(3, dtype=torch.bool),
        cutoff=1.0,
        neighbors_per_atom=3,
        sink_count=2,
    )
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64
    )
    edge_index, shifts = builder.build(positions)
    vectors = (
        positions[edge_index[1]]
        - positions[edge_index[0]]
        + shifts @ cell
    )
    lengths = torch.linalg.vector_norm(vectors, dim=1)
    real = lengths <= 1.0
    assert int(real.sum()) == 2
    torch.testing.assert_close(
        lengths[real], torch.full((2,), 0.7, dtype=torch.float64)
    )
    assert bool((lengths[~real] > 1.0).all())
    assert int(edge_index.max()) < 2
    padding_nodes = edge_index[0, ~real]
    assert set(padding_nodes.tolist()) == {0, 1}


def test_fixed_builder_records_per_centre_overflow_on_device():
    builder = FixedShapeTACENeighborBuilder(
        num_atoms=3,
        cell=torch.eye(3, dtype=torch.float64) * 5.0,
        pbc=torch.ones(3, dtype=torch.bool),
        cutoff=1.0,
        neighbors_per_atom=1,
        sink_count=2,
    )
    builder.build(
        torch.tensor(
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        step=torch.tensor(7),
    )
    stats = builder.stats()
    assert stats["fixed_builder_capacity_misses"] == 1
    assert stats["fixed_builder_first_overflow_step"] == 7
    assert stats["fixed_builder_max_overflow_required"] == 2


def test_whole_step_hot_path_has_no_host_transfer_or_eager_fallback():
    sources = "\n".join(
        textwrap.dedent(source)
        for source in (
            inspect.getsource(TACEWholeStepGraph._graph_body),
            inspect.getsource(TACEWholeStepGraph._step_values),
            inspect.getsource(_integrate_nhc_pure),
            inspect.getsource(TACEWholeStepPotential.evaluate),
            inspect.getsource(FixedShapeTACENeighborBuilder.build),
        )
    )
    tree = ast.parse(sources)
    attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert attributes.isdisjoint({"cpu", "numpy", "item", "tolist", "data_ptr"})
    assert "_integrate_nhc_pure" in sources
    assert "builder.build" in sources
    assert "self.model(self.static_data)" in sources


def test_capture_is_one_whole_step_graph_and_has_no_fallback_branch():
    source = inspect.getsource(TACEWholeStepGraph.capture)
    assert "torch.cuda.graph" in source
    assert "self._graph_body()" in source
    assert "fallback" not in source.lower() or "forbidden" in source.lower()
    assert "_validate_graph_step" in source


def test_initial_force_is_a_measured_replay_before_physical_steps():
    source = inspect.getsource(TACEWholeStepGraph.evaluate_initial)
    assert "self.step()" in source
    assert "self.advance.zero_()" in source
    assert "self.advance.fill_(1.0)" in source


def test_opt3_module_does_not_enable_model_specific_acceleration():
    source = inspect.getsource(TACEWholeStepPotential.__init__)
    for accelerator in (
        "enable_oeq=False",
        "enable_cue=False",
        "enable_eqt=False",
        "enable_compile=False",
        "enable_triton=False",
    ):
        assert accelerator in source


def test_full_periodicity_is_required():
    request = _request()
    request.atoms.pbc = np.array([True, True, False])
    with pytest.raises(ValueError, match="full periodic"):
        _validate_request(request)
