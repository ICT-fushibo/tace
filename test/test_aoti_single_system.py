import json
import zipfile

import torch

from tace.models.compile import aot
from tace.models.compile.aot import (
    AOTICompiledTensorModel,
    TACE_AOTI_CUSTOM_OPS_LIBS_ENTRY,
    _custom_ops_libs_from_model,
    _embed_custom_ops_libs,
    _import_custom_ops_libs,
    _size_oblivious_export,
)
from tace.models.compile.compile import trace_to_fx


class FakeCompiledModel:
    def __init__(self):
        self.inputs = None

    def __call__(self, *inputs):
        self.inputs = inputs
        num_nodes = inputs[0].shape[0]
        num_graphs = inputs[4].shape[0]
        return (
            torch.arange(num_graphs, dtype=torch.float32),
            torch.arange(num_nodes, dtype=torch.float32),
            torch.ones((num_nodes, 3)),
            torch.eye(3).expand(num_graphs, -1, -1).clone(),
        )


def test_dynamic_aoti_package_accepts_single_system():
    compiled_model = FakeCompiledModel()
    metadata = {
        "tace_input_keys": json.dumps(
            [
                "positions",
                "node_attrs",
                "edge_index",
                "edge_shifts",
                "lattice",
                "batch",
                "ptr",
                "fidelity_idx",
            ]
        ),
        "tace_output_keys": json.dumps(
            ["energy", "node_energy", "forces", "stress"]
        ),
        "tace_target_property": json.dumps(["energy", "forces", "stress"]),
        "tace_embedding_property": json.dumps([]),
        "tace_atomic_numbers": json.dumps([1, 8]),
        "tace_cutoff": "5.0",
        "tace_max_neighbors": "null",
        "tace_fidelity_idx": "0",
        "tace_dtype": "float32",
        "tace_export_num_graphs": "2",
    }
    model = AOTICompiledTensorModel(compiled_model, metadata, "cpu")
    data = {
        "positions": torch.zeros((2, 3)),
        "node_attrs": torch.eye(2),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "edge_shifts": torch.zeros((2, 3)),
        "lattice": torch.eye(3).unsqueeze(0),
        "batch": torch.zeros(2, dtype=torch.int64),
        "ptr": torch.tensor([0, 2]),
    }

    output = model(data)

    assert compiled_model.inputs[4].shape == (1, 3, 3)
    assert torch.equal(compiled_model.inputs[6], torch.tensor([0, 2]))
    assert torch.equal(compiled_model.inputs[7], torch.tensor([0]))
    assert output["energy"].shape == (1,)
    assert output["stress"].shape == (1, 3, 3)
    assert output["node_energy"].shape == (2,)
    assert output["forces"].shape == (2, 3)


def test_dynamic_aoti_package_preserves_minimum_graph_inputs():
    compiled_model = FakeCompiledModel()
    metadata = {
        "tace_input_keys": json.dumps(
            [
                "positions",
                "node_attrs",
                "edge_index",
                "edge_shifts",
                "lattice",
                "batch",
                "ptr",
                "fidelity_idx",
            ]
        ),
        "tace_output_keys": json.dumps(
            ["energy", "node_energy", "forces", "stress"]
        ),
        "tace_target_property": json.dumps(["energy", "forces", "stress"]),
        "tace_embedding_property": json.dumps([]),
        "tace_atomic_numbers": json.dumps([1, 8]),
        "tace_cutoff": "5.0",
        "tace_max_neighbors": "null",
        "tace_fidelity_idx": "0",
        "tace_dtype": "float32",
        "tace_export_num_graphs": "2",
        "AOTI_DEVICE_KEY": "cpu",
    }
    model = AOTICompiledTensorModel(compiled_model, metadata, None)
    data = {
        "positions": torch.zeros((2, 3)),
        "node_attrs": torch.eye(2),
        "edge_index": torch.tensor([[0], [1]]),
        "edge_shifts": torch.zeros((1, 3)),
        "lattice": torch.eye(3).unsqueeze(0),
        "batch": torch.zeros(2, dtype=torch.int64),
        "ptr": torch.tensor([0, 2]),
    }

    output = model(data)

    assert compiled_model.inputs[0].shape == (2, 3)
    assert compiled_model.inputs[2].shape == (2, 1)
    assert torch.equal(compiled_model.inputs[5], torch.tensor([0, 0]))
    assert torch.equal(compiled_model.inputs[6], torch.tensor([0, 2]))
    assert output["energy"].shape == (1,)
    assert output["stress"].shape == (1, 3, 3)
    assert output["node_energy"].shape == (2,)
    assert output["forces"].shape == (2, 3)


class GraphCountModel(torch.nn.Module):
    def forward(self, lattice, ptr):
        num_graphs = ptr.numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3), dtype=lattice.dtype, device=lattice.device
        )
        return torch.matmul(lattice, displacement)


def test_exported_graph_accepts_one_system_without_input_changes():
    inputs = (
        torch.eye(3).repeat(2, 1, 1),
        torch.tensor([0, 2, 4]),
    )
    num_graphs = torch.export.Dim("test_num_graphs", min=1)
    with _size_oblivious_export():
        traced = trace_to_fx(GraphCountModel(), inputs)
        exported = torch.export.export(
            traced,
            inputs,
            dynamic_shapes=({0: num_graphs}, {0: num_graphs + 1}),
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )

    lattice = torch.eye(3).unsqueeze(0)
    ptr = torch.tensor([0, 2])
    output = exported.module()(lattice, ptr)

    assert output.shape == (1, 3, 3)
    assert lattice.shape == (1, 3, 3)
    assert torch.equal(ptr, torch.tensor([0, 2]))


class OpenEquivarianceModule(torch.nn.Module):
    pass


OpenEquivarianceModule.__module__ = "tace.models._oeq.test"


class CuequivarianceModule(torch.nn.Module):
    pass


CuequivarianceModule.__module__ = "tace.models._cue.test"


def test_custom_ops_are_detected_from_constructed_modules():
    model = torch.nn.Sequential(OpenEquivarianceModule(), CuequivarianceModule())

    assert _custom_ops_libs_from_model(model) == {
        "openequivariance",
        "cuequivariance",
        "cuequivariance_torch",
    }


def test_custom_ops_metadata_uses_pt2_archive_root(tmp_path, monkeypatch):
    package = tmp_path / "model.pt2"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("model/archive_format", "pt2")

    _embed_custom_ops_libs(package, {"openequivariance"})

    with zipfile.ZipFile(package) as archive:
        entry = f"model/{TACE_AOTI_CUSTOM_OPS_LIBS_ENTRY}"
        assert entry in archive.namelist()
        assert archive.read(entry) == b"openequivariance"

    imported = []
    monkeypatch.setattr(aot.importlib, "import_module", imported.append)
    _import_custom_ops_libs(package)
    assert imported == ["openequivariance"]
