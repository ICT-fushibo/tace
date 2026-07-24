import json

import torch

from tace.models.compile.aot import AOTICompiledTensorModel


class FakeCompiledModel:
    def __init__(self):
        self.inputs = None

    def __call__(self, *inputs):
        self.inputs = inputs
        return (
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.4, 0.6]),
            torch.ones((2, 3)),
            torch.stack((torch.eye(3), torch.zeros((3, 3)))),
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

    assert compiled_model.inputs[4].shape == (2, 3, 3)
    assert torch.equal(compiled_model.inputs[6], torch.tensor([0, 2, 2]))
    assert torch.equal(compiled_model.inputs[7], torch.tensor([0, 0]))
    assert output["energy"].shape == (1,)
    assert output["stress"].shape == (1, 3, 3)
    assert output["node_energy"].shape == (2,)
    assert output["forces"].shape == (2, 3)
