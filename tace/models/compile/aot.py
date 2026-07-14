import json
from pathlib import Path
from typing import Dict, Sequence, Union

import torch

from tace.dataset.element import TorchElement
from .compile import trace_to_fx
from .wrapper import CompileTensorModel, _FlatE3nnCompileModel


ASE_AOTI_FORMAT = "tace_ase_v1"
ASE_AOTI_INPUT_KEYS = (
    "positions",
    "node_attrs",
    "edge_index",
    "edge_shifts",
    "lattice",
    "batch",
    "ptr",
    "fidelity_idx",
)


class AOTICompiledTensorModel(torch.nn.Module):
    def __init__(
        self,
        compiled_model,
        metadata: Dict[str, str],
        device: Union[str, torch.device],
    ) -> None:
        super().__init__()
        self.compiled_model = compiled_model
        self.metadata = metadata
        self.input_keys = tuple(_metadata_json(metadata, "tace_input_keys"))
        self.output_keys = tuple(_metadata_json(metadata, "tace_output_keys"))
        self.exported_target_property = list(
            _metadata_json(metadata, "tace_target_property")
        )
        self.target_property = list(self.exported_target_property)
        self.embedding_property = list(_metadata_json(metadata, "tace_embedding_property"))
        self.atomic_numbers = [int(z) for z in _metadata_json(metadata, "tace_atomic_numbers")]
        self.cutoff = float(metadata["tace_cutoff"])
        self.max_neighbors = _metadata_json(metadata, "tace_max_neighbors")
        self.fidelity_idx = int(metadata["tace_fidelity_idx"])
        self.model_dtype = _dtype_from_name(metadata["tace_dtype"])
        self.compile_device = torch.device(metadata.get("AOTI_DEVICE_KEY", str(device)))
        self._check_device(device)

    def forward(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[torch.Tensor, None]]:
        missing = [key for key in self.input_keys if key not in data]
        if missing:
            raise KeyError(f"missing TACE ASE .pt2 inputs: {missing}")
        outputs = self.compiled_model(*(data[key] for key in self.input_keys))
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        result: Dict[str, Union[torch.Tensor, None]] = {
            "energy": None,
            "node_energy": None,
            "forces": None,
            "virials": None,
            "stress": None,
            "direct_forces": None,
            "direct_virials": None,
            "direct_stress": None,
        }
        result.update(zip(self.output_keys, outputs))
        return result

    def reset_target_property(self, target_property: list[str]) -> None:
        missing = set(target_property) - set(self.exported_target_property)
        if missing:
            raise ValueError(
                "The TACE ASE .pt2 model was not exported with target "
                f"properties {sorted(missing)}."
            )
        self.target_property = list(target_property)

    def reset_fidelity_idx(self, fidelity_idx: Union[int, None] = 0) -> None:
        if fidelity_idx is not None:
            self.fidelity_idx = int(fidelity_idx)

    def get_fidelity_idx(self) -> int:
        return int(self.fidelity_idx)

    def get_embedding_property(self) -> list[str]:
        return list(self.embedding_property)

    def get_target_property(self) -> list[str]:
        return list(self.target_property)

    def get_model_dtype(self) -> torch.dtype:
        return self.model_dtype

    def get_max_neighbors(self) -> Union[int, None]:
        return self.max_neighbors

    def get_cutoff(self) -> float:
        return self.cutoff

    def get_atomic_numbers(self) -> list[int]:
        return list(self.atomic_numbers)

    def get_torch_element(self) -> TorchElement:
        return TorchElement(self.atomic_numbers)

    def _check_device(self, device: Union[str, torch.device]) -> None:
        requested = torch.device(device)
        if self.compile_device == requested:
            return
        if self.compile_device.type == "cuda" and requested.type == "cuda":
            return
        raise RuntimeError(
            f"TACE ASE .pt2 was compiled for {self.compile_device}, "
            f"but device={requested} was requested."
        )


def export_ase_aotinductor(
    model: torch.nn.Module,
    output_path: Union[str, Path],
    sample_data: Union[Dict[str, torch.Tensor], None] = None,
) -> str:
    model.eval()
    compile_model = _as_compile_tensor_model(model)
    CompileTensorModel._validate_compile_properties(compile_model.readout_fn)
    input_keys = ASE_AOTI_INPUT_KEYS
    output_keys = compile_model._output_keys()
    if not output_keys:
        raise ValueError("TACE ASE .pt2 export needs at least one output property.")

    if sample_data is None:
        sample_data = _synthetic_ase_sample(compile_model)
    else:
        sample_data = {key: value for key, value in sample_data.items()}
    _ensure_sample_inputs(sample_data, input_keys, compile_model)

    flat_model = _FlatE3nnCompileModel(compile_model, input_keys, output_keys)
    flat_model.eval()
    inputs = tuple(sample_data[key] for key in input_keys)
    traced = trace_to_fx(flat_model, inputs)
    dynamic_shapes = _ase_dynamic_shapes()
    exported = torch.export.export(
        traced,
        inputs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
        prefer_deferred_runtime_asserts_over_guards=True,
    )

    output_path = str(output_path)
    metadata = _export_metadata(compile_model, input_keys, output_keys)
    inductor_configs = _valid_inductor_configs(
        {
            "aot_inductor.metadata": metadata,
            "max_autotune": False,
            "shape_padding": True,
            "epilogue_fusion": False,
            "triton.cudagraphs": False,
            "max_fusion_size": 8,
            "triton.persistent_reductions": False,
            "triton.max_tiles": 1,
        }
    )
    out_path = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=output_path,
        inductor_configs=inductor_configs,
    )
    return str(out_path)


def load_ase_aotinductor(
    model_path: Union[str, Path],
    device: Union[str, torch.device],
) -> AOTICompiledTensorModel:
    compiled_model = torch._inductor.aoti_load_package(str(model_path))
    metadata = dict(compiled_model.get_metadata())
    if metadata.get("tace_format") != ASE_AOTI_FORMAT:
        raise ValueError(
            f"{model_path} is not a TACE ASE .pt2 package "
            f"({metadata.get('tace_format')!r})."
        )
    return AOTICompiledTensorModel(compiled_model, metadata, device)


def _as_compile_tensor_model(model: torch.nn.Module) -> CompileTensorModel:
    if isinstance(model, CompileTensorModel):
        return model
    if hasattr(model, "readout_fn"):
        wrapped = CompileTensorModel(model.readout_fn)
        wrapped.reset_fidelity_idx(model.get_fidelity_idx())
        wrapped.train(model.training)
        return wrapped
    raise TypeError("TACE ASE .pt2 export requires a TensorModel-like model.")


def _synthetic_ase_sample(model: CompileTensorModel) -> Dict[str, torch.Tensor]:
    dtype = model.get_model_dtype()
    device = next(model.parameters()).device
    num_elements = len(model.get_atomic_numbers())
    node_attrs = torch.zeros((2, num_elements), dtype=dtype, device=device)
    node_attrs[:, 0] = 1.0
    return {
        "positions": torch.tensor(
            [[0.0, 0.0, 0.0], [0.5 * model.get_cutoff(), 0.0, 0.0]],
            dtype=dtype,
            device=device,
        ),
        "node_attrs": node_attrs,
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.int64, device=device),
        "edge_shifts": torch.zeros((2, 3), dtype=dtype, device=device),
        "lattice": torch.eye(3, dtype=dtype, device=device).reshape(1, 3, 3)
        * max(model.get_cutoff() * 4.0, 1.0),
        "batch": torch.zeros(2, dtype=torch.int64, device=device),
        "ptr": torch.tensor([0, 2], dtype=torch.int64, device=device),
        "fidelity_idx": torch.tensor([model.get_fidelity_idx()], dtype=torch.int64, device=device),
    }


def _ensure_sample_inputs(
    sample_data: Dict[str, torch.Tensor],
    input_keys: Sequence[str],
    model: CompileTensorModel,
) -> None:
    missing = [key for key in input_keys if key not in sample_data]
    if missing:
        raise KeyError(f"missing TACE ASE .pt2 sample inputs: {missing}")
    device = next(model.parameters()).device
    dtype = model.get_model_dtype()
    for key, value in list(sample_data.items()):
        if not isinstance(value, torch.Tensor):
            continue
        if torch.is_floating_point(value):
            sample_data[key] = value.to(device=device, dtype=dtype)
        else:
            sample_data[key] = value.to(device=device)


def _ase_dynamic_shapes() -> tuple[Dict[int, object], ...]:
    num_nodes = torch.export.Dim("num_nodes", min=2)
    num_edges = torch.export.Dim("num_edges", min=2)
    return (
        {0: num_nodes},
        {0: num_nodes},
        {1: num_edges},
        {0: num_edges},
        {},
        {0: num_nodes},
        {},
        {},
    )


def _export_metadata(
    model: CompileTensorModel,
    input_keys: Sequence[str],
    output_keys: Sequence[str],
) -> Dict[str, str]:
    return {
        "tace_format": ASE_AOTI_FORMAT,
        "tace_input_keys": json.dumps(list(input_keys)),
        "tace_output_keys": json.dumps(list(output_keys)),
        "tace_target_property": json.dumps(model.get_target_property()),
        "tace_embedding_property": json.dumps(model.get_embedding_property()),
        "tace_atomic_numbers": json.dumps(model.get_atomic_numbers()),
        "tace_cutoff": str(model.get_cutoff()),
        "tace_max_neighbors": json.dumps(model.get_max_neighbors()),
        "tace_fidelity_idx": str(model.get_fidelity_idx()),
        "tace_dtype": str(model.get_model_dtype()).replace("torch.", ""),
    }


def _valid_inductor_configs(configs: Dict[str, object]) -> Dict[str, object]:
    try:
        from torch._inductor import config

        valid = config.get_config_copy()
        return {
            key: value
            for key, value in configs.items()
            if key.replace("-", "_") in valid
        }
    except Exception:
        return configs


def _metadata_json(metadata: Dict[str, str], key: str):
    value = metadata[key]
    if isinstance(value, bytes):
        value = value.decode()
    return json.loads(value)


def _dtype_from_name(name: str) -> torch.dtype:
    if name.startswith("torch."):
        name = name[len("torch.") :]
    if not hasattr(torch, name):
        raise ValueError(f"unsupported TACE ASE .pt2 dtype {name!r}")
    dtype = getattr(torch, name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported TACE ASE .pt2 dtype {name!r}")
    return dtype
