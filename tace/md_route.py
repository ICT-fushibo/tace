"""Stable TACE MD route for the shared acceleration benchmark.

``baseline/eager`` is deliberately a scientific reference rather than TACE's
fastest available ASE configuration: it uses the native eager modules, the
Matscipy neighbor list, checkpoint precision selected by the caller, and no
TF32.  Existing TACE acceleration paths remain available as explicit controls
through ``request.backend`` and are never selected from ambient ``TACE_USE_*``
environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from md_benchmark.md_route import (
    MDRunRequest,
    MDRunResult,
    configure_torch_baseline,
    run_ase_baseline,
    run_optimized_stage,
)


_ACCELERATION_ENV = {
    "oeq": "TACE_USE_OEQ",
    "cueq": "TACE_USE_CUE",
    "eqt": "TACE_USE_EQT",
    "compile": "TACE_USE_COMPILE",
    "triton": "TACE_USE_TRITON",
}
_ACCELERATORS = frozenset((*_ACCELERATION_ENV, "aoti"))
_NEIGHBOR_LISTS = frozenset({"ase", "matscipy", "vesin", "alchemiops"})


def _parse_backend(backend: str) -> tuple[set[str], str]:
    tokens = {token.strip().lower() for token in backend.split("+") if token.strip()}
    if not tokens:
        raise ValueError("TACE backend cannot be empty")
    supported = {"eager", *_ACCELERATORS, *_NEIGHBOR_LISTS}
    if unknown := tokens - supported:
        raise ValueError(f"Unknown TACE backend components: {sorted(unknown)}")

    neighbor_lists = tokens & _NEIGHBOR_LISTS
    if len(neighbor_lists) > 1:
        raise ValueError(
            "Select exactly one TACE neighbor-list backend, got "
            f"{sorted(neighbor_lists)}"
        )
    neighbor_list = next(iter(neighbor_lists), "matscipy")

    accelerators = tokens & _ACCELERATORS
    if "eager" in tokens and accelerators:
        raise ValueError("TACE eager cannot be combined with acceleration backends")
    if {"oeq", "cueq"} <= accelerators:
        raise ValueError("TACE oeq and cueq are alternative edge backends")
    if "aoti" in accelerators:
        accelerators.add("compile")
    return accelerators, neighbor_list


def _set_exact_acceleration_environment(accelerators: set[str]) -> None:
    """Overwrite every TACE runtime switch before importing model modules."""
    for name, env_name in _ACCELERATION_ENV.items():
        os.environ[env_name] = "1" if name in accelerators else "0"


def _detect_acceleration_modules(model: Any) -> list[str]:
    """Return accelerator namespaces actually materialized in the loaded model."""
    detected: set[str] = set()
    needles = {
        ".models.oeq": "oeq",
        "openequivariance": "oeq",
        ".models.cue": "cueq",
        "cuequivariance": "cueq",
        ".models.eqt": "eqt",
        "equitorch": "eqt",
        ".models.compile": "compile",
        ".models.triton_ops": "triton",
    }
    for module in model.modules():
        namespace = f"{type(module).__module__}.{type(module).__qualname__}".lower()
        for needle, component in needles.items():
            if needle in namespace:
                detected.add(component)
    if hasattr(model, "compiled_model"):
        detected.add("aoti")
    return sorted(detected)


def _validate_model_contract(
    calculator: Any, *, requested_accelerators: set[str]
) -> dict[str, Any]:
    model = calculator.model
    targets = list(model.get_target_property())
    missing = {"energy", "forces"} - set(targets)
    if missing:
        raise ValueError(
            "TACE MD requires energy and forces, but the checkpoint is missing "
            f"{sorted(missing)} (targets={targets})"
        )

    detected = _detect_acceleration_modules(model)
    if not requested_accelerators and detected:
        raise RuntimeError(
            "TACE eager baseline materialized acceleration modules "
            f"{detected}. Use a state-dict TACE-OAM-L checkpoint and ensure it was "
            "not serialized after accelerator replacement."
        )
    # AOTI hides the captured custom-op module tree inside the package, but all
    # in-process controls should prove that the requested implementation was
    # actually materialized. This prevents timing a no-op Triton/EQT flag on an
    # incompatible checkpoint architecture.
    if "aoti" not in requested_accelerators:
        missing_accelerators = requested_accelerators - set(detected)
        if missing_accelerators:
            raise RuntimeError(
                "Requested TACE acceleration did not materialize in this checkpoint: "
                f"{sorted(missing_accelerators)}; detected={detected}"
            )
    return {
        "model_dtype": str(model.get_model_dtype()),
        "target_properties": targets,
        "supports_stress": "stress" in targets or "direct_stress" in targets,
        "cutoff_a": float(model.get_cutoff()),
        "detected_acceleration_modules": detected,
    }


def run_md(request: MDRunRequest) -> MDRunResult:
    if request.model != "tace":
        raise ValueError(f"tace.md_route does not own model {request.model!r}")
    if request.stage != "baseline":
        return run_optimized_stage(request, module_prefix="tace.md_stages")

    accelerators, neighbor_list = _parse_backend(request.backend)
    model_path = Path(request.model_path)
    if "aoti" in accelerators and model_path.suffix != ".pt2":
        raise ValueError(
            "TACE aoti control requires a pre-exported .pt2 package. The shared "
            "PyTorch 2.11 environment is valid for eager/in-process controls, but "
            "the current TACE AOTI exporter requires PyTorch >=2.13."
        )
    if model_path.suffix == ".pt2" and "aoti" not in accelerators:
        raise ValueError(
            "A TACE .pt2 model must be selected with backend containing aoti"
        )

    # Do this before importing TACE. Model construction reads the environment and
    # otherwise preserves ambient values, which can silently contaminate baseline.
    _set_exact_acceleration_environment(accelerators)
    configure_torch_baseline()

    from tace.interface.ase import TACEAseCalc

    model_dtype = request.options.get("model_dtype", "checkpoint")
    if model_dtype not in {"checkpoint", "float32", "float64"}:
        raise ValueError(
            "TACE route option model_dtype must be checkpoint, float32, or float64"
        )
    calculator = TACEAseCalc(
        model=str(model_path),
        # Matbench's TACE factory deliberately keeps checkpoint precision even
        # when the shared CLI says dtype=float64. Preserve that published
        # behavior by default; casting remains available as an explicit control.
        dtype=None if model_dtype == "checkpoint" else model_dtype,
        device=request.config.device,
        neighborlist_backend=neighbor_list,
        enable_oeq="oeq" in accelerators,
        enable_cue="cueq" in accelerators,
        enable_eqt="eqt" in accelerators,
        enable_compile="compile" in accelerators,
        enable_triton="triton" in accelerators,
    )
    model_metadata = _validate_model_contract(
        calculator,
        requested_accelerators=accelerators,
    )
    if request.config.collect_trajectory and not model_metadata["supports_stress"]:
        raise ValueError(
            "TACE Matbench trajectory evaluation requires checkpoint stress output; "
            f"targets are {model_metadata['target_properties']}"
        )

    return run_ase_baseline(
        request,
        calculator,
        metadata={
            "baseline_kind": (
                "scientific_eager"
                if not accelerators
                else "existing_optimization_control"
            ),
            "tace_accelerators": sorted(accelerators),
            "neighborlist_backend": neighbor_list,
            "tf32": False,
            "requested_md_dtype": request.config.dtype,
            "model_dtype_policy": model_dtype,
            "checkpoint_format": model_path.suffix,
            **model_metadata,
        },
    )
