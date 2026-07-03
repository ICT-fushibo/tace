################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

import ase.optimize
import ase.optimize.sciopt
from ase import units
from ase.filters import ExpCellFilter, FrechetCellFilter, StrainFilter, UnitCellFilter
from ase.io import read, write

from tace.interface.ase import TACEAseCalc, add_dispersion


FILTER_CLS = {
    None: None,
    "none": None,
    "null": None,
    "unit": UnitCellFilter,
    "UnitCellFilter": UnitCellFilter,
    "strain": StrainFilter,
    "StrainFilter": StrainFilter,
    "exp": ExpCellFilter,
    "ExpCellFilter": ExpCellFilter,
    "frechet": FrechetCellFilter,
    "FrechetCellFilter": FrechetCellFilter,
}

OPTIMIZER_CLS = {
    "BFGS": ase.optimize.BFGS,
    "LBFGS": ase.optimize.LBFGS,
    "BFGSLineSearch": ase.optimize.BFGSLineSearch,
    "LBFGSLineSearch": ase.optimize.LBFGSLineSearch,
    "QuasiNewton": ase.optimize.BFGSLineSearch,
    "SciPyFminBFGS": ase.optimize.sciopt.SciPyFminBFGS,
    "SciPyFminCG": ase.optimize.sciopt.SciPyFminCG,
    "FIRE": ase.optimize.FIRE,
    "FIRE2": ase.optimize.FIRE2,
    "GPMin": ase.optimize.GPMin,
    "GOQN": ase.optimize.GoodOldQuasiNewton,
}


def _load_yaml(path: Union[str, Path, None]) -> Dict[str, Any]:
    if path is None:
        return {}

    path = Path(path)
    if not path.exists():
        return {}

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _load_parameters_json(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _split_list(value: Union[str, List[str], None]) -> Union[List[str], None]:
    if value is None or isinstance(value, list):
        return value
    return [item.strip() for item in value.split(",") if item.strip()]


def _set_if_not_none(cfg: Dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        cfg[key] = value


def _merged_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_yaml(args.config)
    params = _load_parameters_json(args.parameters)

    for key in ("ExternalPressure", "externalPressure", "external_pressure"):
        if key in params and "external_pressure_gpa" not in cfg:
            cfg["external_pressure_gpa"] = params[key]

    _set_if_not_none(cfg, "model", args.model)
    _set_if_not_none(cfg, "input", args.input)
    _set_if_not_none(cfg, "output", args.output)
    _set_if_not_none(cfg, "energy_output", args.energy_output)
    _set_if_not_none(cfg, "dtype", args.dtype)
    _set_if_not_none(cfg, "device", args.device)
    _set_if_not_none(cfg, "fidelity_idx", args.fidelity_idx)
    _set_if_not_none(cfg, "target_property", args.target_property)
    _set_if_not_none(cfg, "neighborlist_backend", args.neighborlist_backend)
    _set_if_not_none(cfg, "fmax", args.fmax)
    _set_if_not_none(cfg, "max_step", args.max_step)
    _set_if_not_none(cfg, "optimizer_name", args.optimizer_name)
    _set_if_not_none(cfg, "filter_name", args.filter_name)
    _set_if_not_none(cfg, "trajectory", args.trajectory)
    _set_if_not_none(cfg, "logfile", args.logfile)
    _set_if_not_none(cfg, "external_pressure_gpa", args.external_pressure_gpa)
    _set_if_not_none(cfg, "energy_mode", args.energy_mode)
    _set_if_not_none(cfg, "dispersion", args.dispersion)

    cfg.setdefault("input", "geom.in")
    cfg.setdefault("output", "geom.out")
    cfg.setdefault("energy_output", "energy.txt")
    cfg.setdefault("dtype", "float32")
    cfg.setdefault("device", "cpu")
    cfg.setdefault("neighborlist_backend", "matscipy")
    cfg.setdefault("fmax", 0.05)
    cfg.setdefault("max_step", 1000)
    cfg.setdefault("optimizer_name", "FIRE")
    cfg.setdefault("filter_name", None)
    cfg.setdefault("logfile", "-")
    cfg.setdefault("energy_mode", "energy")
    cfg.setdefault("dispersion", False)
    cfg["target_property"] = _split_list(cfg.get("target_property"))
    return cfg


def _build_filter(atoms, filter_name: Union[str, None], pressure_gpa: float):
    filter_cls = FILTER_CLS.get(filter_name)
    if filter_cls is None:
        if pressure_gpa != 0.0:
            raise ValueError("external_pressure_gpa requires a cell filter")
        return atoms

    kwargs = {}
    if pressure_gpa != 0.0:
        parameters = inspect.signature(filter_cls).parameters.values()
        accepts_pressure = any(
            parameter.name == "scalar_pressure"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if not accepts_pressure:
            raise ValueError(f"{filter_name} does not accept scalar_pressure")
        kwargs["scalar_pressure"] = pressure_gpa * units.GPa
    return filter_cls(atoms, **kwargs)


def _compute_output_energy(atoms, cfg: Dict[str, Any]) -> float:
    energy = float(atoms.get_potential_energy())
    mode = str(cfg.get("energy_mode", "energy")).lower()
    if mode == "energy":
        return energy
    if mode == "enthalpy":
        pressure = float(cfg.get("external_pressure_gpa", 0.0)) * units.GPa
        return energy + pressure * float(atoms.get_volume())
    raise ValueError("energy_mode must be 'energy' or 'enthalpy'")


def _check_relaxation_properties(calc, cfg: Dict[str, Any]) -> None:
    properties = set(calc.implemented_properties)
    if "energy" not in properties:
        raise ValueError("TACE model must provide energy for USPEX relaxation")
    if "forces" not in properties:
        raise ValueError("TACE model must provide forces for USPEX relaxation")
    if FILTER_CLS.get(cfg.get("filter_name")) is not None and "stress" not in properties:
        raise ValueError(
            "Cell relaxation requires stress. Provide a model with stress output "
            "or set filter_name: null."
        )


def run_relaxation(cfg: Dict[str, Any]) -> float:
    model = cfg.get("model")
    if model is None:
        raise ValueError("Missing 'model' in usercode.in or --model")

    atoms = read(cfg["input"], format="vasp")
    calc = TACEAseCalc(
        model=model,
        dtype=cfg.get("dtype"),
        device=cfg.get("device"),
        fidelity_idx=cfg.get("fidelity_idx"),
        target_property=cfg.get("target_property"),
        neighborlist_backend=cfg.get("neighborlist_backend", "matscipy"),
    )

    if bool(cfg.get("dispersion", False)):
        calc = add_dispersion(
            calc,
            damping=cfg.get("dispersion_damping", "bj"),
            dispersion_xc=cfg.get("dispersion_xc", "pbe"),
        )

    _check_relaxation_properties(calc, cfg)
    atoms.calc = calc

    pressure_gpa = float(cfg.get("external_pressure_gpa", 0.0))
    atoms_opt = _build_filter(atoms, cfg.get("filter_name"), pressure_gpa)

    optimizer_name = cfg.get("optimizer_name", "FIRE")
    if optimizer_name not in OPTIMIZER_CLS:
        raise ValueError(
            f"Unknown optimizer_name={optimizer_name}. "
            f"Available: {sorted(OPTIMIZER_CLS)}"
        )

    optimizer_kwargs = {"logfile": cfg.get("logfile", "-")}
    if cfg.get("trajectory") is not None:
        optimizer_kwargs["trajectory"] = cfg["trajectory"]

    opt = OPTIMIZER_CLS[optimizer_name](atoms_opt, **optimizer_kwargs)
    opt.run(fmax=float(cfg["fmax"]), steps=int(cfg["max_step"]))

    output_energy = _compute_output_energy(atoms, cfg)
    write(cfg["output"], atoms, format="vasp", direct=True, vasp5=True)
    Path(cfg["energy_output"]).write_text(f"{output_energy:.16g}\n")
    return output_energy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "USPEX abinitioCode=99 wrapper for TACE. Reads geom.in, writes "
            "geom.out and energy.txt."
        )
    )
    parser.add_argument("--config", default="usercode.in")
    parser.add_argument("--parameters", default="parameters.json")
    parser.add_argument("--model")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--energy-output")
    parser.add_argument("--dtype")
    parser.add_argument("--device")
    parser.add_argument("--fidelity-idx", type=int)
    parser.add_argument("--target-property")
    parser.add_argument("--neighborlist-backend")
    parser.add_argument("--fmax", type=float)
    parser.add_argument("--max-step", type=int)
    parser.add_argument("--optimizer-name")
    parser.add_argument("--filter-name")
    parser.add_argument("--trajectory")
    parser.add_argument("--logfile")
    parser.add_argument("--external-pressure-gpa", type=float)
    parser.add_argument("--energy-mode", choices=["energy", "enthalpy"])
    parser.add_argument("--dispersion", action="store_true")
    return parser.parse_args()


def main() -> None:
    cfg = _merged_config(parse_args())
    energy = run_relaxation(cfg)
    print(f"TACE USPEX relaxation finished: {energy:.16g}")


if __name__ == "__main__":
    main()
