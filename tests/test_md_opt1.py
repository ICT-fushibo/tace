from __future__ import annotations

import ast
import inspect
import sys
import textwrap
import types

import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.nvtberendsen import NVTBerendsen

from md_benchmark.md_route import MDConfig, MDRunRequest
from tace import md_route
from tace.md_stages.opt1 import (
    BerendsenIntegrator,
    GPUMDState,
    NoseHooverChainIntegrator,
    TACETorchSimEvaluator,
    _snapshot,
    _validate_request,
)


class _ConstantForceCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, forces: np.ndarray) -> None:
        super().__init__()
        self.forces = forces

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": self.forces.copy()}


class _ConstantEvaluator:
    def __init__(self, forces: torch.Tensor) -> None:
        self.forces = forces

    def __call__(self, positions):
        return self.forces.to(positions), positions.new_zeros(()), None


@pytest.mark.parametrize("integrator_name", ["berendsen", "nose_hoover_chain"])
def test_gpu_integrators_match_one_ase_step_on_cpu(integrator_name: str):
    positions = np.array([[0.1, 0.2, 0.3], [1.1, 0.7, 0.4]])
    momenta = np.array([[0.22, -0.13, 0.31], [-0.19, 0.17, -0.28]])
    forces = np.array([[0.03, -0.02, 0.01], [-0.04, 0.02, -0.01]])
    masses = np.array([12.0, 16.0])
    atoms = Atoms("CO", positions=positions, masses=masses)
    atoms.set_momenta(momenta)
    atoms.calc = _ConstantForceCalculator(forces)
    if integrator_name == "berendsen":
        ase_md = NVTBerendsen(
            atoms,
            timestep=units.fs,
            temperature_K=300.0,
            taut=100.0 * units.fs,
            fixcm=True,
        )
        gpu_integrator = BerendsenIntegrator(
            torch.tensor(masses, dtype=torch.float64),
            timestep_fs=1.0,
            temperature_k=300.0,
            thermostat_time_fs=100.0,
            degrees_of_freedom=atoms.get_number_of_degrees_of_freedom(),
        )
    else:
        ase_md = NoseHooverChainNVT(
            atoms,
            timestep=units.fs,
            temperature_K=300.0,
            tdamp=100.0 * units.fs,
        )
        gpu_integrator = NoseHooverChainIntegrator(
            torch.tensor(masses, dtype=torch.float64),
            timestep_fs=1.0,
            temperature_k=300.0,
            thermostat_time_fs=100.0,
        )
    ase_md.run(1)
    state = GPUMDState(
        torch.tensor(positions, dtype=torch.float64),
        torch.tensor(momenta, dtype=torch.float64),
    )
    gpu_integrator.step(
        state, _ConstantEvaluator(torch.tensor(forces, dtype=torch.float64))
    )
    np.testing.assert_allclose(
        state.positions.numpy(), atoms.positions, rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(
        state.momenta.numpy(), atoms.get_momenta(), rtol=1e-13, atol=1e-13
    )


def _request(*, backend: str = "gpu-resident", dtype: str = "float64"):
    return MDRunRequest(
        model="tace",
        stage="opt1",
        model_path="TACE-OAM-L.pt",
        atoms=Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]]),
        config=MDConfig(
            device="cuda:0", dtype=dtype, steps=1, observation_steps=(1,)
        ),
        backend=backend,
        options={"model_dtype": "checkpoint"},
    )


def test_opt1_rejects_non_gpu_resident_backend_before_loading_model():
    with pytest.raises(ValueError, match="gpu-resident"):
        _validate_request(_request(backend="eager"))


def test_opt1_requires_fp64_md_state():
    with pytest.raises(ValueError, match="float64"):
        _validate_request(_request(dtype="float32"))


def test_model_route_dispatches_opt1_without_touching_baseline(monkeypatch):
    request = _request()
    sentinel = object()

    def fake_optimized_stage(actual_request, *, module_prefix):
        assert actual_request is request
        assert module_prefix == "tace.md_stages"
        return sentinel

    monkeypatch.setattr(md_route, "run_optimized_stage", fake_optimized_stage)
    assert md_route.run_md(request) is sentinel


def test_opt1_rejects_constraints():
    request = _request()
    request.atoms.set_constraint()
    # An empty constraint list is allowed; use a real fixed-atom constraint.
    from ase.constraints import FixAtoms

    request.atoms.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(NotImplementedError, match="constraints"):
        _validate_request(request)


def test_matbench_snapshot_contains_step_energy_forces_and_stress():
    atoms = Atoms(
        "H2", positions=[[0, 0, 0], [0, 0, 0.7]], cell=[5, 5, 5], pbc=True
    )
    state = GPUMDState(
        positions=torch.tensor(atoms.positions, dtype=torch.float64),
        momenta=torch.zeros(2, 3, dtype=torch.float64),
        forces=torch.ones(2, 3, dtype=torch.float64),
        potential_energy=torch.tensor(-1.25, dtype=torch.float64),
        stress=torch.eye(3, dtype=torch.float64),
    )
    frame = _snapshot(atoms, state, step=0, require_stress=True)
    assert frame.info["md_step"] == 0
    assert frame.get_potential_energy() == -1.25
    np.testing.assert_allclose(frame.get_forces(), np.ones((2, 3)))
    np.testing.assert_allclose(frame.get_stress(voigt=False), np.eye(3))


@pytest.mark.parametrize("compute_stress", [False, True])
def test_evaluator_forwards_stress_flag_and_caches_static_inputs(
    monkeypatch, compute_stress: bool
):
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, requested_stress: bool) -> None:
            self.flags = types.SimpleNamespace(
                compute_forces=True, compute_stress=requested_stress
            )

        def get_model_dtype(self):
            return torch.float32

        def get_target_property(self):
            return ["energy", "forces", "stress"]

        def get_cutoff(self):
            return 5.0

        def modules(self):
            return []

    class FakeCalculator:
        def __init__(self, model_path, **kwargs) -> None:
            captured["model_path"] = model_path
            captured.update(kwargs)
            self.compute_stress = kwargs["compute_stress"]
            self.model = FakeModel(self.compute_stress)

        def __call__(self, state):
            outputs = {
                "energy": state.positions.new_tensor([-1.0]),
                "forces": torch.ones_like(state.positions),
            }
            if self.compute_stress:
                outputs["stress"] = torch.eye(
                    3, dtype=state.positions.dtype, device=state.positions.device
                ).unsqueeze(0)
            return outputs

    def atoms_to_state(atoms, *, device, dtype):
        return types.SimpleNamespace(
            positions=torch.tensor(
                atoms[0].positions,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )

    torch_sim_module = types.ModuleType("torch_sim")
    torch_sim_module.io = types.SimpleNamespace(atoms_to_state=atoms_to_state)
    neighbors_module = types.ModuleType("torch_sim.neighbors")
    neighbors_module.torchsim_nl = object()
    interface_module = types.ModuleType("tace.interface.torchsim")
    interface_module.TACETorchSimCalc = FakeCalculator
    monkeypatch.setitem(sys.modules, "torch_sim", torch_sim_module)
    monkeypatch.setitem(sys.modules, "torch_sim.neighbors", neighbors_module)
    monkeypatch.setitem(sys.modules, "tace.interface.torchsim", interface_module)

    atoms = Atoms(
        "H2", positions=[[0, 0, 0], [0, 0, 0.7]], cell=[5, 5, 5], pbc=True
    )
    evaluator = TACETorchSimEvaluator(
        atoms,
        "TACE-OAM-L.pt",
        device=torch.device("cpu"),
        compute_stress=compute_stress,
    )
    _, _, stress = evaluator(torch.tensor(atoms.positions, dtype=torch.float64))

    assert captured["compute_stress"] is compute_stress
    assert captured["compute_forces"] is True
    assert torch.equal(
        captured["atomic_numbers"], torch.tensor([1, 1], dtype=torch.int64)
    )
    assert torch.equal(captured["system_idx"], torch.zeros(2, dtype=torch.int64))
    assert evaluator.calculator.model.flags.compute_stress is compute_stress
    assert (stress is not None) is compute_stress


def test_evaluator_hot_path_has_no_temporary_cast_or_host_transfer():
    tree = ast.parse(textwrap.dedent(inspect.getsource(TACETorchSimEvaluator.__call__)))
    forbidden = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert forbidden.isdisjoint({"to", "cpu", "numpy", "item", "tolist"})


def test_opt1_rejects_non_boolean_compute_stress_option():
    request = _request()
    request.options["compute_stress"] = "false"
    with pytest.raises(ValueError, match="compute_stress must be a boolean"):
        _validate_request(request)
