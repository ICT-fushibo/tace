import unittest
from types import SimpleNamespace

import torch
from md_benchmark.stress_test_support import assert_replay_stable

from tace.md_stages.opt3 import TACEWholeStepPotential
from tace.models.utils import compute_symmetric_displacement


class ToyStrainedModel(torch.nn.Module):
    def forward(self, data):
        original = data["positions"]
        strain = compute_symmetric_displacement(data, 1)
        r = data["positions"][1] - data["positions"][0] + data["lattice"][0, 0]
        energy = 0.5 * r.square().sum()
        force, virial = torch.autograd.grad(energy, (original, strain))
        return {
            "energy": energy,
            "forces": -force,
            "stress": virial / data["_opt4_stress_volume"].reshape(1, 1, 1),
        }


class TACEStressTests(unittest.TestCase):
    def test_strain_dict_not_retained_across_replay(self):
        potential = object.__new__(TACEWholeStepPotential)
        potential.capture_stress = True
        potential.num_atoms = 2
        potential.static_positions = torch.tensor(
            [[0.2, 0.4, 0.6], [1.0, 0.3, 0.7]],
            device="cuda:0",
            dtype=torch.float64,
            requires_grad=True,
        )
        lattice = torch.tensor(
            [[[3.0, 0.2, 0], [0.1, 4.0, 0.3], [0.2, 0, 5.0]]],
            device="cuda:0",
            dtype=torch.float64,
        )
        potential.static_data = {
            "positions": potential.static_positions,
            "lattice": lattice,
            "batch": torch.zeros(2, device="cuda:0", dtype=torch.long),
            "_opt4_stress_volume": torch.linalg.det(lattice).abs().detach(),
        }
        potential.builder = SimpleNamespace(build=lambda *args, **kwargs: None)
        potential.model = ToyStrainedModel().eval()
        input_positions = potential.static_positions.detach().clone()
        step = torch.zeros((), device="cuda:0", dtype=torch.long)

        def body():
            force, energy = potential.evaluate(input_positions, step=step)
            return force, energy, potential.last_stress

        assert_replay_stable(body)
        self.assertIs(potential.static_data["positions"], potential.static_positions)
        self.assertIs(potential.static_data["lattice"], lattice)
        self.assertIsNone(potential.static_data["lattice"].grad_fn)


if __name__ == "__main__":
    unittest.main()
