"""Independent affine and finite-strain checks of the production edge helper."""
import importlib.util
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

path = Path(__file__).resolve().parents[1] / "tace/md_stages/stress_geometry.py"
spec = importlib.util.spec_from_file_location("edge_strain_production", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TACEEdgeStrainTests(unittest.TestCase):
    def fixture(self, dtype=torch.float64, device="cpu", dummy=False):
        # Non-contiguous positions, tilted cell, periodic bonds, repeated edges.
        p = torch.tensor([[0.2, 9, 0.3, 9, 0.4, 9], [1.1, 9, 0.1, 9, 0.8, 9]],
                         dtype=dtype, device=device)[:, ::2].requires_grad_()
        cell = torch.tensor([[[3, .2, .1], [.1, 4, .3], [.4, .2, 5]]], dtype=dtype, device=device)
        index = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 1]], device=device)
        shifts = torch.tensor([[1., 0, 0], [1, 0, 0], [-1, 0, 0], [4, 0, 0]], dtype=dtype, device=device)
        batch = torch.zeros(2, dtype=torch.long, device=device)
        strain = torch.zeros(1, 3, 3, dtype=dtype, device=device, requires_grad=True)
        mask = torch.tensor([0, 0, 0, 0] if dummy else [1, 1, 1, 0], dtype=dtype, device=device)
        return p, cell, index, shifts, batch, strain, mask

    def energy(self, data, *, native=False):
        p, cell, idx, shifts, batch, strain, mask = data
        if native:
            sym = (strain + strain.transpose(-1, -2)) * .5
            pp = p + torch.einsum("ni,nij->nj", p, sym[batch])
            cc = cell + cell @ sym
            r = pp[idx[1]] - pp[idx[0]] + shifts @ cc[0]
        else:
            r = module.strained_edge_vectors(p, cell, idx, shifts, batch, strain)
        return (.5 * r.square().sum(-1) * mask).sum()

    def test_joint_vjp_matches_native_affine_with_periodic_padding(self):
        for dtype in (torch.float32, torch.float64):
            for dummy in (False, True):
                data = self.fixture(dtype, dummy=dummy)
                actual = self.energy(data)
                expected = self.energy(data, native=True)
                a = torch.autograd.grad(actual, (data[0], data[5]))
                b = torch.autograd.grad(expected, (data[0], data[5]))
                torch.testing.assert_close(actual, expected)
                for x, y in zip(a, b):
                    torch.testing.assert_close(x, y)
                    self.assertTrue(bool(torch.isfinite(x).all()))
                self.assertEqual(float(data[5].abs().max()), 0)

    def test_six_components_against_affinely_strained_energy_difference(self):
        data = self.fixture()
        grad = torch.autograd.grad(self.energy(data), data[5])[0]
        for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
            h = 1e-5
            plus, minus = list(data), list(data)
            plus[5], minus[5] = torch.zeros_like(data[5]), torch.zeros_like(data[5])
            plus[5][0, i, j], minus[5][0, i, j] = h, -h
            fd = (self.energy(plus, native=True) - self.energy(minus, native=True)) / (2*h)
            torch.testing.assert_close(grad[0, i, j], fd, atol=1e-8, rtol=1e-8)

    def test_zero_strain_does_not_change_model_geometry_or_force(self):
        data = self.fixture()
        p, cell, idx, shift, batch, strain, _ = data
        want = p[idx[1]] - p[idx[0]] + shift @ cell[0]
        got = module.strained_edge_vectors(p, cell, idx, shift, batch, strain)
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        self.assertTrue(p.is_leaf)
        self.assertIsNone(cell.grad_fn)

    def test_production_adapter_selects_new_path_only_with_private_key(self):
        # Execute the actual prepare_graph method without importing e3nn on a
        # CPU-only development host. No replacement implementation is tested.
        adapter_path = path.parents[1] / "models/adapter.py"
        tree = ast.parse(adapter_path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TensorModel")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_graph")
        ns = {
            "torch": torch, "Dict": dict, "Graph": SimpleNamespace,
            "PROPERTY": {"energy": {"requires_grad_with": []}},
            "strained_edge_vectors": module.strained_edge_vectors,
        }
        utils_tree = ast.parse((path.parents[1] / "models/utils.py").read_text(encoding="utf-8"))
        native = next(n for n in utils_tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_symmetric_displacement")
        exec(compile(ast.Module(body=[native, fn], type_ignores=[]), str(adapter_path), "exec"), ns)
        obj = SimpleNamespace(lmp=False, fidelity_idx=0, training=False,
                              readout_fn=SimpleNamespace(),
                              flags=SimpleNamespace(compute_virials=False, compute_stress=True),
                              get_target_property=lambda: ["energy"])
        p, cell, idx, shifts, batch, strain, _ = self.fixture()
        data = dict(positions=p, lattice=cell, edge_index=idx, edge_shifts=shifts,
                    batch=batch, ptr=torch.tensor([0, 2]), node_attrs=torch.ones(2, 1))
        fixed = dict(data, _opt4_edge_strain=strain)
        graph = ns["prepare_graph"](obj, fixed)
        self.assertIs(graph.displacement, strain)
        self.assertIs(fixed["positions"], p)
        self.assertIs(fixed["lattice"], cell)
        graph_native = ns["prepare_graph"](obj, dict(data))
        torch.testing.assert_close(graph.edge_vector, graph_native.edge_vector)
        self.assertIsNot(graph_native.displacement, strain)
        obj.readout_fn.les = object()
        with self.assertRaisesRegex(RuntimeError, "local-potential"):
            ns["prepare_graph"](obj, fixed)


if __name__ == "__main__":
    unittest.main()
