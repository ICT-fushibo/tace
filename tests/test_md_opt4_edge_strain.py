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

density_path = path.with_name("padding_density.py")
density_spec = importlib.util.spec_from_file_location("padding_density_production", density_path)
density_module = importlib.util.module_from_spec(density_spec)
density_spec.loader.exec_module(density_module)


class TACEPaddingDensityTests(unittest.TestCase):
    def test_inside_zero_envelope_keeps_density_and_vjp(self):
        for dtype in (torch.float32, torch.float64):
            density = torch.tensor([[.3], [.5], [.7]], dtype=dtype, requires_grad=True)
            # Real edge 1 has an envelope rounded to zero. Edge 2 is padding.
            cutoff = torch.tensor([[.2], [0.], [0.]], dtype=dtype, requires_grad=True)
            length = torch.tensor([[3.], [4.9999], [12.]], dtype=dtype)
            actual = density_module.apply_opt4_density_mask(
                density, cutoff, length, 5., False)
            torch.testing.assert_close(actual, torch.tensor([[.3], [.5], [0.]], dtype=dtype))
            grad, gc = torch.autograd.grad(actual.sum(), (density, cutoff), allow_unused=True)
            torch.testing.assert_close(grad, torch.tensor([[1.], [1.], [0.]], dtype=dtype))
            self.assertIsNone(gc)

    def test_native_smooth_density_cutoff_derivative_is_preserved(self):
        density = torch.tensor([[.3], [.5], [.7]], dtype=torch.float64, requires_grad=True)
        cutoff = torch.tensor([[.2], [.1], [0.]], dtype=torch.float64, requires_grad=True)
        length = torch.tensor([[3.], [4.9], [12.]], dtype=torch.float64)
        result = density_module.apply_opt4_density_mask(density, cutoff, length, 5., True)
        gd, gc = torch.autograd.grad(result.sum(), (density, cutoff))
        torch.testing.assert_close(gd, cutoff.detach())
        torch.testing.assert_close(gc, torch.tensor([[.3], [.5], [0.]], dtype=torch.float64))
        for applied in (False, True):
            dummy = density_module.apply_opt4_density_mask(
                density, cutoff, torch.full_like(length, 12.), 5., applied)
            torch.testing.assert_close(dummy, torch.zeros_like(dummy))

    def test_setup_is_idempotent_nonpersistent_and_does_not_mutate_native_flag(self):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1))
        model[0].edge_density = torch.nn.Linear(1, 1)
        model[0].apply_density_cutoff = False
        model[0]._opt2_binary_density_mask = True
        self.assertEqual(density_module.enable_opt4_density_masks_(model, 5.), 1)
        ptr = model[0]._opt4_density_padding_cutoff.data_ptr()
        self.assertEqual(density_module.enable_opt4_density_masks_(model, 5.), 0)
        self.assertEqual(model[0]._opt4_density_padding_cutoff.data_ptr(), ptr)
        self.assertTrue(model[0]._opt2_binary_density_mask)
        self.assertFalse(model[0].apply_density_cutoff)
        self.assertFalse(any('padding_cutoff' in k for k in model.state_dict()))

    def test_both_production_interactions_select_opt4_and_preserve_legacy(self):
        source = density_path.parents[1] / 'models/_e3nn/inter.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        for name in ('CgtpInteraction', 'uuSO2Interaction'):
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
            forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
            block = next(n for n in forward.body if isinstance(n, ast.If) and
                         isinstance(n.test, ast.Call) and isinstance(n.test.func, ast.Name) and
                         n.test.func.id == 'hasattr' and n.test.args[1].value == 'edge_density')
            features = torch.tensor([[.3], [.5], [.7]], dtype=torch.float64, requires_grad=True)
            cutoff = torch.tensor([[.2], [0.], [0.]], dtype=torch.float64)
            obj = SimpleNamespace(edge_density=lambda x: x, apply_density_cutoff=False,
                                  _opt2_binary_density_mask=True, alpha=0., beta=1.,
                                  truncate_ghosts=lambda x, n: x)
            env = dict(torch=torch, self=obj, edge_feats=features, cutoff=cutoff,
                       graph=SimpleNamespace(edge_length=torch.tensor([[3.], [4.9], [12.]])),
                       edge_index=torch.tensor([[0, 0, 0], [0, 0, 0]]), nlocal=None,
                       node_attrs_total=torch.zeros(1, 1),
                       scatter_sum=lambda x, *a, **kw: x.sum(dim=0, keepdim=True),
                       apply_opt4_density_mask=density_module.apply_opt4_density_mask)
            code = compile(ast.Module(body=[block], type_ignores=[]), str(source), 'exec')
            exec(code, env)
            torch.testing.assert_close(env['density'], torch.tanh(features[:1]**2))
            obj._opt4_density_padding_cutoff = torch.tensor(5.)
            exec(code, env)
            torch.testing.assert_close(env['density'], torch.tanh(features[:2]**2).sum(0, keepdim=True))


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
