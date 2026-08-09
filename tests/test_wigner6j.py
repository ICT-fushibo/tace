import pytest
import torch
from e3nn import o3

from tace.models._e3nn.fused import O3ScatterTensorProduct
from tace.models._e3nn.wigner6j import (
    O3Wigner6jScatterTensorProduct,
    sympy_wigner_6j,
    wigner_6j,
)


def test_standard_wigner_6j_symbol():
    assert sympy_wigner_6j(1, 1, 1, 1, 1, 1) == pytest.approx(1.0 / 6.0)
    assert wigner_6j(1, 1, 1, 1, 1, 1) == pytest.approx(1.0 / 2.0)


def _build_tensor_product():
    irreps_node = o3.Irreps("2x0e + 2x1o + 2x1e")
    irreps_edge = o3.Irreps.spherical_harmonics(2, p=-1)
    irreps_out = o3.Irreps("2x0e + 2x0o + 2x1e + 2x1o + 2x2e + 2x2o")
    module = O3Wigner6jScatterTensorProduct(
        irreps_node,
        irreps_edge,
        irreps_out,
        magnetic_irreps=o3.Irreps("1x1e"),
    )
    assert isinstance(module.recoupled_pos_tp, O3ScatterTensorProduct)
    return module


def _random_inputs(module, *, requires_grad=False):
    num_nodes = 5
    num_edges = 11
    edge_index = torch.stack(
        [
            torch.randint(num_nodes, (num_edges,)),
            torch.randint(num_nodes, (num_edges,)),
        ]
    )
    node_feats = torch.randn(
        num_nodes,
        module.irreps_node.dim,
        requires_grad=requires_grad,
    )
    edge_attrs = torch.randn(
        num_edges,
        module.irreps_edge.dim,
        requires_grad=requires_grad,
    )
    magnetic_moments = torch.randn(
        num_nodes,
        3,
        requires_grad=requires_grad,
    )
    radial_weights = torch.randn(
        num_edges,
        module.radial_weight_numel,
        requires_grad=requires_grad,
    )
    magnetic_weights = torch.randn(
        num_nodes,
        module.magnetic_weight_numel,
        requires_grad=requires_grad,
    )
    return (
        node_feats,
        edge_attrs,
        magnetic_moments,
        radial_weights,
        magnetic_weights,
        edge_index,
    )


def test_wigner6j_recoupling_matches_position_first_and_gradients():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    module = _build_tensor_product()
    inputs = _random_inputs(module, requires_grad=True)

    recoupled = module(*inputs)
    reference = module.forward_reference(*inputs)
    torch.testing.assert_close(recoupled, reference, atol=2.0e-12, rtol=2.0e-12)

    grad_output = torch.randn_like(recoupled)
    differentiable_inputs = inputs[:-1]
    recoupled_grads = torch.autograd.grad(
        (recoupled * grad_output).sum(),
        differentiable_inputs,
        retain_graph=True,
    )
    reference_grads = torch.autograd.grad(
        (reference * grad_output).sum(),
        differentiable_inputs,
    )
    for recoupled_grad, reference_grad in zip(recoupled_grads, reference_grads):
        torch.testing.assert_close(
            recoupled_grad,
            reference_grad,
            atol=3.0e-12,
            rtol=3.0e-12,
        )


@pytest.mark.parametrize("improper", [False, True])
def test_wigner6j_magnetic_tensor_product_is_o3_equivariant(improper):
    torch.manual_seed(1)
    torch.set_default_dtype(torch.float64)
    module = _build_tensor_product()
    inputs = _random_inputs(module)
    node_feats, edge_attrs, magnetic_moments, radial_weights, magnetic_weights, _ = (
        inputs
    )

    rotation = o3.rand_matrix(dtype=torch.float64)
    if improper:
        rotation = -rotation
    rotated_inputs = (
        node_feats @ module.irreps_node.D_from_matrix(rotation).T,
        edge_attrs @ module.irreps_edge.D_from_matrix(rotation).T,
        magnetic_moments @ module.magnetic_irreps.D_from_matrix(rotation).T,
        radial_weights,
        magnetic_weights,
        inputs[-1],
    )

    output = module(*inputs)
    rotated_output = module(*rotated_inputs)
    expected = output @ module.irreps_out.D_from_matrix(rotation).T
    torch.testing.assert_close(
        rotated_output,
        expected,
        atol=3.0e-11,
        rtol=3.0e-11,
    )
