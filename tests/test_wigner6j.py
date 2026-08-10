import pytest
import torch
from e3nn import o3

from tace.models._e3nn.fused import O3ScatterTensorProduct
from tace.models._e3nn.inter import O3Wigner6jMagneticInteraction
from tace.models._e3nn.wigner6j import (
    O3Wigner6jScatterTensorProduct,
    sympy_wigner_6j,
    wigner_6j,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_standard_wigner_6j_symbol():
    assert sympy_wigner_6j(1, 1, 1, 1, 1, 1) == pytest.approx(1.0 / 6.0)
    assert wigner_6j(1, 1, 1, 1, 1, 1) == pytest.approx(1.0 / 2.0)


def _build_tensor_product(*, magnetic_weight_level="edge"):
    irreps_node = o3.Irreps("2x0e + 2x1o + 2x1e")
    irreps_edge = o3.Irreps.spherical_harmonics(2, p=-1)
    irreps_out = o3.Irreps("2x0e + 2x0o + 2x1e + 2x1o + 2x2e + 2x2o")
    module = O3Wigner6jScatterTensorProduct(
        irreps_node,
        irreps_edge,
        irreps_out,
        magnetic_irreps=o3.Irreps("1x1e"),
        magnetic_weight_level=magnetic_weight_level,
    )
    assert isinstance(module.recoupled_pos_tp, O3ScatterTensorProduct)
    return module.to(DEVICE)


def _random_inputs(module, *, requires_grad=False):
    num_nodes = 5
    num_edges = 11
    edge_index = torch.stack(
        [
            torch.randint(num_nodes, (num_edges,), device=DEVICE),
            torch.randint(num_nodes, (num_edges,), device=DEVICE),
        ]
    )
    node_feats = torch.randn(
        num_nodes,
        module.irreps_node.dim,
        device=DEVICE,
        requires_grad=requires_grad,
    )
    edge_attrs = torch.randn(
        num_edges,
        module.irreps_edge.dim,
        device=DEVICE,
        requires_grad=requires_grad,
    )
    magnetic_moments = torch.randn(
        num_nodes,
        3,
        device=DEVICE,
        requires_grad=requires_grad,
    )
    radial_weights = torch.randn(
        num_edges,
        module.radial_weight_numel,
        device=DEVICE,
        requires_grad=requires_grad,
    )
    num_magnetic_weights = (
        num_edges if module.magnetic_weight_level == "edge" else num_nodes
    )
    magnetic_weights = torch.randn(
        num_magnetic_weights,
        module.magnetic_weight_numel,
        device=DEVICE,
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


@pytest.mark.parametrize("magnetic_weight_level", ["edge", "node"])
def test_wigner6j_recoupling_matches_position_first_and_gradients(
    magnetic_weight_level,
):
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    module = _build_tensor_product(magnetic_weight_level=magnetic_weight_level)
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


@pytest.mark.parametrize("magnetic_weight_level", ["edge", "node"])
@pytest.mark.parametrize("improper", [False, True])
def test_wigner6j_magnetic_tensor_product_is_o3_equivariant(
    improper,
    magnetic_weight_level,
):
    torch.manual_seed(1)
    torch.set_default_dtype(torch.float64)
    module = _build_tensor_product(magnetic_weight_level=magnetic_weight_level)
    inputs = _random_inputs(module)
    node_feats, edge_attrs, magnetic_moments, radial_weights, magnetic_weights, _ = (
        inputs
    )

    rotation = o3.rand_matrix(dtype=torch.float64)
    if improper:
        rotation = -rotation
    node_rotation = module.irreps_node.D_from_matrix(rotation).to(DEVICE)
    edge_rotation = module.irreps_edge.D_from_matrix(rotation).to(DEVICE)
    magnetic_rotation = module.magnetic_irreps.D_from_matrix(rotation).to(DEVICE)
    output_rotation = module.irreps_out.D_from_matrix(rotation).to(DEVICE)
    rotated_inputs = (
        node_feats @ node_rotation.T,
        edge_attrs @ edge_rotation.T,
        magnetic_moments @ magnetic_rotation.T,
        radial_weights,
        magnetic_weights,
        inputs[-1],
    )

    output = module(*inputs)
    rotated_output = module(*rotated_inputs)
    expected = output @ output_rotation.T
    torch.testing.assert_close(
        rotated_output,
        expected,
        atol=3.0e-11,
        rtol=3.0e-11,
    )


def test_wigner6j_uses_edge_magnetic_weights_by_default():
    module = _build_tensor_product()
    assert module.magnetic_weight_level == "edge"


def test_wigner6j_rejects_unknown_magnetic_weight_level():
    with pytest.raises(ValueError, match="magnetic_weight_level"):
        _build_tensor_product(magnetic_weight_level="graph")


def _build_interaction():
    module = O3Wigner6jMagneticInteraction(
        layer=0,
        num_layers=1,
        num_elements=2,
        avg_num_neighbors=4.0,
        mmax=2,
        Lmax=2,
        lmax=2,
        correlation=[1],
        num_channel=2,
        use_temperature=False,
        edge_feats_channel=4,
        target_irreps=o3.Irreps("0e"),
        num_radial_basis=4,
        radial_mlp=[8],
        radial_bias=True,
        irreps_in=o3.Irreps("2x0e + 2x1o"),
        scalar_act=None,
        tensor_act=None,
        edge_ace_hidden=None,
        parity=True,
        nonlinear=None,
    )
    return module.to(DEVICE)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"magnetic_weight_level": "node"},
        {"use_magnetic_edge_features": False},
        {"use_message_magnetic_tensor_product": False},
        {
            "use_message_magnetic_tensor_product": False,
            "use_magnetic_edge_features": False,
        },
    ],
)
def test_wigner6j_interaction_magnetic_options(options, monkeypatch):
    torch.manual_seed(2)
    for name, value in options.items():
        monkeypatch.setattr(O3Wigner6jMagneticInteraction, name, value)
    module = _build_interaction()
    num_nodes = 5
    num_edges = 9
    edge_index = torch.stack(
        [
            torch.randint(num_nodes, (num_edges,), device=DEVICE),
            torch.randint(num_nodes, (num_edges,), device=DEVICE),
        ]
    )
    node_feats = torch.randn(
        num_nodes,
        module.irreps_in.dim,
        device=DEVICE,
        requires_grad=True,
    )
    node_attrs = torch.nn.functional.one_hot(
        torch.randint(2, (num_nodes,), device=DEVICE),
        2,
    ).to(node_feats)
    edge_feats = torch.randn(num_edges, 4, device=DEVICE, requires_grad=True)
    edge_attrs = torch.randn(
        num_edges,
        module.irreps_sh.dim,
        device=DEVICE,
        requires_grad=True,
    )
    magnetic_moments = torch.randn(
        num_nodes,
        3,
        device=DEVICE,
        requires_grad=True,
    )
    if (
        not module.use_message_magnetic_tensor_product
        and not module.use_magnetic_edge_features
    ):
        magnetic_moments = None

    output = module._compute_messages(
        node_feats,
        node_attrs,
        edge_feats,
        edge_attrs,
        edge_index,
        torch.rand(num_edges, 1, device=DEVICE),
        magnetic_moments,
    )
    assert output.shape == (num_nodes, module.rejector.irreps_out.dim)
    output.square().sum().backward()
