import pytest
import torch
from e3nn import o3

pytest.importorskip("cuequivariance")
pytest.importorskip("cuequivariance_torch")

from tace.models._cue import e3nnCueTensorProduct
from tace.models._e3nn.fused import uuuTensorProduct
from tace.models._e3nn.paths import generate_paths


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUEQ requires CUDA")
@pytest.mark.parametrize("trainable", [False, True])
def test_cue_uuu_matches_e3nn_forward_and_backward(trainable):
    irreps_in1 = o3.Irreps("4x0e + 4x1o + 4x2e + 4x3o")
    irreps_in2 = o3.Irreps("4x0e + 4x1o + 4x2e + 4x3o + 4x0e")
    irreps_out = o3.Irreps("4x0e + 4x1o + 4x2e + 4x3o")
    instructions, actual_irreps_out = generate_paths(
        irreps_out,
        irreps_in1,
        irreps_in2,
        l1l2="<=",
        e3nn_mode="uuu",
        trainable=trainable,
    )
    reference = o3.TensorProduct(
        irreps_in1,
        irreps_in2,
        actual_irreps_out,
        instructions,
        shared_weights=False,
        internal_weights=False,
    ).cuda()
    accelerated = e3nnCueTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        l1l2="<=",
        trainable=trainable,
    ).cuda()

    torch.manual_seed(0)
    lhs = torch.randn(8, irreps_in1.dim, device="cuda")
    rhs = torch.randn(8, irreps_in2.dim, device="cuda")
    weights = torch.randn(8, reference.weight_numel, device="cuda")
    probe = torch.randn(8, actual_irreps_out.dim, device="cuda")

    def evaluate(module, pass_weights, differentiate_weights):
        x = lhs.detach().clone().requires_grad_()
        y = rhs.detach().clone().requires_grad_()
        w = weights.detach().clone().requires_grad_()
        output = module(x, y, w if pass_weights else None)
        inputs = (x, y, w) if differentiate_weights else (x, y)
        gradients = torch.autograd.grad((output * probe).sum(), inputs)
        return output, gradients

    expected, expected_gradients = evaluate(reference, True, trainable)
    actual, actual_gradients = evaluate(
        accelerated,
        trainable,
        trainable,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-6)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=2e-5,
            atol=5e-6,
        )

    if not trainable:
        assert accelerated.weight_numel == 0
        assert not hasattr(accelerated, "unit_weights")
        assert "unit_weights" not in accelerated.state_dict()


def test_weighted_uuu_uses_e3nn_when_cue_is_enabled(monkeypatch):
    monkeypatch.setenv("TACE_USE_CUE", "1")
    irreps = o3.Irreps("2x0e + 2x1o")
    tensor_product = uuuTensorProduct(
        irreps,
        irreps,
        irreps,
        trainable=True,
    )

    assert tensor_product.weight_numel > 0
    assert not hasattr(tensor_product, "fused_tp")
