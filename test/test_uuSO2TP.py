################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math

import torch

from tace.models.so2 import uuSO2TensorProduct


def _expanded_components(components_per_m):
    return sum(n if m == 0 else 2 * n for m, n in enumerate(components_per_m))


def _rotate(x, components_per_m, theta):
    out = x.clone()
    offset = components_per_m[0]
    for m in range(1, len(components_per_m)):
        n = components_per_m[m]
        block = out[:, offset:offset + 2 * n].view(x.size(0), 2, n, x.size(2))
        c = math.cos(m * theta)
        s = math.sin(m * theta)
        real = block[:, 0].clone()
        imag = block[:, 1].clone()
        block[:, 0] = c * real - s * imag
        block[:, 1] = s * real + c * imag
        offset += 2 * n
    return out


def test_path_uu_so2_tensor_product_identity():
    components = [16, 9, 4]
    x = torch.randn(5, _expanded_components(components), 7)
    tp = uuSO2TensorProduct(
        mmax=2,
        num_channels=7,
        num_components_per_m=components,
        correlation=1,
    )
    torch.testing.assert_close(tp(x), x)


def test_path_uu_so2_tensor_product_correlation2_shapes_and_grad():
    components = [16, 9, 4]
    x = torch.randn(5, _expanded_components(components), 7, requires_grad=True)
    for weight_type in ("w1_w2", "w1_w1", "w1"):
        tp = uuSO2TensorProduct(
            mmax=2,
            num_channels=7,
            num_components_per_m=components,
            correlation=2,
            weight_type=weight_type,
        )
        out = tp(x)
        assert out.shape == (5, _expanded_components(tp.num_output_components_per_m), 7)
        loss = out.square().mean()
        loss.backward(retain_graph=True)
        assert tp.weight.grad is not None
        assert torch.isfinite(tp.weight.grad).all()
        tp.zero_grad(set_to_none=True)


def test_path_uu_so2_tensor_product_equivariance():
    torch.manual_seed(0)
    components = [4, 3, 2]
    theta = 0.37
    x = torch.randn(3, _expanded_components(components), 5)
    for weight_type in ("w1_w2", "w1_w1", "w1"):
        tp = uuSO2TensorProduct(
            mmax=2,
            num_channels=5,
            num_components_per_m=components,
            correlation=2,
            weight_type=weight_type,
        )
        out1 = tp(_rotate(x, components, theta))
        out2 = _rotate(tp(x), tp.num_output_components_per_m, theta)
        torch.testing.assert_close(out1, out2, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_path_uu_so2_tensor_product_identity()
    test_path_uu_so2_tensor_product_correlation2_shapes_and_grad()
    test_path_uu_so2_tensor_product_equivariance()
