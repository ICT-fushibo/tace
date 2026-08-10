import pytest
import torch

from tace.models import o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def test_o2_linear_is_exported_without_prefixed_class_name():
    assert o2.Linear.__name__ == "Linear"
    assert o2.Irrep.__name__ == "Irrep"
    assert o2.Irreps.__name__ == "Irreps"


def test_o2_linear_defaults_to_uv_path_mode():
    module = o2.Linear(
        "0e+1m",
        "0e+1m",
        channels_in=2,
        channels_out=3,
    )
    assert module.path_mode == "uv"
    assert module.weight_shape == (2, 2, 3)


@pytest.mark.parametrize("reflected", [False, True])
@pytest.mark.parametrize("path_mode", ["uv", "uu"])
def test_o2_linear_is_equivariant_with_complete_irreps(reflected, path_mode):
    torch.manual_seed(0)
    irreps_in = o2.Irreps("2x0e+0o+2x1m+2m")
    irreps_out = o2.Irreps("0e+2x0o+1m+2x2m")
    module = o2.Linear(
        irreps_in,
        irreps_out,
        channels_in=2,
        channels_out=2 if path_mode == "uu" else 3,
        path_mode=path_mode,
        bias=True,
    ).to(device=DEVICE, dtype=DTYPE)
    input = torch.randn(5, irreps_in.dim, 2, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.41, dtype=DTYPE, device=DEVICE)
    input_transform = o2.o2_irreps_representation(
        irreps_in,
        angle,
        reflected,
    )
    output_transform = o2.o2_irreps_representation(
        irreps_out,
        angle,
        reflected,
    )

    transformed_input = torch.einsum("ij,bjc->bic", input_transform, input)
    actual = module(transformed_input)
    expected = torch.einsum("ij,bjc->bic", output_transform, module(input))
    torch.testing.assert_close(actual, expected)


def test_o2_linear_positive_order_components_share_one_real_matrix():
    module = o2.Linear(
        "1m",
        "1m",
        channels_in=2,
        channels_out=3,
        internal_weights=False,
        bias=False,
        path_norm=False,
    ).to(device=DEVICE, dtype=DTYPE)
    input = torch.randn(4, 2, 2, dtype=DTYPE, device=DEVICE)
    weight = torch.randn(*module.weight_shape, dtype=DTYPE, device=DEVICE)
    actual = module(input, weight)
    expected = torch.matmul(input, weight[0])
    torch.testing.assert_close(actual, expected)


def test_o2_linear_uu_is_channelwise_and_sums_paths():
    module = o2.Linear(
        "2x0e",
        "0e",
        channels_in=2,
        path_mode="uu",
        internal_weights=False,
        bias=False,
        path_norm=False,
    ).to(device=DEVICE, dtype=DTYPE)
    input = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]],
        dtype=DTYPE,
        device=DEVICE,
    )
    weight = torch.tensor(
        [[2.0, 3.0], [5.0, 7.0]],
        dtype=DTYPE,
        device=DEVICE,
    )
    actual = module(input, weight)
    expected = input[:, :1] * weight[0] + input[:, 1:] * weight[1]
    torch.testing.assert_close(actual, expected)


def test_o2_linear_uu_requires_matching_channels():
    with pytest.raises(ValueError, match="channels_in == channels_out"):
        o2.Linear(
            "0e",
            "0e",
            channels_in=2,
            channels_out=3,
            path_mode="uu",
        )


def test_o2_linear_rejects_unknown_path_mode():
    with pytest.raises(ValueError, match="path_mode"):
        o2.Linear("0e", "0e", channels_in=1, path_mode="invalid")


@pytest.mark.parametrize("path_mode", ["uv", "uu"])
def test_o2_linear_supports_batched_external_weights(path_mode):
    torch.manual_seed(1)
    module = o2.Linear(
        "2x0e+0o+2x1m",
        "0e+2x0o+1m",
        channels_in=2,
        channels_out=2 if path_mode == "uu" else 3,
        path_mode=path_mode,
        internal_weights=False,
        bias=False,
    ).to(device=DEVICE, dtype=DTYPE)
    input = torch.randn(
        4,
        module.irreps_in.dim,
        module.channels_in,
        dtype=DTYPE,
        device=DEVICE,
    )
    weight = torch.randn(
        4,
        *module.weight_shape,
        dtype=DTYPE,
        device=DEVICE,
    )
    actual = module(input, weight)
    expected = torch.stack(
        [module(input[index], weight[index]) for index in range(input.shape[0])]
    )
    torch.testing.assert_close(actual, expected)


def test_o2_linear_bias_is_restricted_to_even_scalars():
    module = o2.Linear(
        "0e",
        "0e+0o+1m",
        channels_in=1,
        channels_out=2,
        bias=True,
    ).to(device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        module.bias.copy_(torch.tensor([[2.0, -3.0]], device=DEVICE, dtype=DTYPE))
    input = torch.zeros(3, 1, 1, dtype=DTYPE, device=DEVICE)
    output = module(input)
    torch.testing.assert_close(
        output[:, 0],
        torch.tensor([2.0, -3.0], dtype=DTYPE, device=DEVICE).expand(3, -1),
    )
    torch.testing.assert_close(output[:, 1:], torch.zeros_like(output[:, 1:]))


@pytest.mark.parametrize("path_mode", ["uv", "uu"])
def test_o2_linear_external_weight_gradcheck_and_gradgradcheck(path_mode):
    module = o2.Linear(
        "0e+0o+1m",
        "2x0e+0o+1m",
        channels_in=2,
        channels_out=2,
        path_mode=path_mode,
        internal_weights=False,
        bias=False,
    ).to(device=DEVICE, dtype=DTYPE)
    input = torch.randn(
        2,
        module.irreps_in.dim,
        module.channels_in,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    weight = torch.randn(
        *module.weight_shape,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )

    def function(features, weights):
        return module(features, weights)

    assert torch.autograd.gradcheck(function, (input, weight), fast_mode=True)
    assert torch.autograd.gradgradcheck(function, (input, weight), fast_mode=True)


@pytest.mark.parametrize("path_mode", ["uv", "uu"])
def test_o2_linear_zero_pads_missing_output_irreps(path_mode):
    module = o2.Linear(
        "0e+1m",
        "0e+0o+2m",
        channels_in=2,
        channels_out=2,
        path_mode=path_mode,
        internal_weights=False,
        bias=False,
        path_norm=False,
    ).to(device=DEVICE, dtype=DTYPE)
    input = torch.randn(
        3,
        module.irreps_in.dim,
        module.channels_in,
        dtype=DTYPE,
        device=DEVICE,
    )
    weight = torch.randn(*module.weight_shape, dtype=DTYPE, device=DEVICE)
    output = module(input, weight)

    if path_mode == "uv":
        expected_scalar = torch.matmul(input[:, :1], weight[0])
    else:
        expected_scalar = input[:, :1] * weight[0]
    torch.testing.assert_close(output[:, :1], expected_scalar)
    torch.testing.assert_close(output[:, 1:], torch.zeros_like(output[:, 1:]))


def test_o2_linear_rejects_non_equivariant_path():
    with pytest.raises(ValueError, match="identical irreps"):
        o2.Linear(
            "0e",
            "0o",
            channels_in=1,
            path=[(0, 0)],
        )


def test_o2_linear_without_paths_is_zero_and_differentiable():
    module = o2.Linear(
        "0e",
        "0o",
        channels_in=2,
        internal_weights=False,
        bias=False,
    ).to(device=DEVICE, dtype=DTYPE)
    input = torch.randn(
        3,
        1,
        2,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    weight = torch.empty(
        *module.weight_shape,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    output = module(input, weight)
    torch.testing.assert_close(output, torch.zeros_like(output))
    gradients = torch.autograd.grad(output.sum(), (input, weight))
    torch.testing.assert_close(gradients[0], torch.zeros_like(input))
    assert gradients[1].numel() == 0


def test_o2_linear_rejects_complex_input_and_weight():
    internal = o2.Linear("1m", "1m", channels_in=1)
    with pytest.raises(TypeError, match="real inputs"):
        internal(torch.ones(2, 1, dtype=torch.complex64))

    external = o2.Linear(
        "1m",
        "1m",
        channels_in=1,
        internal_weights=False,
    )
    with pytest.raises(TypeError, match="real weights"):
        external(
            torch.ones(2, 1),
            torch.ones(*external.weight_shape, dtype=torch.complex64),
        )
