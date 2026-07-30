import torch
from e3nn import o3

from tace.models._e3nn.fused import uuuTensorProduct
from tace.models._e3nn.nonlinear import get_nonlinear_layer
from tace.models._e3nn.paths import generate_paths


def test_odd_scalars_use_even_gates_and_preserve_o3_equivariance():
    irreps = o3.Irreps("2x0e+2x0o+2x1o+2x1e")

    nonlinearity, linear, _ = get_nonlinear_layer(
        nonlinear_type="gate",
        irreps_in=irreps,
        irreps_out=irreps,
        gate_m0=False,
    )
    nonlinearity = nonlinearity.double()
    linear = linear.double()

    assert nonlinearity.irreps_out == irreps
    inputs = torch.randn(
        4,
        nonlinearity.irreps_in.dim,
        dtype=torch.float64,
    )
    outputs = linear(nonlinearity(inputs))
    rotation = o3.rand_matrix(dtype=torch.float64)
    inversion = -torch.eye(3, dtype=torch.float64)

    for matrix in (rotation, inversion):
        input_transform = nonlinearity.irreps_in.D_from_matrix(matrix)
        output_transform = irreps.D_from_matrix(matrix)
        transformed = linear(nonlinearity(inputs @ input_transform.T))
        expected = outputs @ output_transform.T
        torch.testing.assert_close(
            transformed,
            expected,
            atol=1e-10,
            rtol=1e-10,
        )


def test_identical_inputs_remove_only_zero_paths():
    irreps = o3.Irreps("2x0e+2x1o+2x1e+2x2e")
    irreps_out = o3.Irreps("0e+0o+1e+1o+2e+2o")
    full_paths, full_actual_irreps = generate_paths(
        irreps_out,
        irreps,
        irreps,
        e3nn_mode="uuu",
    )
    pruned_paths, pruned_actual_irreps = generate_paths(
        irreps_out,
        irreps,
        irreps,
        e3nn_mode="uuu",
        identical_inputs=True,
    )

    def signature(path, actual_irreps):
        i, j, k, mode, trainable = path
        return i, j, str(actual_irreps[k].ir), mode, trainable

    pruned_signatures = {
        signature(path, pruned_actual_irreps) for path in pruned_paths
    }
    removed = [
        path
        for path in full_paths
        if signature(path, full_actual_irreps) not in pruned_signatures
    ]
    assert removed
    for i, j, k, _, _ in removed:
        ir1 = irreps[i].ir
        ir2 = irreps[j].ir
        ir_out = full_actual_irreps[k].ir
        assert i == j
        assert (ir1.l + ir2.l - ir_out.l) % 2 == 1

    tensor_product = uuuTensorProduct(
        irreps,
        irreps,
        irreps_out,
        identical_inputs=False,
    ).double()
    inputs = torch.randn(4, irreps.dim, dtype=torch.float64)
    outputs = tensor_product(inputs, inputs)
    output_slices = tensor_product.irreps_out.slices()
    for path in removed:
        full_index = full_paths.index(path)
        assert outputs[:, output_slices[full_index]].abs().max() < 1e-12


def test_parity_false_paths_are_unchanged():
    irreps = o3.Irreps("2x0e+2x1o+2x2e+2x3o")
    irreps_out = o3.Irreps("0e+1o+2e")
    nonlinearity, linear, _ = get_nonlinear_layer(
        nonlinear_type="gate",
        irreps_in=irreps,
        irreps_out=irreps,
        gate_m0=False,
    )
    full_paths, full_irreps = generate_paths(
        irreps_out,
        irreps,
        irreps,
        e3nn_mode="uuu",
    )
    pruned_paths, pruned_irreps = generate_paths(
        irreps_out,
        irreps,
        irreps,
        e3nn_mode="uuu",
        identical_inputs=True,
    )

    assert nonlinearity.irreps_out == irreps
    assert linear.irreps_in == irreps
    assert pruned_paths == full_paths
    assert pruned_irreps == full_irreps
