import math

import pytest
import torch

from tace.models.o2 import (
    Irrep,
    Irreps,
    fully_connected_tensor_product_paths,
    has_tensor_product_path,
    o2_clebsch_gordan,
    o2_irreps_representation,
    o2_recoupling,
    o2_representation,
    restrict_o3_irrep,
    restrict_o3_irreps,
    tensor_product_irrep,
    tensor_product_irreps,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def _contract(input1, input2, cg):
    return torch.einsum("...i,...j,ijk->...k", input1, input2, cg)


def _coupling_tensors(input1, input2, input3, output):
    recoupling = o2_recoupling(
        input1,
        input2,
        input3,
        output,
        dtype=DTYPE,
        device=DEVICE,
    )
    left = []
    for intermediate in recoupling.left_intermediates:
        first = o2_clebsch_gordan(
            input1,
            input2,
            intermediate,
            dtype=DTYPE,
            device=DEVICE,
        )
        second = o2_clebsch_gordan(
            intermediate,
            input3,
            output,
            dtype=DTYPE,
            device=DEVICE,
        )
        left.append(torch.einsum("abe,ecd->abcd", first, second))
    right = []
    for intermediate in recoupling.right_intermediates:
        first = o2_clebsch_gordan(
            input2,
            input3,
            intermediate,
            dtype=DTYPE,
            device=DEVICE,
        )
        second = o2_clebsch_gordan(
            input1,
            intermediate,
            output,
            dtype=DTYPE,
            device=DEVICE,
        )
        right.append(torch.einsum("bcf,afd->abcd", first, second))
    return recoupling, torch.stack(left), torch.stack(right)


def test_o2_irrep_and_irreps_metadata():
    assert Irrep("0e") == Irrep(0, 1)
    assert Irrep("0o") == Irrep((0, -1))
    assert Irrep("3m").dim == 2

    irreps = Irreps("2x0e + 0o + 3x1m + 2m")
    assert irreps.dim == 2 + 1 + 6 + 2
    assert irreps.num_irreps == 7
    assert irreps.m_max == 2
    assert irreps.expanded() == (
        Irrep("0e"),
        Irrep("0e"),
        Irrep("0o"),
        Irrep("1m"),
        Irrep("1m"),
        Irrep("1m"),
        Irrep("2m"),
    )
    assert irreps.expanded_slices()[-1] == slice(9, 11)
    with pytest.raises(AttributeError, match="immutable"):
        irreps._groups = ()


@pytest.mark.parametrize("value", ["0m", "1e", "1o", "-1m", "x"])
def test_o2_irrep_rejects_invalid_labels(value):
    with pytest.raises((TypeError, ValueError)):
        Irrep(value)


def test_o3_restriction_is_complete_and_parity_aware():
    assert restrict_o3_irrep(1, "o") == Irreps("0e+1m")
    assert restrict_o3_irrep(1, "e") == Irreps("0o+1m")
    assert restrict_o3_irrep(2, "e") == Irreps("0e+1m+2m")
    assert restrict_o3_irrep(2, "o") == Irreps("0o+1m+2m")
    assert restrict_o3_irreps([(2, 1, "e"), (1, 0, "o")]) == Irreps("2x0o+2x1m+0o")


def test_o2_direct_sum_representation_uses_complete_layout():
    irreps = Irreps("0e+0o+2x1m")
    angle = torch.tensor([0.2, -0.7], dtype=DTYPE, device=DEVICE)
    representation = o2_irreps_representation(irreps, angle, True)
    assert representation.shape == (2, irreps.dim, irreps.dim)
    torch.testing.assert_close(representation[:, 0, 0], torch.ones_like(angle))
    torch.testing.assert_close(representation[:, 1, 1], -torch.ones_like(angle))
    expected = o2_representation("1m", angle, True)
    torch.testing.assert_close(representation[:, 2:4, 2:4], expected)
    torch.testing.assert_close(representation[:, 4:6, 4:6], expected)


def test_complete_o2_tensor_product_rules_and_paths():
    assert tensor_product_irrep("0o", "0o") == (Irrep("0e"),)
    assert tensor_product_irrep("0o", "2m") == (Irrep("2m"),)
    assert tensor_product_irrep("1m", "1m") == (
        Irrep("0e"),
        Irrep("0o"),
        Irrep("2m"),
    )
    assert tensor_product_irrep("1m", "2m") == (
        Irrep("1m"),
        Irrep("3m"),
    )
    assert tensor_product_irreps("2x1m", "0o") == Irreps("2x1m")
    assert has_tensor_product_path("0o", "1m", "1m")
    assert not has_tensor_product_path("0o", "1m", "2m")

    paths = fully_connected_tensor_product_paths("0e+0o", "2x1m", "1m")
    assert len(paths) == 4


def test_all_small_o2_cg_paths_are_normalized_and_equivariant():
    torch.manual_seed(0)
    irreps = [Irrep("0e"), Irrep("0o")]
    irreps.extend(Irrep(m, "m") for m in range(1, 4))
    angle = torch.tensor(0.37, dtype=DTYPE, device=DEVICE)

    for input1 in irreps:
        for input2 in irreps:
            for output in tensor_product_irrep(input1, input2):
                cg = o2_clebsch_gordan(
                    input1,
                    input2,
                    output,
                    dtype=DTYPE,
                    device=DEVICE,
                )
                gram = torch.einsum("ijk,ijl->kl", cg, cg)
                torch.testing.assert_close(
                    gram,
                    torch.eye(output.dim, dtype=DTYPE, device=DEVICE),
                )

                value1 = torch.randn(5, input1.dim, dtype=DTYPE, device=DEVICE)
                value2 = torch.randn(5, input2.dim, dtype=DTYPE, device=DEVICE)
                value_out = _contract(value1, value2, cg)
                for reflected in (False, True):
                    transform1 = o2_representation(input1, angle, reflected)
                    transform2 = o2_representation(input2, angle, reflected)
                    transform_out = o2_representation(output, angle, reflected)
                    transformed1 = torch.einsum("ij,bj->bi", transform1, value1)
                    transformed2 = torch.einsum("ij,bj->bi", transform2, value2)
                    actual = _contract(transformed1, transformed2, cg)
                    expected = torch.einsum(
                        "ij,bj->bi",
                        transform_out,
                        value_out,
                    )
                    torch.testing.assert_close(actual, expected)


def test_o2_recoupling_is_nontrivial_and_exact():
    recoupling, left, right = _coupling_tensors("1m", "1m", "1m", "1m")
    assert recoupling.left_intermediates == (
        Irrep("0e"),
        Irrep("0o"),
        Irrep("2m"),
    )
    assert recoupling.right_intermediates == recoupling.left_intermediates

    expected_matrix = torch.tensor(
        [
            [0.5, 0.5, math.sqrt(0.5)],
            [-0.5, -0.5, math.sqrt(0.5)],
            [math.sqrt(0.5), -math.sqrt(0.5), 0.0],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    torch.testing.assert_close(recoupling.matrix, expected_matrix)
    torch.testing.assert_close(
        recoupling.matrix @ recoupling.matrix.T,
        torch.eye(3, dtype=DTYPE, device=DEVICE),
    )
    torch.testing.assert_close(
        left, torch.einsum("ij,jabcd->iabcd", recoupling.matrix, right)
    )


def test_all_small_o2_recoupling_matrices_are_orthogonal():
    irreps = [Irrep("0e"), Irrep("0o"), Irrep("1m"), Irrep("2m")]
    for input1 in irreps:
        for input2 in irreps:
            for input3 in irreps:
                left_product = tensor_product_irreps(input1, input2)
                outputs = tensor_product_irreps(left_product, input3).regroup()
                for _, output in outputs:
                    recoupling = o2_recoupling(
                        input1,
                        input2,
                        input3,
                        output,
                        dtype=DTYPE,
                        device=DEVICE,
                    )
                    identity = torch.eye(
                        recoupling.matrix.shape[0],
                        dtype=DTYPE,
                        device=DEVICE,
                    )
                    torch.testing.assert_close(
                        recoupling.matrix @ recoupling.matrix.T,
                        identity,
                    )
