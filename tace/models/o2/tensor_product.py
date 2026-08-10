"""Complete real O(2) tensor-product metadata and recoupling tools."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .irreps import (
    Irrep,
    IrrepLike,
    Irreps,
    IrrepsLike,
    check_o2_irrep,
    check_o2_irreps,
)


def tensor_product_irrep(
    input1: IrrepLike,
    input2: IrrepLike,
) -> Tuple[Irrep, ...]:
    """Decompose the tensor product of two complete real O(2) irreps."""
    input1 = check_o2_irrep(input1)
    input2 = check_o2_irrep(input2)

    if input1.m == 0 and input2.m == 0:
        return (Irrep(0, input1.p * input2.p),)
    if input1.m == 0:
        return (input2,)
    if input2.m == 0:
        return (input1,)
    if input1.m == input2.m:
        return (
            Irrep("0e"),
            Irrep("0o"),
            Irrep(input1.m + input2.m, "m"),
        )
    return (
        Irrep(abs(input1.m - input2.m), "m"),
        Irrep(input1.m + input2.m, "m"),
    )


def tensor_product_irreps(
    input1: IrrepsLike,
    input2: IrrepsLike,
    *,
    regroup: bool = True,
) -> Irreps:
    """Decompose a direct-sum tensor product, including multiplicities."""
    input1 = check_o2_irreps(input1)
    input2 = check_o2_irreps(input2)
    groups = []
    for multiplicity1, irrep1 in input1:
        for multiplicity2, irrep2 in input2:
            multiplicity = multiplicity1 * multiplicity2
            groups.extend(
                (multiplicity, output)
                for output in tensor_product_irrep(irrep1, irrep2)
            )
    output = Irreps(groups)
    return output.regroup() if regroup else output


def has_tensor_product_path(
    output: IrrepLike,
    input1: IrrepLike,
    input2: IrrepLike,
) -> bool:
    output = check_o2_irrep(output)
    return output in tensor_product_irrep(input1, input2)


@dataclass(frozen=True)
class O2TensorProductPath:
    output: int
    input1: int
    input2: int


def fully_connected_tensor_product_paths(
    irreps_out: IrrepsLike,
    irreps_in1: IrrepsLike,
    irreps_in2: IrrepsLike,
) -> Tuple[O2TensorProductPath, ...]:
    """Enumerate paths using indices into expanded irrep-copy layouts."""
    outputs = check_o2_irreps(irreps_out).expanded()
    inputs1 = check_o2_irreps(irreps_in1).expanded()
    inputs2 = check_o2_irreps(irreps_in2).expanded()
    return tuple(
        O2TensorProductPath(output_index, input1_index, input2_index)
        for output_index, output in enumerate(outputs)
        for input1_index, input1 in enumerate(inputs1)
        for input2_index, input2 in enumerate(inputs2)
        if has_tensor_product_path(output, input1, input2)
    )


def _new_cg_tensor(
    input1: Irrep,
    input2: Irrep,
    output: Irrep,
    dtype: Optional[torch.dtype],
    device: Optional[torch.device],
) -> torch.Tensor:
    dtype = torch.get_default_dtype() if dtype is None else dtype
    return torch.zeros(
        input1.dim,
        input2.dim,
        output.dim,
        dtype=dtype,
        device=device,
    )


def o2_clebsch_gordan(
    input1: IrrepLike,
    input2: IrrepLike,
    output: IrrepLike,
    *,
    normalization: str = "component",
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a complete real O(2) Clebsch--Gordan tensor.

    The returned tensor has shape ``(input1.dim, input2.dim, output.dim)``.
    ``component`` normalization gives every output component unit norm.
    """
    input1 = check_o2_irrep(input1)
    input2 = check_o2_irrep(input2)
    output = check_o2_irrep(output)
    if not has_tensor_product_path(output, input1, input2):
        raise ValueError(f"No O(2) path {input1} x {input2} -> {output}.")
    if normalization not in {"component", "none"}:
        raise ValueError("normalization must be 'component' or 'none'.")

    cg = _new_cg_tensor(input1, input2, output, dtype, device)
    pair_scale = math.sqrt(0.5) if normalization == "component" else 1.0

    if input1.m == 0 and input2.m == 0:
        cg[0, 0, 0] = 1.0
        return cg

    if input1.m == 0 or input2.m == 0:
        scalar = input1 if input1.m == 0 else input2
        intertwiner = torch.eye(2, dtype=cg.dtype, device=cg.device)
        if scalar.p == -1:
            # J intertwines the determinant-twisted positive-order irrep with
            # the canonical real m block.
            intertwiner = cg.new_tensor([[0.0, -1.0], [1.0, 0.0]])
        if input1.m == 0:
            cg[0, :, :] = intertwiner.transpose(0, 1)
        else:
            cg[:, 0, :] = intertwiner.transpose(0, 1)
        return cg

    if output.m == input1.m + input2.m:
        # Complex multiplication: z1 * z2.
        cg[0, 0, 0] = pair_scale
        cg[1, 1, 0] = -pair_scale
        cg[0, 1, 1] = pair_scale
        cg[1, 0, 1] = pair_scale
        return cg

    if output.m == 0:
        if output.p == 1:
            cg[0, 0, 0] = pair_scale
            cg[1, 1, 0] = pair_scale
        else:
            cg[0, 1, 0] = pair_scale
            cg[1, 0, 0] = -pair_scale
        return cg

    # Difference-frequency path. The larger frequency is multiplied by the
    # complex conjugate of the smaller one, fixing a canonical positive order.
    cg[0, 0, 0] = pair_scale
    cg[1, 1, 0] = pair_scale
    if input1.m > input2.m:
        cg[1, 0, 1] = pair_scale
        cg[0, 1, 1] = -pair_scale
    else:
        cg[0, 1, 1] = pair_scale
        cg[1, 0, 1] = -pair_scale
    return cg


@dataclass(frozen=True)
class O2Recoupling:
    """Change of basis between ``(1 x 2) x 3`` and ``1 x (2 x 3)``."""

    left_intermediates: Tuple[Irrep, ...]
    right_intermediates: Tuple[Irrep, ...]
    matrix: torch.Tensor


def _left_coupling_tensor(
    input1: Irrep,
    input2: Irrep,
    input3: Irrep,
    intermediate: Irrep,
    output: Irrep,
    dtype: Optional[torch.dtype],
    device: Optional[torch.device],
) -> torch.Tensor:
    first = o2_clebsch_gordan(input1, input2, intermediate, dtype=dtype, device=device)
    second = o2_clebsch_gordan(intermediate, input3, output, dtype=dtype, device=device)
    return torch.einsum("abe,ecd->abcd", first, second)


def _right_coupling_tensor(
    input1: Irrep,
    input2: Irrep,
    input3: Irrep,
    intermediate: Irrep,
    output: Irrep,
    dtype: Optional[torch.dtype],
    device: Optional[torch.device],
) -> torch.Tensor:
    first = o2_clebsch_gordan(input2, input3, intermediate, dtype=dtype, device=device)
    second = o2_clebsch_gordan(input1, intermediate, output, dtype=dtype, device=device)
    return torch.einsum("bcf,afd->abcd", first, second)


def o2_recoupling(
    input1: IrrepLike,
    input2: IrrepLike,
    input3: IrrepLike,
    output: IrrepLike,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> O2Recoupling:
    """Compute the real O(2) Racah/F matrix by contracting CG tensors."""
    input1 = check_o2_irrep(input1)
    input2 = check_o2_irrep(input2)
    input3 = check_o2_irrep(input3)
    output = check_o2_irrep(output)

    left_intermediates = tuple(
        intermediate
        for intermediate in tensor_product_irrep(input1, input2)
        if has_tensor_product_path(output, intermediate, input3)
    )
    right_intermediates = tuple(
        intermediate
        for intermediate in tensor_product_irrep(input2, input3)
        if has_tensor_product_path(output, input1, intermediate)
    )
    if len(left_intermediates) != len(right_intermediates):
        raise RuntimeError("Inconsistent O(2) recoupling path dimensions.")

    dtype = torch.get_default_dtype() if dtype is None else dtype
    matrix = torch.empty(
        len(left_intermediates),
        len(right_intermediates),
        dtype=dtype,
        device=device,
    )
    left_tensors = [
        _left_coupling_tensor(
            input1,
            input2,
            input3,
            intermediate,
            output,
            dtype,
            device,
        )
        for intermediate in left_intermediates
    ]
    right_tensors = [
        _right_coupling_tensor(
            input1,
            input2,
            input3,
            intermediate,
            output,
            dtype,
            device,
        )
        for intermediate in right_intermediates
    ]
    for left_index, left in enumerate(left_tensors):
        for right_index, right in enumerate(right_tensors):
            matrix[left_index, right_index] = torch.sum(left * right) / output.dim
    return O2Recoupling(left_intermediates, right_intermediates, matrix)


def o2_racah_matrix(*args, **kwargs) -> torch.Tensor:
    """Return only the matrix from :func:`o2_recoupling`."""
    return o2_recoupling(*args, **kwargs).matrix
