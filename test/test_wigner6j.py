
# import torch

# from tace.models.wigner6j.tensor_product import E2TensorProductArbitraryOrder


# for o in range(1, 4):
#     head, hidden = 8, 16 # total channel
#     f_N1, f_N2 = 7, 8
#     alpha_ij = torch.randn(f_N1, f_N2, head) # [edge]

#     h = torch.randn(f_N1, (o + 1) ** 2, head * hidden) # n_i
#     exp_h = torch.randn(f_N2, (o + 1) ** 2, head * hidden) # n_j

#     pos = torch.randn(f_N1, 3)
#     exp_pos = torch.randn(f_N2, 3)

#     irreps_in = "+".join(
#         [
#             f"{head*hidden}x0e",
#             f"{head*hidden}x1e",
#             f"{head*hidden}x2e",
#             f"{head*hidden}x3e",
#             f"{head*hidden}x4e",
#         ][: o + 1]
#     )
#     irreps_out = irreps_in

#     learnable_weight = True
#     connection_mode = "uvw"

#     # Test arbitrary order with second order case
#     model_arbitrary = E2TensorProductArbitraryOrder(
#         irreps_in,
#         irreps_out,
#         head,
#         order=o,
#         learnable_weight=learnable_weight,
#         connection_mode=connection_mode,
#         path_normalization="element",
#     )
#     out_arbitrary = model_arbitrary(pos, exp_pos, h, exp_h, alpha_ij)
#     print(out_arbitrary.shape)
#     # for name, param in model_arbitrary.named_parameters():
#     #     print(name, param.shape, param.requires_grad)
#     out_second = model_arbitrary.vanilla_forward(pos, exp_pos, h, exp_h, alpha_ij)
#     # Print comparison metrics
#     diff = out_arbitrary / out_second
#     print(f"\nComparing Arbitrary Order (n={o}) vs Second Order:")
#     print(f"Max difference: {torch.max(diff):.8f}")
#     print(f"Mean difference: {torch.mean(diff):.8f}")
#     print(f"Min difference: {torch.min(diff):.8f}")

import torch
from sympy.physics.wigner import (
    clebsch_gordan,
    wigner_6j,
)
from sympy import S
import math


dtype = torch.float64

def cg_matrix(l1, l2, L):

    C = torch.zeros(
        2 * L + 1,
        2 * l1 + 1,
        2 * l2 + 1,
        dtype=dtype,
    )

    for M in range(-L, L + 1):
        for m1 in range(-l1, l1 + 1):
            for m2 in range(-l2, l2 + 1):

                if m1 + m2 != M:
                    continue

                val = clebsch_gordan(
                    S(l1),
                    S(l2),
                    S(L),
                    S(m1),
                    S(m2),
                    S(M),
                )

                C[M + L, m1 + l1, m2 + l2] = float(val)

    return C


def couple(x, y, l1, l2, L):

    C = cg_matrix(l1, l2, L)

    return torch.einsum(
        "Mab,a,b->M",
        C,
        x,
        y,
    )


def allowed_L(l1, l2):

    return list(range(
        abs(l1 - l2),
        l1 + l2 + 1,
    ))


def verify_recoupling(
    l1,
    l2,
    l3,
    l23,
    L,
):

    torch.manual_seed(0)

    x1 = torch.randn(2 * l1 + 1, dtype=dtype)
    x2 = torch.randn(2 * l2 + 1, dtype=dtype)
    x3 = torch.randn(2 * l3 + 1, dtype=dtype)

    # LEFT:
    # l1 ⊗ (l2 ⊗ l3)_l23 -> L
    y23 = couple(
        x2,
        x3,
        l2,
        l3,
        l23,
    )

    LEFT = couple(
        x1,
        y23,
        l1,
        l23,
        L,
    )

    # RIGHT:
    # Σ_l12 ((l1 ⊗ l2)_l12 ⊗ l3)_L
  
    RIGHT = torch.zeros(
        2 * L + 1,
        dtype=dtype,
    )

    for l12 in allowed_L(l1, l2):

        if L not in allowed_L(l12, l3):
            continue

        y12 = couple(
            x1,
            x2,
            l1,
            l2,
            l12,
        )

        tmp = couple(
            y12,
            x3,
            l12,
            l3,
            L,
        )

        sixj = float(
            wigner_6j(
                S(l1),
                S(l2),
                S(l12),
                S(l3),
                S(L),
                S(l23),
            )
        )

        coeff = math.sqrt(
            (2 * l12 + 1)
            * (2 * l23 + 1)
        ) * sixj

        RIGHT += coeff * tmp

    err = (LEFT - RIGHT).abs().max()

    return LEFT, RIGHT, err


LEFT, RIGHT, err = verify_recoupling(
    l1=4,
    l2=6,
    l3=5,
    l23=7,
    L=7,
)

print("LEFT:")
print(LEFT)

print()

print("RIGHT:")
print(RIGHT)

print()

print("max error:")
print(err)