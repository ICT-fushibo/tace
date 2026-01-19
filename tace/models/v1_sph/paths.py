################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
"""
Hardcode Highest weight (ICTP) + (ICTC-like) STP
"""

from typing import Optional, List, Tuple


from e3nn import o3


from tace.models.v1.paths import satisfy


def generate_inter_paths(
    irreps_in1: o3.Irreps, 
    irreps_in2: o3.Irreps, 
    irreps_out: o3.Irreps,
    l1l2: Optional[str] = None,
    l2l3: Optional[str] = None,
    l3l1: Optional[str] = None,
) -> Tuple[o3.Irreps, List]:
    irreps_out_list: List[Tuple[int, o3.Irreps]] = []
    instructions = []
    for i, (mul, ir_left) in enumerate(irreps_in1):
        for j, (_, ir_right) in enumerate(irreps_in2):
            l1 = ir_left.l
            l2 = ir_right.l
            for ir_out in ir_left * ir_right:
                l3 = ir_out.l
                if l3 in list(range(abs(l1 - l2), (l1 + l2) + 1, 2)):
                    if (satisfy(l1, l2, l1l2) and satisfy(l2, l3, l2l3) and satisfy(l3, l1, l3l1)):
                        if ir_out in irreps_out:
                            k = len(irreps_out_list)  # instruction index
                            irreps_out_list.append((mul, ir_out))
                            instructions.append((i, j, k, "uvu", True))

    irreps_out = o3.Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]
    instructions = sorted(instructions, key=lambda x: x[2])

    return irreps_out, instructions


def generate_prod_paths(
    irreps_in1: o3.Irreps, 
    irreps_in2: o3.Irreps, 
    irreps_out: o3.Irreps,
    l1l2: Optional[str] = None,
    l2l3: Optional[str] = None,
    l3l1: Optional[str] = None,
) -> Tuple[o3.Irreps, List]:
    irreps_out_list: List[Tuple[int, o3.Irreps]] = []
    instructions = []
    for i, (mul, ir_left) in enumerate(irreps_in1):
        for j, (_, ir_right) in enumerate(irreps_in2):
            l1 = ir_left.l
            l2 = ir_right.l
            for ir_out in ir_left * ir_right:
                l3 = ir_out.l
                if l3 in list(range(abs(l1 - l2), (l1 + l2) + 1, 2)):
                    if (satisfy(l1, l2, l1l2) and satisfy(l2, l3, l2l3) and satisfy(l3, l1, l3l1)):
                        if ir_out in irreps_out:
                            k = len(irreps_out_list)
                            irreps_out_list.append((mul, ir_out))
                            instructions.append((i, j, k, "uuu", False))

    irreps_out = o3.Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]
    instructions = sorted(instructions, key=lambda x: x[2])

    return irreps_out, instructions