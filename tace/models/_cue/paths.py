################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Union
import itertools

from e3nn import o3
try:
    import cuequivariance as cue
    from cuequivariance.group_theory.irreps_array.irrep_utils import into_list_of_irrep
    from cuequivariance.group_theory.experimental.e3nn import O3_e3nn
except Exception:
    pass


def satisfy(l1: int, l2: int, restriction: Union[str, None] = None) -> bool:
    if restriction == None:
        return True
    elif restriction == "<":
        return l1 < l2
    elif restriction == "<=":
        return l1 <= l2
    elif restriction == ">":
        return l1 > l2
    elif restriction == ">=":
        return l1 >= l2
    elif restriction == "==":
        return l1 == l2
    elif restriction == "!=":
        return l1 != l2
    else:
        raise ValueError(f"Unknown restriction: {restriction}")
    
    
def generate_cueq_paths(
    irreps_out: o3.Irreps,
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    l1l2: Union[str, None] = None,
    l2l3: Union[str, None] = None,
    l3l1: Union[str, None] = None,
    ictp_ictc_like: bool = True,
):
    """
    Based on ceuq channelwise_tensor_product
    """
    irreps_in1: cue.Irreps = cue.Irreps(O3_e3nn, irreps_in1)
    irreps_in2: cue.Irreps = cue.Irreps(O3_e3nn, irreps_in2)
    irreps_out: cue.Irreps = cue.Irreps(O3_e3nn, irreps_out)
    G = irreps_in1.irrep_class
    irrep_out_list = into_list_of_irrep(G, irreps_out)

    d = cue.SegmentedTensorProduct.from_subscripts("uv,iu,jv,kuv+ijk")

    for mul1, ir1 in irreps_in1:
        d.add_segment(1, (ir1.dim, mul1))
    for mul2, ir2 in irreps_in2:
        d.add_segment(2, (ir2.dim, mul2))

    irreps_out_list = []
    for (i, (mul1, ir1)), (j, (mul2, ir2)) in itertools.product(
        enumerate(irreps_in1), enumerate(irreps_in2)
    ):
        for ir_out in ir1 * ir2:
            if ir_out not in irrep_out_list:
                continue

            l1 = ir1.l
            l2 = ir2.l
            l3 = ir_out.l

            triangle_ok = (
                l3 in range(abs(l1 - l2), l1 + l2 + 1, 2)
                if ictp_ictc_like
                else True
            )
            if (
                triangle_ok
                and satisfy(l1, l2, l1l2)
                and satisfy(l2, l3, l2l3)
                and satisfy(l3, l1, l3l1)
            ):
                for cg in cue.clebsch_gordan(ir1, ir2, ir_out):
                    d.add_path(None, i, j, None, c=cg, dims={"u": mul1, "v": mul2})
                    irreps_out_list.append((mul1 * mul2, ir_out))

    actual_irreps_out = cue.Irreps(G, irreps_out_list)
    actual_irreps_out, perm, inv = actual_irreps_out.sort()
    d = d.permute_segments(0, inv)
    d = d.permute_segments(3, inv)
    d = d.normalize_paths_for_operand(-1)
    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps_in1.new_scalars(d.operands[0].size), cue.ir_mul),
            cue.IrrepsAndLayout(irreps_in1, cue.ir_mul),
            cue.IrrepsAndLayout(irreps_in2, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(actual_irreps_out, cue.ir_mul)],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )