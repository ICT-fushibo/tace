################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional
import itertools


try:
    import cuequivariance as cue
    from cuequivariance.group_theory.irreps_array.irrep_utils import into_list_of_irrep
except Exception:
    pass


def satisfy(l1: int, l2: int, restriction: Optional[str] = None) -> bool:
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
    irreps1,
    irreps2,
    irreps3,
    l1l2: str | None = None,
    l2l3: str | None = None,
    l3l1: str | None = None,
    ictp_ictc_like: bool = True,
):
    """
    Based on ceuq channelwise_tensor_product
    """

    G = irreps1.irrep_class
    irreps3_filter = into_list_of_irrep(G, irreps3)

    d = cue.SegmentedTensorProduct.from_subscripts("uv,iu,jv,kuv+ijk")

    for mul, ir in irreps1:
        d.add_segment(1, (ir.dim, mul))
    for mul, ir in irreps2:
        d.add_segment(2, (ir.dim, mul))

    irreps3 = []
    for (i1, (mul1, ir1)), (i2, (mul2, ir2)) in itertools.product(
        enumerate(irreps1), enumerate(irreps2)
    ):
        for ir3 in ir1 * ir2:
            if ir3 not in irreps3_filter:
                continue

            l1 = ir1.l
            l2 = ir2.l
            l3 = ir3.l

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
                for cg in cue.clebsch_gordan(ir1, ir2, ir3):
                    d.add_path(None, i1, i2, None, c=cg, dims={"u": mul1, "v": mul2})
                    irreps3.append((mul1 * mul2, ir3))

    irreps3 = cue.Irreps(G, irreps3)
    irreps3, perm, inv = irreps3.sort()
    d = d.permute_segments(3, inv)
    d = d.normalize_paths_for_operand(-1)

    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps1.new_scalars(d.operands[0].size), cue.ir_mul),
            cue.IrrepsAndLayout(irreps1, cue.ir_mul),
            cue.IrrepsAndLayout(irreps2, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(irreps3, cue.ir_mul)],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )



