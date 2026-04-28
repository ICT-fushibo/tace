################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional, List, Tuple


from e3nn import o3


def satisfy(l1: int, l2: int, restriction: str | None = None) -> bool:
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
    

def generate_paths(
    irreps_out: o3.Irreps, 
    irreps_in1: o3.Irreps, 
    irreps_in2: o3.Irreps,  
    *,
    l1l2: str | None = None,
    l2l3: str | None = None,
    l3l1: str | None = None,
    l3s: List[int] | None = None,
    ictp_ictc_like: bool = True,
    e3nn_mode = 'uvu',
    trainable: bool = False,
):

    e3nn_paths: List[Tuple[int, int, int, str, bool]] = []
    e3nn_out_irreps: List[Tuple[int, o3.Irrep]] = [] 


    for _, (_, ir_out) in enumerate(irreps_out):
        for i, (mul, ir1) in enumerate(irreps_in1):
            for j, (_, ir2) in enumerate(irreps_in2):
                
                l1 = ir1.l
                l2 = ir2.l
                l3 = ir_out.l

                triangle_ok = (
                    l3 in range(abs(l1 - l2), l1 + l2 + 1, 2)
                    if ictp_ictc_like
                    else True
                )
                l3_ok = l3 in l3s if l3s else True

                if (
                    triangle_ok
                    and satisfy(l1, l2, l1l2)
                    and satisfy(l2, l3, l2l3)
                    and satisfy(l3, l1, l3l1)
                    and ir_out in ir1 * ir2
                    and l3_ok
                ):

                    k = len(e3nn_out_irreps)
                    e3nn_out_irreps.append((mul, (ir_out.l, ir_out.p)))
                    e3nn_paths.append((i, j, k, e3nn_mode, e3nn_mode=='uvu' or trainable))
                    
    return e3nn_paths, o3.Irreps(e3nn_out_irreps)

