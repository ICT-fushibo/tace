################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional, List, Tuple


import equitorch as eqt
from e3nn import o3


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
    

def generate_eqt_e3nn_paths(
    irreps_out: eqt.irreps.Irreps, 
    irreps_in1: eqt.irreps.Irreps, 
    irreps_in2: eqt.irreps.Irreps, 
    num_channel: int,
    *,
    l1l2: Optional[str] = None,
    l2l3: Optional[str] = None,
    l3l1: Optional[str] = None,
    ictp_ictc_like: bool = True,
    e3nn_mode = 'uvu',
):
    
    eqt_paths: List[Tuple[int, int, int]] = []
    eqt_out_irreps: List[str] = [] 

    e3nn_paths: List[Tuple[int, int, int, str, bool]] = []
    e3nn_out_irreps: List[Tuple[int, o3.Irrep]] = [] 

    for (_, ir_out) in enumerate(irreps_out):
        for (i, ir1) in enumerate(irreps_in1):
            for (j, ir2) in enumerate(irreps_in2):
                
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
                    and eqt.irreps.has_path(ir_out, ir1, ir2)
                ):
                    k = len(eqt_out_irreps)
                    eqt_paths.append((k, i, j))  
                    eqt_out_irreps.append(str(ir_out)) 

                    e3nn_out_irreps.append((num_channel, (ir_out.l, ir_out.p)))
                    e3nn_paths.append((i, j, k, e3nn_mode, e3nn_mode=='uvu'))

    return eqt_paths, eqt.irreps.Irreps("+".join(eqt_out_irreps)),  e3nn_paths, o3.Irreps(e3nn_out_irreps)
