################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


from typing import Optional, List, Tuple
from collections import defaultdict


from ...utils._global import LETTERS


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


def generate_combinations(
    l1max: int,
    l2max: int,
    l3max: int,
    l1l2: Optional[str] = None,
    l2l3: Optional[str] = None,
    l3l1: Optional[str] = None,
) -> List[Tuple]:
    combs = []
    for l1 in range(l1max + 1):
        for l2 in range(l2max + 1):
            for l3 in range(abs(l1 - l2), min(l3max, l1 + l2) + 1, 2):
                if (satisfy(l1, l2, l1l2) and satisfy(l2, l3, l2l3) and satisfy(l3, l1, l3l1)):
                    k = (l1 + l2 - l3) // 2
                    combs.append((l1, l2, l3, k))
    return combs


def generate_path(
    l1: int,
    l2: int,
    l3: int,
    k : int,
    add_batch_and_channel: bool = True,
) -> str:
    assert isinstance(l1, int) and l1 >= 0, "l1 must be a non-negative integer"
    assert isinstance(l2, int) and l2 >= 0, "l2 must be a non-negative integer"
    assert isinstance(l3, int) and l3 >= 0, "l3 must be a non-negative integer"
    assert isinstance(l3, int) and k >= 0, "k must be a non-negative integer"
    assert (l1 + l2 - l3) % 2 == 0, "Illegal contraction combination"

    in1_list = LETTERS[:l1]
    in2_list = LETTERS[l1 : l1 + l2]
    in1_list_copy = in1_list.copy()
    in2_list_copy = in2_list.copy()

    in1_idx = list(range(l1 - k, l1))
    in2_idx = list(range(k))
    
    for i in range(k):
        in2_list_copy[in2_idx[i]] = in1_list_copy[in1_idx[i]]

    in1 = "".join(in1_list_copy)
    in2 = "".join(in2_list_copy)
    out = "".join(
        [l for i, l in enumerate(in1_list_copy) if i not in in1_idx] 
        + 
        [l for _, l in enumerate(in2_list_copy) if l not in in1_list_copy]
    )

    if add_batch_and_channel:
        in1 = "bc" + in1
        in2 = "bc" + in2
        out = "bc" + out

    einsum_expr = f"{in1},{in2}->{out}"
    return einsum_expr


def generate_prod_paths(
    lmax_in, ls_out, correlation, l1l2, l2l3, l3l1,
):
    paths_list_list = []
    exprs_list_list = []
    current_ranks = set(range(lmax_in + 1))

    for nu in range(2, correlation + 1):
        path_list = [] # TODO convert this to dict
        expr_list = [] # TODO convert this to dict
        next_ranks = set()
        for l1 in current_ranks:
            for l2 in range(lmax_in + 1):
                lmin = abs(l1 - l2)
                lmax = min(lmax_in, l1 + l2)
                for l3 in range(lmin, lmax + 1, 2):
                    if satisfy(l1, l2, l1l2) and satisfy(l2, l3, l2l3) and satisfy(l3, l1, l3l1):
                        k = (l1 + l2 - l3) // 2
                        path_list.append((l1, l2, l3, k))
                        next_ranks.add(l3)
                        expr_list.append(generate_path(l1, l2, l3, k, True))
         
        paths_list_list.append(path_list)
        exprs_list_list.append(expr_list)
        current_ranks = next_ranks

    valid_ranks = ls_out
    for nu in reversed(range(correlation - 1)):
        filtered_path = []
        filtered_expr = []
        next_valid_ranks = set()

        for (l1, l2, l3, k), expr in zip(paths_list_list[nu], exprs_list_list[nu]):
            if l3 in valid_ranks:
                filtered_path.append((l1, l2, l3, k))
                filtered_expr.append(expr)
                next_valid_ranks.update([l1, l2])

        paths_list_list[nu] = filtered_path
        exprs_list_list[nu] = filtered_expr
        valid_ranks = next_valid_ranks

    return paths_list_list, exprs_list_list



def count_irreps(
        combs: List[Tuple],
        ictp_lower_weight: bool = False,
        ictc_lower_weight: bool = False,
        ictp_highest_weight: bool = True, 
        ictc_highest_weight: bool = True, 
    ):

    l3_count = defaultdict(int) 
    for comb in combs:
        l1, l2, _, k = comb 
        for l3 in range(abs(l1 - l2), l1 + l2 - 2 * k + 1):
            hw = (l3 == l1 + l2 - 2 * k)
            lw = (l3 <  l1 + l2 - 2 * k)
            if k == 0:
                if ictp_highest_weight and hw:
                    l3_count[l3] += 1
                if ictp_lower_weight and lw:
                    l3_count[l3] += 1
            else:
                if ictc_highest_weight and hw:
                    l3_count[l3] += 1
                if ictc_lower_weight and lw:
                    l3_count[l3] += 1
                    
    return dict(sorted(l3_count.items(), key=lambda x: x[0]))
