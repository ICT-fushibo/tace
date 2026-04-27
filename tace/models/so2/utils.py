################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import torch


def so2_expand_index(mmax: int, lmax: int, start: int = 0) -> tuple[int, torch.Tensor]:
    expand_index = []
    offset = 0
    for m in range(start, mmax + 1):
        index = torch.arange((lmax + 1 - m))
        index = index + offset
        expand_index.append(index)
        if m > 0:
            expand_index.append(index)    # +- m
        offset = offset + len(index)
    expand_index = torch.cat(expand_index, dim=0)
    expand_index = expand_index.long()
    num_m_components = offset
    return num_m_components, expand_index


def so3_expand_index(mmax: int, lmax: int) -> tuple[int, torch.Tensor]:
    expand_index = torch.zeros([((lmax + 1) ** 2)]).long()
    start_idx = 0
    for l in range(lmax + 1):
        length = 2 * l + 1
        expand_index[start_idx : (start_idx + length)] = l
        start_idx = start_idx + length
    num_l_components = lmax + 1
    return num_l_components, expand_index
