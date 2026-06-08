################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
''''
Some components are from eqv3 and Fair.
'''

import os
import math
import torch
import copy

from e3nn import o3
from e3nn.o3 import FromS2Grid, ToS2Grid


# from .triton_wigner import fused_sparse_wigner_bmm
from .utils import so2_expand_index, so3_expand_index


# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L10
# _Jd is a list of tensors of shape (2l+1, 2l+1)
_Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt")) # to L = 10


# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L37
#
# In 0.5.0, e3nn shifted to torch.matrix_exp which is significantly slower:
# https://github.com/e3nn/e3nn/blob/0.5.0/e3nn/o3/_wigner.py#L92
def wigner_D(l, alpha, beta, gamma):
    if not l < len(_Jd):
        raise NotImplementedError(
            f"wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more"
        )

    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[l].to(dtype=alpha.dtype, device=alpha.device)
    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)
    return Xa @ J @ Xb @ J @ Xc


def _z_rot_mat(angle, l):
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
    inds = torch.arange(0, 2 * l + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M


"""
    For gradient methods, we do not backpropogate rotation if y component of 
    the unit vector of relative position is very close to `_ROTATION_MASK_THRESHOLD`.
"""
_ROTATION_MASK_THRESHOLD = 0.999999


def init_edge_rot_mat(edge_distance_vec, use_rotation_mask=True):
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.sqrt(torch.sum(edge_vec_0**2, dim=1))

    # Make sure the atoms are far enough apart
    #assert torch.min(edge_vec_0_distance) < 0.0001
    if torch.min(edge_vec_0_distance) < 0.0001:
        print(
            "Error edge_vec_0_distance: {}".format(
                torch.min(edge_vec_0_distance)
            )
        )

    norm_x = edge_vec_0 / (edge_vec_0_distance.view(-1, 1))

    if use_rotation_mask:
        """
            For gradient methods, we do not backpropogate rotation if y component of 
            the unit vector of relative position is very close to `_ROTATION_MASK_THRESHOLD`.
        """
        yprod = norm_x @ norm_x.new_tensor([0.0, 1.0, 0.0])
        norm_x[yprod >  _ROTATION_MASK_THRESHOLD] = norm_x.new_tensor([0.0,  1.0, 0.0])
        norm_x[yprod < -_ROTATION_MASK_THRESHOLD] = norm_x.new_tensor([0.0, -1.0, 0.0])

    edge_vec_2 = torch.rand_like(edge_vec_0) - 0.5
    edge_vec_2 = edge_vec_2 / (
        torch.sqrt(torch.sum(edge_vec_2**2, dim=1)).view(-1, 1)
    )
    # Create two rotated copys of the random vectors in case the random vector is aligned with norm_x
    # With two 90 degree rotated vectors, at least one should not be aligned with norm_x
    edge_vec_2b = edge_vec_2.clone()
    edge_vec_2b[:, 0] = -edge_vec_2[:, 1]
    edge_vec_2b[:, 1] = edge_vec_2[:, 0]
    edge_vec_2c = edge_vec_2.clone()
    edge_vec_2c[:, 1] = -edge_vec_2[:, 2]
    edge_vec_2c[:, 2] = edge_vec_2[:, 1]
    vec_dot_b = torch.abs(torch.sum(edge_vec_2b * norm_x, dim=1)).view(
        -1, 1
    )
    vec_dot_c = torch.abs(torch.sum(edge_vec_2c * norm_x, dim=1)).view(
        -1, 1
    )

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(
        torch.gt(vec_dot, vec_dot_b), edge_vec_2b, edge_vec_2
    )
    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1)).view(-1, 1)
    edge_vec_2 = torch.where(
        torch.gt(vec_dot, vec_dot_c), edge_vec_2c, edge_vec_2
    )

    vec_dot = torch.abs(torch.sum(edge_vec_2 * norm_x, dim=1))
    # Check the vectors aren't aligned
    assert torch.max(vec_dot) < 0.99

    norm_z = torch.cross(norm_x, edge_vec_2, dim=1)
    norm_z = norm_z / (
        torch.sqrt(torch.sum(norm_z**2, dim=1, keepdim=True))
    )
    norm_z = norm_z / (
        torch.sqrt(torch.sum(norm_z**2, dim=1)).view(-1, 1)
    )
    norm_y = torch.cross(norm_x, norm_z, dim=1)
    norm_y = norm_y / (
        torch.sqrt(torch.sum(norm_y**2, dim=1, keepdim=True))
    )

    # Construct the 3D rotation matrix
    norm_x = norm_x.view(-1, 3, 1)
    norm_y = -norm_y.view(-1, 3, 1)
    norm_z = norm_z.view(-1, 3, 1)

    edge_rot_mat_inv = torch.cat([norm_z, norm_x, norm_y], dim=2)
    edge_rot_mat = torch.transpose(edge_rot_mat_inv, 1, 2)

    if use_rotation_mask:
        return edge_rot_mat
    else:
        return edge_rot_mat.detach()
        

class CoefficientMappingModule(torch.nn.Module):
    """
    Helper module for coefficients used to reshape l <--> m and to get coefficients of specific degree or order

    Args:
        lmax (int):             Maximum degree of the spherical harmonics
        mmax (int):             Maximum order of the spherical harmonics
        use_rotate_inv_rescale (bool): 
                                Whether to pre-compute inverse rotation rescale matrices
    """
    def __init__(
        self,
        lmax,
        mmax,
        use_rotate_inv_rescale=False
    ):
        super().__init__()

        self.lmax = lmax
        self.mmax = mmax
        self.use_rotate_inv_rescale = use_rotate_inv_rescale

        m_complex  = [] # this m belongs to which SO(3) m
        l_harmonic = [] # this m belongs to which SO(3) l
        m_harmonic = [] # this m belongs to which SO(2) m

        for l in range(0, self.lmax + 1):
            mmax = min(self.mmax, l)
            m = torch.arange(-mmax, mmax + 1).long()
            m_complex.append(m)
            m_harmonic.append(torch.abs(m).long())
            l_harmonic.append(torch.fill(m, l))
        m_complex = torch.cat(m_complex, dim=0)   # tensor([0, -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3])
        m_harmonic = torch.cat(m_harmonic, dim=0) # tensor([0, 1, 0, 1, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3])
        l_harmonic = torch.cat(l_harmonic, dim=0) # tensor([0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3])

        num_components = len(l_harmonic)
        to_m = torch.zeros([num_components, num_components])

        offset = 0
        for m in range(self.mmax + 1):
            idx_r, idx_i = self.complex_idx(m, -1, m_complex, l_harmonic)
            for idx_out, idx_in in enumerate(idx_r):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)
            for idx_out, idx_in in enumerate(idx_i):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)

        to_m = to_m.detach()

        self.register_buffer('l_harmonic', l_harmonic)
        self.register_buffer('m_harmonic', m_harmonic) 
        self.register_buffer('m_complex',  m_complex) 
        self.register_buffer('to_m',       to_m)     

        # for `torch.compile()` compatibility
        self.pre_compute_coefficient_idx()
        if self.use_rotate_inv_rescale:
            self.pre_compute_rotate_inv_rescale()

    def complex_idx(self, m, lmax, m_complex, l_harmonic):
        if lmax == -1:
            lmax = self.lmax

        indices = torch.arange(len(l_harmonic))
        mask_r = torch.bitwise_and(
            l_harmonic.le(lmax), m_complex.eq(m)
        )
        mask_idx_r = torch.masked_select(indices, mask_r)

        mask_idx_i = torch.tensor([]).long()
        if m != 0:
            mask_i = torch.bitwise_and(
                l_harmonic.le(lmax), m_complex.eq(-m)
            )
            mask_idx_i = torch.masked_select(indices, mask_i)

        return mask_idx_r, mask_idx_i


    def pre_compute_coefficient_idx(self):
        for l in range(self.lmax + 1):
            for m in range(self.lmax + 1):
                mask = torch.bitwise_and(
                    self.l_harmonic.le(l), self.m_harmonic.le(m)
                )
                indices = torch.arange(len(mask))
                mask_indices = torch.masked_select(indices, mask)
                self.register_buffer('coefficient_idx_l{}_m{}'.format(l, m), mask_indices)
        return
    

    def prepare_coefficient_idx(self) -> list[list[torch.Tensor]]:
        # idx = lmax, mmax
        coefficient_idx_list = []
        for l in range(self.lmax + 1):
            l_list = []
            for m in range(self.lmax + 1):
                l_list.append(getattr(self, 'coefficient_idx_l{}_m{}'.format(l, m), None))
            coefficient_idx_list.append(l_list)
        return coefficient_idx_list
    

    def coefficient_idx(self, lmax, mmax):
        if lmax > self.lmax or mmax > self.lmax:
            mask = torch.bitwise_and(
                self.l_harmonic.le(lmax), self.m_harmonic.le(mmax)
            )
            indices = torch.arange(len(mask), device=mask.device)
            mask_indices = torch.masked_select(indices, mask)
            return mask_indices
        else:
            temp = self.prepare_coefficient_idx()
            return temp[lmax][mmax]
        
    
    def pre_compute_rotate_inv_rescale(self):
        for l in range(self.lmax + 1):
            for m in range(self.lmax + 1):
                mask_indices = self.coefficient_idx(l, m)
                rotate_inv_rescale = torch.ones((1, int((l + 1)**2), int((l + 1)**2)))
                for l_sub in range(l + 1):
                    if l_sub <= m:
                        continue
                    start_idx = l_sub ** 2
                    length = 2 * l_sub + 1
                    rescale_factor = math.sqrt(length / (2 * m + 1))
                    rotate_inv_rescale[:, start_idx : (start_idx + length), start_idx : (start_idx + length)] = rescale_factor
                rotate_inv_rescale = rotate_inv_rescale[:, :, mask_indices]
                self.register_buffer('rotate_inv_rescale_l{}_m{}'.format(l, m), rotate_inv_rescale)
        return 
    

    def prepare_rotate_inv_rescale(self):
        rotate_inv_rescale_list = []
        for l in range(self.lmax + 1):
            l_list = []
            for m in range(self.lmax + 1):
                l_list.append(getattr(self, 'rotate_inv_rescale_l{}_m{}'.format(l, m), None))
            rotate_inv_rescale_list.append(l_list)
        return rotate_inv_rescale_list
    

    def get_rotate_inv_rescale(self, lmax, mmax):
        temp = self.prepare_rotate_inv_rescale()
        return temp[lmax][mmax]


    def __repr__(self):
        return f"{self.__class__.__name__}(mmax={self.mmax}, lmax={self.lmax})"


class WignerD(torch.nn.Module):
    def __init__(
        self,
        lmax,
        mmax,
        use_rotation_mask=True,
        use_sparse_matmul=True,
    ):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.use_rotation_mask = use_rotation_mask
        self.use_sparse_matmul = use_sparse_matmul
        
        mapping = CoefficientMappingModule(
            lmax=self.lmax, 
            mmax=self.lmax, 
            use_rotate_inv_rescale=True
        )
        wigner_index_mask = mapping.coefficient_idx(self.lmax, self.mmax)
        wigner_inv_rescale = mapping.get_rotate_inv_rescale(self.lmax, self.mmax)
        
        # Merge converting m and l layout
        mapping = CoefficientMappingModule(
            lmax=self.lmax,
            mmax=self.mmax,
            use_rotate_inv_rescale=False
        )
        to_m = mapping.to_m
        wigner_inv_rescale = torch.einsum('nia, ba -> nib', wigner_inv_rescale, to_m)
        wigner_index_to_m_array = torch.zeros(
            to_m.shape[0],
            ((self.lmax + 1) ** 2)
        )
        # to_m [14, 14]
        # wigner_index_mask [14], tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 11, 12, 13, 14])
        wigner_index_to_m_array[:, wigner_index_mask] = to_m

        self.register_buffer('wigner_index_to_m_array', wigner_index_to_m_array)
        self.register_buffer('wigner_inv_rescale', wigner_inv_rescale) # [1, 16, 14]

        # # sparse
        # output_to_input = wigner_index_to_m_array.argmax(dim=1)
        # rotate_row_ptr, rotate_col_idx = self._sparse_wigner_indices(
        #     output_to_input, transpose=False
        # )
        # inverse_row_ptr, inverse_col_idx = self._sparse_wigner_indices(
        #     output_to_input, transpose=True
        # )
        # self.register_buffer("rotate_row_ptr", rotate_row_ptr)
        # self.register_buffer("rotate_col_idx", rotate_col_idx)
        # self.register_buffer("inverse_row_ptr", inverse_row_ptr)
        # self.register_buffer("inverse_col_idx", inverse_col_idx)
        # self.rotate_max_nnz = 2 * self.lmax + 1
        # self.inverse_max_nnz = max(
        #     1,
        #     max(
        #         int((output_to_input >= l * l).logical_and(
        #             output_to_input < (l + 1) * (l + 1)
        #         ).sum())
        #         for l in range(self.lmax + 1)
        #     ),
        # )

    def _sparse_wigner_indices(self, output_to_input, transpose):
        """Build the fixed CSR pattern while keeping TACE's existing layouts."""
        rows = (self.lmax + 1) ** 2 if transpose else output_to_input.numel()
        row_ptr = [0]
        col_idx = []
        for row in range(rows):
            if transpose:
                l = math.isqrt(row)
                columns = torch.nonzero(
                    (output_to_input >= l * l)
                    & (output_to_input < (l + 1) * (l + 1)),
                    as_tuple=False,
                ).flatten()
            else:
                l = math.isqrt(int(output_to_input[row]))
                columns = torch.arange(l * l, (l + 1) * (l + 1))
            col_idx.extend(columns.tolist())
            row_ptr.append(len(col_idx))
        return torch.tensor(row_ptr, dtype=torch.int32), torch.tensor(
            col_idx, dtype=torch.int32
        )
    
    def set_wigner(self, edge_vector):
        rot_mat3x3 = init_edge_rot_mat(edge_vector, use_rotation_mask=self.use_rotation_mask)
        wigner = self._rotation_to_wigner_matrix(rot_mat3x3, 0, self.lmax)
        wigner = torch.einsum('mi, nij -> nmj', self.wigner_index_to_m_array, wigner) # [14, 16] @ [16, 16]
        wigner_inv = torch.transpose(wigner, 1, 2).contiguous()
        wigner_inv = wigner_inv * self.wigner_inv_rescale
        self.wigner = wigner
        self.wigner_inv = wigner_inv

    def get_wigner(self, edge_vector) -> tuple[torch.Tensor]:
        rot_mat3x3 = init_edge_rot_mat(edge_vector, use_rotation_mask=self.use_rotation_mask)
        wigner = self._rotation_to_wigner_matrix(rot_mat3x3, 0, self.lmax)
        wigner = torch.einsum('mi, nij -> nmj', self.wigner_index_to_m_array, wigner) # [14, 16] @ [16, 16]
        wigner_inv = torch.transpose(wigner, 1, 2).contiguous()
        wigner_inv = wigner_inv * self.wigner_inv_rescale
        return wigner, wigner_inv

    # def rotate(self, wigner, inputs):
    #     if not self.use_sparse_matmul:
    #         return torch.bmm(wigner, inputs)
    #     return fused_sparse_wigner_bmm(
    #         wigner,
    #         inputs,
    #         self.rotate_row_ptr,
    #         self.rotate_col_idx,
    #         self.inverse_row_ptr,
    #         self.inverse_col_idx,
    #         self.rotate_max_nnz,
    #         self.inverse_max_nnz,
    #     )

    # def rotate_inv(self, wigner_inv, inputs):
    #     if not self.use_sparse_matmul:
    #         return torch.bmm(wigner_inv, inputs)
    #     return fused_sparse_wigner_bmm(
    #         wigner_inv,
    #         inputs,
    #         self.inverse_row_ptr,
    #         self.inverse_col_idx,
    #         self.rotate_row_ptr,
    #         self.rotate_col_idx,
    #         self.inverse_max_nnz,
    #         self.rotate_max_nnz,
    #     )
    
    def _rotation_to_wigner_matrix(self, edge_rot_mat, start_lmax, end_lmax):
        x = edge_rot_mat[:, :, 1]

        alpha, beta = o3.xyz_to_angles(x)
        R = o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2)
        R = torch.bmm(R, edge_rot_mat)

        gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

        if self.use_rotation_mask:
            yprod = (x @ x.new_tensor([0, 1, 0])).detach()
            backprop_mask = (yprod > -_ROTATION_MASK_THRESHOLD) & (yprod < _ROTATION_MASK_THRESHOLD)
            alpha_detach = alpha[(~backprop_mask)].clone().detach()
            gamma_detach = gamma[(~backprop_mask)].clone().detach()
            beta_detach = beta.clone().detach()
            beta_detach[yprod >  _ROTATION_MASK_THRESHOLD] = 0.0
            beta_detach[yprod < -_ROTATION_MASK_THRESHOLD] = math.pi
            beta_detach = beta_detach[(~backprop_mask)]

        size = int((end_lmax + 1) ** 2) - int((start_lmax) ** 2)
        wigner = torch.zeros(len(alpha), size, size, device=edge_rot_mat.device)
        start = 0
        for lmax in range(start_lmax, end_lmax + 1):
            if self.use_rotation_mask:
                block = wigner_D(
                    lmax, 
                    alpha[backprop_mask], 
                    beta[backprop_mask], 
                    gamma[backprop_mask]
                )
                block_detach = wigner_D(
                    lmax, 
                    alpha_detach, 
                    beta_detach, 
                    gamma_detach
                )
                end = start + block.size()[1]
                wigner[   backprop_mask, start:end, start:end] = block
                wigner[(~backprop_mask), start:end, start:end] = block_detach
            elif not self.use_rotation_mask:
                block = wigner_D(lmax, alpha, beta, gamma)
                end = start + block.size()[1]
                wigner[:, start:end, start:end] = block
            start = end
        if self.use_rotation_mask:
            return wigner
        else:
            return wigner.detach()

    def extra_repr(self):
        return 'mmax={}, lmax={}'.format(self.mmax, self.lmax)


# class WignerD(torch.nn.Module):
#     def __init__(
#         self,
#         lmax,
#         mmax,
#         use_rotation_mask=True,
#         use_sparse_matmul=True,
#     ):
#         super().__init__()
#         self.lmax = lmax
#         self.mmax = mmax
#         self.use_rotation_mask = use_rotation_mask
#         self.use_sparse_matmul = use_sparse_matmul
        
#         mapping = CoefficientMappingModule(
#             lmax=self.lmax, 
#             mmax=self.lmax, 
#             use_rotate_inv_rescale=True
#         )
#         wigner_index_mask = mapping.coefficient_idx(self.lmax, self.mmax)
#         wigner_inv_rescale = mapping.get_rotate_inv_rescale(self.lmax, self.mmax)
        
#         # Merge converting m and l layout
#         mapping = CoefficientMappingModule(
#             lmax=self.lmax,
#             mmax=self.mmax,
#             use_rotate_inv_rescale=False
#         )
#         to_m = mapping.to_m
#         wigner_inv_rescale = torch.einsum('nia, ba -> nib', wigner_inv_rescale, to_m)
#         wigner_index_to_m_array = torch.zeros(
#             to_m.shape[0],
#             ((self.lmax + 1) ** 2)
#         )
#         # to_m [14, 14]
#         # wigner_index_mask [14], tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 11, 12, 13, 14])
#         wigner_index_to_m_array[:, wigner_index_mask] = to_m

#         self.register_buffer('wigner_index_to_m_array', wigner_index_to_m_array)
#         self.register_buffer('wigner_inv_rescale', wigner_inv_rescale) # [1, 16, 14]
#         output_to_input = wigner_index_to_m_array.argmax(dim=1)
#         rotate_row_ptr, rotate_col_idx = self._sparse_wigner_indices(
#             output_to_input, transpose=False
#         )
#         inverse_row_ptr, inverse_col_idx = self._sparse_wigner_indices(
#             output_to_input, transpose=True
#         )
#         self.register_buffer("rotate_row_ptr", rotate_row_ptr)
#         self.register_buffer("rotate_col_idx", rotate_col_idx)
#         self.register_buffer("inverse_row_ptr", inverse_row_ptr)
#         self.register_buffer("inverse_col_idx", inverse_col_idx)
#         self.rotate_max_nnz = 2 * self.lmax + 1
#         self.inverse_max_nnz = max(
#             1,
#             max(
#                 int((output_to_input >= l * l).logical_and(
#                     output_to_input < (l + 1) * (l + 1)
#                 ).sum())
#                 for l in range(self.lmax + 1)
#             ),
#         )

#     def _sparse_wigner_indices(self, output_to_input, transpose):
#         """Build the fixed CSR pattern while keeping TACE's existing layouts."""
#         rows = (self.lmax + 1) ** 2 if transpose else output_to_input.numel()
#         row_ptr = [0]
#         col_idx = []
#         for row in range(rows):
#             if transpose:
#                 l = math.isqrt(row)
#                 columns = torch.nonzero(
#                     (output_to_input >= l * l)
#                     & (output_to_input < (l + 1) * (l + 1)),
#                     as_tuple=False,
#                 ).flatten()
#             else:
#                 l = math.isqrt(int(output_to_input[row]))
#                 columns = torch.arange(l * l, (l + 1) * (l + 1))
#             col_idx.extend(columns.tolist())
#             row_ptr.append(len(col_idx))
#         return torch.tensor(row_ptr, dtype=torch.int32), torch.tensor(
#             col_idx, dtype=torch.int32
#         )

#     def set_wigner(self, edge_vector):
#         rot_mat3x3 = init_edge_rot_mat(edge_vector, use_rotation_mask=self.use_rotation_mask)
#         wigner = self._rotation_to_wigner_matrix(rot_mat3x3, 0, self.lmax)
#         wigner = torch.einsum('mi, nij -> nmj', self.wigner_index_to_m_array, wigner) # [14, 16] @ [16, 16]
#         wigner_inv = torch.transpose(wigner, 1, 2).contiguous()
#         wigner_inv = wigner_inv * self.wigner_inv_rescale
#         self.wigner = wigner
#         self.wigner_inv = wigner_inv

#     def rotate(self, inputs, wigner):
#         if not self.use_sparse_matmul:
#             return torch.bmm(wigner, inputs)
#         return fused_sparse_wigner_bmm(
#             wigner,
#             inputs,
#             self.rotate_row_ptr,
#             self.rotate_col_idx,
#             self.inverse_row_ptr,
#             self.inverse_col_idx,
#             self.rotate_max_nnz,
#             self.inverse_max_nnz,
#         )

#     def rotate_inv(self, inputs, wigner_inv):
#         if not self.use_sparse_matmul:
#             return torch.bmm(wigner_inv, inputs)
#         return fused_sparse_wigner_bmm(
#             wigner_inv,
#             inputs,
#             self.inverse_row_ptr,
#             self.inverse_col_idx,
#             self.rotate_row_ptr,
#             self.rotate_col_idx,
#             self.inverse_max_nnz,
#             self.rotate_max_nnz,
#         )
    

#     def _rotation_to_wigner_matrix(self, edge_rot_mat, start_lmax, end_lmax):
#         x = edge_rot_mat[:, :, 1]

#         alpha, beta = o3.xyz_to_angles(x)
#         R = o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2)
#         R = torch.bmm(R, edge_rot_mat)

#         gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

#         if self.use_rotation_mask:
#             yprod = (x @ x.new_tensor([0, 1, 0])).detach()
#             backprop_mask = (yprod > -_ROTATION_MASK_THRESHOLD) & (yprod < _ROTATION_MASK_THRESHOLD)
#             alpha_detach = alpha[(~backprop_mask)].clone().detach()
#             gamma_detach = gamma[(~backprop_mask)].clone().detach()
#             beta_detach = beta.clone().detach()
#             beta_detach[yprod >  _ROTATION_MASK_THRESHOLD] = 0.0
#             beta_detach[yprod < -_ROTATION_MASK_THRESHOLD] = math.pi
#             beta_detach = beta_detach[(~backprop_mask)]

#         size = int((end_lmax + 1) ** 2) - int((start_lmax) ** 2)
#         wigner = torch.zeros(len(alpha), size, size, device=edge_rot_mat.device)
#         start = 0
#         for lmax in range(start_lmax, end_lmax + 1):
#             if self.use_rotation_mask:
#                 block = wigner_D(
#                     lmax, 
#                     alpha[backprop_mask], 
#                     beta[backprop_mask], 
#                     gamma[backprop_mask]
#                 )
#                 block_detach = wigner_D(
#                     lmax, 
#                     alpha_detach, 
#                     beta_detach, 
#                     gamma_detach
#                 )
#                 end = start + block.size()[1]
#                 wigner[   backprop_mask, start:end, start:end] = block
#                 wigner[(~backprop_mask), start:end, start:end] = block_detach
#             elif not self.use_rotation_mask:
#                 block = wigner_D(lmax, alpha, beta, gamma)
#                 end = start + block.size()[1]
#                 wigner[:, start:end, start:end] = block
#             start = end
#         if self.use_rotation_mask:
#             return wigner
#         else:
#             return wigner.detach()

#     def extra_repr(self):
#         return 'mmax={}, lmax={}, use_sparse_matmul={}'.format(
#             self.mmax, self.lmax, self.use_sparse_matmul
#         )
    
    
class SO3Grid(torch.nn.Module):
    """
    Helper functions for grid representation of the irreps

    Args:
        lmax (int):   Maximum degree of the spherical harmonics
        mmax (int):   Maximum order of the spherical harmonics
        normalization (str):    Default: 'component'
                                How grid samples are normalized.
        resolution_list (list:int):  
                                Default: None
                                List of grid resolutions corresponding to `lat_resolution` and `long_resolution`.
                                Set to `None` to use default resolutions.
        use_m_primary (bool):   Default: False
                                Whether to change the layout of m components.
                                If `False`, the layout of m is (0), (-1, 0, +1), (-2, -1, 0, +1, +2), ...
                                If `True`, the layout of m is (0, 0, ...), (1, 1, ...), ...
                                The second one is used in SO(2) linear operations to avoid redundant 
                                matrix multiplications.
    """
    def __init__(
        self,
        lmax,
        mmax,
        normalization='component',
        resolution_list=None,
        use_m_primary=False
    ):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.use_m_primary = use_m_primary
        self.lat_resolution = 2 * (self.lmax + 1)
        if lmax == mmax:
            self.long_resolution = 2 * (self.mmax + 1) + 1
        else:
            self.long_resolution = 2 * (self.mmax) + 1
        if resolution_list is not None:
            assert isinstance(resolution_list, list)
            resolution_list = copy.deepcopy(resolution_list)
            self.lat_resolution = resolution_list[0]
            self.long_resolution = resolution_list[1]

        mapping = CoefficientMappingModule(
            lmax=self.lmax,
            mmax=self.lmax,
            use_rotate_inv_rescale=False
        )

        to_grid = ToS2Grid(
            self.lmax,
            (self.lat_resolution, self.long_resolution),
            normalization=normalization, #normalization="integral",
            device='cpu',
        )
        to_grid_mat = torch.einsum("mbi, am -> bai", to_grid.shb, to_grid.sha).detach()
        # rescale based on mmax
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l ** 2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                to_grid_mat[:, :, start_idx : (start_idx + length)] = to_grid_mat[:, :, start_idx : (start_idx + length)] * rescale_factor
        to_grid_mat = to_grid_mat[:, :, mapping.coefficient_idx(self.lmax, self.mmax)]

        from_grid = FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization=normalization, #normalization="integral",
            device='cpu',
        )
        from_grid_mat = torch.einsum("am, mbi -> bai", from_grid.sha, from_grid.shb).detach()
        # rescale based on mmax
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l ** 2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                from_grid_mat[:, :, start_idx : (start_idx + length)] = from_grid_mat[:, :, start_idx : (start_idx + length)] * rescale_factor
        from_grid_mat = from_grid_mat[:, :, mapping.coefficient_idx(self.lmax, self.mmax)]

        # flatten and permute
        to_grid_mat   = to_grid_mat.flatten(0, 1)
        from_grid_mat = from_grid_mat.flatten(0, 1)
        from_grid_mat = from_grid_mat.permute(1, 0)

        # change the layout of m components
        if self.use_m_primary:
            temp = CoefficientMappingModule(self.lmax, self.mmax, False)
            to_grid_mat = torch.einsum('ai, ji -> aj', to_grid_mat, temp.to_m)
            from_grid_mat = torch.einsum('ia, ji -> ja', from_grid_mat, temp.to_m)

        # save tensors and they will be moved to GPU
        self.register_buffer('to_grid_mat',   to_grid_mat)
        self.register_buffer('from_grid_mat', from_grid_mat)


    # Compute matrices to transform irreps to grid
    def get_to_grid_mat(self):
        return self.to_grid_mat


    # Compute matrices to transform grid to irreps
    def get_from_grid_mat(self):
        return self.from_grid_mat


    # Compute grid from irreps representation
    def to_grid(self, embedding):
        #grid = torch.matmul(self.to_grid_mat, embedding)
        grid = torch.einsum('aj, njc -> nac', self.to_grid_mat, embedding)
        #grid = torch.einsum('baj, njc -> nbac', self.to_grid_mat, embedding)
        return grid


    # Compute irreps from grid representation
    def from_grid(self, grid):
        #embedding = torch.matmul(self.from_grid_mat, grid)
        embedding = torch.einsum('ja, nac -> njc', self.from_grid_mat, grid)
        #embedding = torch.einsum('aj, nac -> njc', self.from_grid_mat, grid)
        #embedding = torch.einsum('baj, nbac -> njc', self.from_grid_mat, grid)
        return embedding


    def extra_repr(self):
        return 'lmax={}, mmax={}, lat_resolution={}, long_resolution={}, use_m_primary={}'.format(self.lmax, self.mmax, self.lat_resolution, self.long_resolution, self.use_m_primary)


class SO3Linear(torch.nn.Module):
    def __init__(
            self, 
            mmax: int,
            lmax: int, 
            in_features: int,
            out_features: int,
            bias: bool = True, # only so3 l=0 or so2 m=0 can have bias
            use_m_primary: bool = False,
        ):
        super().__init__()
        self.mmax = mmax
        self.lmax = lmax
        self.in_features = in_features
        self.out_features = out_features
        self.use_m_primary = use_m_primary
        self.weight = torch.nn.Parameter(torch.randn((self.lmax + 1), out_features, in_features))
        self.bias = torch.nn.Parameter(torch.zeros(1, 1, out_features)) if bias else None
        _, expand_index = so3_expand_index(lmax, lmax)
        if self.use_m_primary:
            temp = CoefficientMappingModule(self.lmax, self.lmax, False)
            expand_index = torch.einsum('i, ji -> j', expand_index.to(temp.to_m.dtype), temp.to_m).long()
            _sum = lmax+1
            for m in range(1, mmax+1):
                _sum += 2*(lmax+1-m)
            expand_index = expand_index[:_sum]
        self.register_buffer('expand_index', expand_index, persistent=False)
        self.alpha = 1.0 / math.sqrt(self.in_features)

    def forward(self, inputs):
        weight = self.weight * self.alpha
        weight = torch.index_select(weight, dim=0, index=self.expand_index) # [so3_m, C_out, C_in]
        outputs = torch.einsum('bmi, moi -> bmo', inputs, weight)      
        if self.bias is not None:
            outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + self.bias
        return outputs


    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax}, bias={(self.bias is not None)})"
    

class SO3VstpGrid(torch.nn.Module):
    """
    Using the same grid convention as e3nn ToS2Grid / FromS2Grid:
        beta_i  = pi * (i + 0.5) / res_beta
        alpha_j = 2pi * j / res_alpha
    """

    def __init__(
        self,
        lmax,
        mmax=None,
        normalization="component",
        resolution_list=None,
        use_m_primary=False,
        eps=1e-8,
        normalize_products=True,
    ):
        super().__init__()

        self.lmax = lmax
        self.mmax = lmax if mmax is None else mmax
        self.normalization = normalization
        self.use_m_primary = use_m_primary
        self.eps = eps
        self.normalize_products = normalize_products

        self.lat_resolution = 2 * (self.lmax + 1)

        if self.lmax == self.mmax:
            self.long_resolution = 2 * (self.mmax + 1) + 1
        else:
            self.long_resolution = 2 * self.mmax + 1

        if resolution_list is not None:
            assert isinstance(resolution_list, list)
            resolution_list = copy.deepcopy(resolution_list)
            self.lat_resolution = resolution_list[0]
            self.long_resolution = resolution_list[1]

        mapping = CoefficientMappingModule(
            lmax=self.lmax,
            mmax=self.lmax,
            use_rotate_inv_rescale=False,
        )

        to_grid = ToS2Grid(
            self.lmax,
            (self.lat_resolution, self.long_resolution),
            normalization=normalization,
        )

        from_grid = FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization=normalization,
        )

        to_grid_mat = torch.einsum(
            "mbi,am->bai",
            to_grid.shb,
            to_grid.sha,
        ).detach()

        from_grid_mat = torch.einsum(
            "am,mbi->bai",
            from_grid.sha,
            from_grid.shb,
        ).detach()

        d_beta_mat, d_alpha_mat, sin_beta = self._build_derivative_matrices(to_grid)

        if self.lmax != self.mmax:
            for l in range(self.lmax + 1):
                if l <= self.mmax:
                    continue

                start_idx = l ** 2
                length = 2 * l + 1
                rescale_factor = math.sqrt(
                    length / (2 * self.mmax + 1)
                )

                to_grid_mat[:, :, start_idx:start_idx + length] *= rescale_factor
                from_grid_mat[:, :, start_idx:start_idx + length] *= rescale_factor
                d_beta_mat[:, :, start_idx:start_idx + length] *= rescale_factor
                d_alpha_mat[:, :, start_idx:start_idx + length] *= rescale_factor

        coeff_idx = mapping.coefficient_idx(
            self.lmax,
            self.mmax,
        )
        coeff_l_index = mapping.l_harmonic[coeff_idx].clone()

        to_grid_mat = to_grid_mat[:, :, coeff_idx]
        from_grid_mat = from_grid_mat[:, :, coeff_idx]
        d_beta_mat = d_beta_mat[:, :, coeff_idx]
        d_alpha_mat = d_alpha_mat[:, :, coeff_idx]

        to_grid_mat = to_grid_mat.flatten(0, 1)
        from_grid_mat = from_grid_mat.flatten(0, 1).permute(1, 0)

        d_beta_mat = d_beta_mat.flatten(0, 1)
        d_alpha_mat = d_alpha_mat.flatten(0, 1)

        sin_beta = sin_beta[:, None].expand(
            self.lat_resolution,
            self.long_resolution,
        )
        sin_beta = sin_beta.flatten()

        if self.use_m_primary:
            temp = CoefficientMappingModule(
                self.lmax,
                self.mmax,
                False,
            )
            source_idx = temp.to_m.argmax(dim=1)
            coeff_l_index = coeff_l_index[source_idx]

            to_grid_mat = torch.einsum(
                "ai,ji->aj",
                to_grid_mat,
                temp.to_m,
            )

            from_grid_mat = torch.einsum(
                "ia,ji->ja",
                from_grid_mat,
                temp.to_m,
            )

            d_beta_mat = torch.einsum(
                "ai,ji->aj",
                d_beta_mat,
                temp.to_m,
            )

            d_alpha_mat = torch.einsum(
                "ai,ji->aj",
                d_alpha_mat,
                temp.to_m,
            )

        self.register_buffer("to_grid_mat", to_grid_mat)
        self.register_buffer("from_grid_mat", from_grid_mat)

        self.register_buffer("d_beta_to_grid_mat", d_beta_mat)
        self.register_buffer("d_alpha_to_grid_mat", d_alpha_mat)

        self.register_buffer(
            "inv_sin_beta",
            1.0 / sin_beta.clamp_min(self.eps),
        )
        self.register_buffer("coeff_l_index", coeff_l_index.long())

        if self.normalize_products:
            sym_norm, anti_norm = self._build_product_normalizers(
                to_grid_mat,
                from_grid_mat,
                d_beta_mat,
                d_alpha_mat,
                self.inv_sin_beta,
                self.coeff_l_index,
            )
        else:
            shape = (1, from_grid_mat.size(0), 1)
            sym_norm = torch.ones(shape, dtype=from_grid_mat.dtype)
            anti_norm = torch.ones(shape, dtype=from_grid_mat.dtype)

        self.register_buffer("symmetric_output_norm", sym_norm)
        self.register_buffer("antisymmetric_output_norm", anti_norm)
        self.register_buffer(
            "d_beta_grid_mat",
            torch.einsum("ai,ib->ab", d_beta_mat, from_grid_mat),
        )
        self.register_buffer(
            "d_alpha_grid_mat",
            torch.einsum("ai,ib->ab", d_alpha_mat, from_grid_mat),
        )
        self.register_buffer(
            "antisymmetric_grid_norm",
            self._build_grid_antisymmetric_normalizer(
                self.to_grid_mat,
                self.d_beta_grid_mat,
                self.d_alpha_grid_mat,
                self.inv_sin_beta,
            ),
        )

    def _build_derivative_matrices(self, to_grid,):
        """
        Build analytic spectral derivative matrices:
            d_beta_mat[b, a, i]  = ∂beta Y_i(beta_b, alpha_a)
            d_alpha_mat[b, a, i] = ∂alpha Y_i(beta_b, alpha_a)
        This uses autograd on the analytic spherical harmonics once during
        module construction.
        """

        betas = to_grid.betas.detach().clone()
        alphas = to_grid.alphas.detach().clone()

        beta_grid, alpha_grid = torch.meshgrid(
            betas,
            alphas,
            indexing="ij",
        )

        beta_flat = beta_grid.reshape(-1).detach().clone()
        alpha_flat = alpha_grid.reshape(-1).detach().clone()

        beta_flat.requires_grad_(True)
        alpha_flat.requires_grad_(True)

        xyz = o3.angles_to_xyz(
            alpha_flat,
            beta_flat,
        )

        Y = o3.spherical_harmonics(
            list(range(self.lmax + 1)),
            xyz,
            normalize=True,
            normalization=self.normalization,
        )

        num_grid, num_coeffs = Y.shape

        d_beta_cols = []
        d_alpha_cols = []

        for i in range(num_coeffs):
            grad_beta, grad_alpha = torch.autograd.grad(
                Y[:, i].sum(),
                [beta_flat, alpha_flat],
                retain_graph=True,
                create_graph=False,
            )

            d_beta_cols.append(grad_beta)
            d_alpha_cols.append(grad_alpha)

        d_beta = torch.stack(
            d_beta_cols,
            dim=1,
        )

        d_alpha = torch.stack(
            d_alpha_cols,
            dim=1,
        )

        d_beta = d_beta.reshape(
            self.lat_resolution,
            self.long_resolution,
            num_coeffs,
        )

        d_alpha = d_alpha.reshape(
            self.lat_resolution,
            self.long_resolution,
            num_coeffs,
        )

        sin_beta = torch.sin(betas)

        return (
            d_beta.detach(),
            d_alpha.detach(),
            sin_beta.detach(),
        )

    def _degree_average_norm(
        self,
        variance,
        coeff_l_index,
    ):
        norm = torch.zeros_like(variance)
        for l in range(self.lmax + 1):
            mask = coeff_l_index == l
            if not bool(mask.any()):
                continue
            degree_var = variance[mask].mean().clamp_min(0.0)
            if degree_var.item() > self.eps:
                norm[mask] = torch.rsqrt(degree_var)
        return norm.view(1, -1, 1)

    def _build_product_normalizers(
        self,
        to_grid_mat,
        from_grid_mat,
        d_beta_mat,
        d_alpha_mat,
        inv_sin_beta,
        coeff_l_index,
    ):
        """
        Estimate per-degree output normalizers for random unit-variance inputs.

        This keeps the integral GTP and VSTP products close to CGTP scale at
        initialization without introducing dense path-wise factors that would
        destroy the factorized grid implementation.
        """
        to_grid_mat = to_grid_mat.double()
        from_grid_mat = from_grid_mat.double()
        d_beta_mat = d_beta_mat.double()
        d_alpha_mat = d_alpha_mat.double()
        inv_sin_beta = inv_sin_beta.double()

        signal_cov = to_grid_mat @ to_grid_mat.T
        sym_grid_cov = signal_cov.square()
        sym_var = torch.einsum(
            "ia,ib,ab->i",
            from_grid_mat,
            from_grid_mat,
            sym_grid_cov,
        ).clamp_min(0.0)

        beta_beta = d_beta_mat @ d_beta_mat.T
        alpha_alpha = d_alpha_mat @ d_alpha_mat.T
        beta_alpha = d_beta_mat @ d_alpha_mat.T
        alpha_beta = d_alpha_mat @ d_beta_mat.T

        anti_grid_cov = (
            beta_beta * alpha_alpha
            + alpha_alpha * beta_beta
            - beta_alpha * alpha_beta
            - alpha_beta * beta_alpha
        )
        anti_grid_cov = anti_grid_cov * inv_sin_beta[:, None] * inv_sin_beta[None, :]
        anti_var = torch.einsum(
            "ia,ib,ab->i",
            from_grid_mat,
            from_grid_mat,
            anti_grid_cov,
        ).clamp_min(0.0)

        sym_norm = self._degree_average_norm(sym_var, coeff_l_index)
        anti_norm = self._degree_average_norm(anti_var, coeff_l_index)
        return sym_norm.float(), anti_norm.float()

    def _build_grid_antisymmetric_normalizer(
        self,
        to_grid_mat,
        d_beta_grid_mat,
        d_alpha_grid_mat,
        inv_sin_beta,
    ):
        to_grid_mat = to_grid_mat.double()
        d_beta_grid_mat = d_beta_grid_mat.double()
        d_alpha_grid_mat = d_alpha_grid_mat.double()
        inv_sin_beta = inv_sin_beta.double()

        signal_cov = to_grid_mat @ to_grid_mat.T
        sym_variance = signal_cov.diagonal().square().mean().clamp_min(0.0)

        beta_cov = d_beta_grid_mat @ signal_cov @ d_beta_grid_mat.T
        alpha_cov = d_alpha_grid_mat @ signal_cov @ d_alpha_grid_mat.T
        beta_alpha_cov = d_beta_grid_mat @ signal_cov @ d_alpha_grid_mat.T

        anti_variance = 2.0 * (
            beta_cov.diagonal() * alpha_cov.diagonal()
            -
            beta_alpha_cov.diagonal().square()
        )
        anti_variance = anti_variance * inv_sin_beta.square()
        anti_variance = anti_variance.mean().clamp_min(0.0)

        if anti_variance.item() <= self.eps:
            return torch.tensor(0.0, dtype=torch.get_default_dtype())
        return torch.sqrt(sym_variance / anti_variance).to(dtype=torch.get_default_dtype())

    def get_to_grid_mat(self):
        return self.to_grid_mat

    def get_from_grid_mat(self):
        return self.from_grid_mat

    def to_grid(self, embedding):
        """
        embedding:
            [num_nodes, num_coefficients, channels]

        return:
            [num_nodes, num_grid_points, channels]
        """
        return torch.matmul(self.to_grid_mat.to(dtype=embedding.dtype), embedding)

    def from_grid(self, grid):
        """
        grid:
            [num_nodes, num_grid_points, channels]

        return:
            [num_nodes, num_coefficients, channels]
        """
        return torch.matmul(self.from_grid_mat.to(dtype=grid.dtype), grid)

    def d_beta_to_grid(self, embedding):
        """
        Compute ∂beta f
        """
        return torch.matmul(self.d_beta_to_grid_mat.to(dtype=embedding.dtype), embedding)

    def d_alpha_to_grid(self, embedding):
        """
        Compute ∂alpha f
        """
        return torch.matmul(self.d_alpha_to_grid_mat.to(dtype=embedding.dtype), embedding)

    def d_beta_grid(self, grid):
        return torch.matmul(self.d_beta_grid_mat.to(dtype=grid.dtype), grid)

    def d_alpha_grid(self, grid):
        return torch.matmul(self.d_alpha_grid_mat.to(dtype=grid.dtype), grid)

    def symmetric_grid_product(self, x_grid, y_grid):
        return x_grid * y_grid

    def antisymmetric_grid_product(self, x_grid, y_grid):
        x_beta = self.d_beta_grid(x_grid)
        x_alpha = self.d_alpha_grid(x_grid)
        y_beta = self.d_beta_grid(y_grid)
        y_alpha = self.d_alpha_grid(y_grid)
        inv_sin_beta = self.inv_sin_beta.to(x_grid.dtype).view(1, -1, 1)
        bracket_grid = (
            x_beta * y_alpha
            -
            x_alpha * y_beta
        ) * inv_sin_beta
        return bracket_grid * self.antisymmetric_grid_norm.to(dtype=bracket_grid.dtype)

    def full_grid_product(
        self,
        x_grid,
        y_grid,
        symmetric_scale=1.0,
        antisym_scale=1.0,
    ):
        sym = self.symmetric_grid_product(x_grid, y_grid)
        anti = self.antisymmetric_grid_product(x_grid, y_grid)
        return sym + anti

    def symmetric_product(self, x, y):
        """
        symmetric GTP
        """
        x_grid = self.to_grid(x)
        y_grid = self.to_grid(y)

        product_grid = x_grid * y_grid

        out = self.from_grid(product_grid)
        return out * self.symmetric_output_norm.to(dtype=out.dtype)

    def antisymmetric_product(self, x, y):
        """
        Antisymmetric VSTP
        """
        x_beta = self.d_beta_to_grid(x)
        x_alpha = self.d_alpha_to_grid(x)

        y_beta = self.d_beta_to_grid(y)
        y_alpha = self.d_alpha_to_grid(y)

        inv_sin_beta = self.inv_sin_beta.to(x.dtype).view(1, -1, 1)

        bracket_grid = (
            x_beta * y_alpha
            -
            x_alpha * y_beta
        ) * inv_sin_beta

        out = self.from_grid(bracket_grid)
        return out * self.antisymmetric_output_norm.to(dtype=out.dtype)

    def full_vstp_product(
        self,
        x,
        y,
        symmetric_scale=1.0,
        antisym_scale=1.0,
    ):
        """
        Theorem 2: GTP + VSTP
        """
        sym = self.symmetric_product(x, y)
        anti = self.antisymmetric_product(x, y)

        return sym + anti # TODO

        # return symmetric_scale * sym + antisym_scale * anti

    def extra_repr(self):
        return (
            f"lmax={self.lmax}, "
            f"mmax={self.mmax}, "
            f"lat_resolution={self.lat_resolution}, "
            f"long_resolution={self.long_resolution}, "
            f"use_m_primary={self.use_m_primary}"
        )