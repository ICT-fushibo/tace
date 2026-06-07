# import torch

# try:
#     import triton
#     import triton.language as tl

#     TRITON_AVAILABLE = True
# except ImportError:
#     TRITON_AVAILABLE = False

# if TRITON_AVAILABLE:

#     @triton.jit
#     def _csr_bmm_kernel(
#         matrix,
#         input,
#         row_ptr,
#         col_idx,
#         output,
#         num_rows,
#         num_channels,
#         matrix_stride_b,
#         matrix_stride_r,
#         matrix_stride_k,
#         input_stride_b,
#         input_stride_k,
#         input_stride_c,
#         output_stride_b,
#         output_stride_r,
#         output_stride_c,
#         MAX_NNZ: tl.constexpr,
#         BLOCK_C: tl.constexpr,
#     ):
#         batch_row = tl.program_id(0)
#         channel_block = tl.program_id(1)
#         batch = batch_row // num_rows
#         row = batch_row - batch * num_rows
#         channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
#         channel_mask = channels < num_channels
#         start = tl.load(row_ptr + row)
#         end = tl.load(row_ptr + row + 1)
#         accumulator = tl.zeros((BLOCK_C,), dtype=tl.float32)

#         for offset in range(MAX_NNZ):
#             active = start + offset < end
#             col = tl.load(col_idx + start + offset, mask=active, other=0)
#             value = tl.load(
#                 matrix
#                 + batch * matrix_stride_b
#                 + row * matrix_stride_r
#                 + col * matrix_stride_k,
#                 mask=active,
#                 other=0.0,
#             )
#             vector = tl.load(
#                 input
#                 + batch * input_stride_b
#                 + col * input_stride_k
#                 + channels * input_stride_c,
#                 mask=active & channel_mask,
#                 other=0.0,
#             )
#             accumulator += value * vector

#         tl.store(
#             output
#             + batch * output_stride_b
#             + row * output_stride_r
#             + channels * output_stride_c,
#             accumulator,
#             mask=channel_mask,
#         )


#     @triton.jit
#     def _csr_matrix_grad_kernel(
#         input,
#         grad_output,
#         row_ptr,
#         col_idx,
#         grad_matrix,
#         num_rows,
#         num_channels,
#         input_stride_b,
#         input_stride_k,
#         input_stride_c,
#         grad_output_stride_b,
#         grad_output_stride_r,
#         grad_output_stride_c,
#         grad_matrix_stride_b,
#         grad_matrix_stride_r,
#         grad_matrix_stride_k,
#         MAX_NNZ: tl.constexpr,
#         BLOCK_C: tl.constexpr,
#     ):
#         batch_row = tl.program_id(0)
#         offset = tl.program_id(1)
#         batch = batch_row // num_rows
#         row = batch_row - batch * num_rows
#         start = tl.load(row_ptr + row)
#         end = tl.load(row_ptr + row + 1)
#         active = start + offset < end
#         col = tl.load(col_idx + start + offset, mask=active, other=0)
#         channels = tl.arange(0, BLOCK_C)
#         channel_mask = channels < num_channels
#         x = tl.load(
#             input
#             + batch * input_stride_b
#             + col * input_stride_k
#             + channels * input_stride_c,
#             mask=active & channel_mask,
#             other=0.0,
#         )
#         grad = tl.load(
#             grad_output
#             + batch * grad_output_stride_b
#             + row * grad_output_stride_r
#             + channels * grad_output_stride_c,
#             mask=active & channel_mask,
#             other=0.0,
#         )
#         value = tl.sum(x * grad, axis=0)
#         tl.store(
#             grad_matrix
#             + batch * grad_matrix_stride_b
#             + row * grad_matrix_stride_r
#             + col * grad_matrix_stride_k,
#             value,
#             mask=active,
#         )


# def _triton_csr_bmm(matrix, input, row_ptr, col_idx, max_nnz):
#     output = input.new_empty(matrix.shape[0], matrix.shape[1], input.shape[2])
#     block_c = min(triton.next_power_of_2(input.shape[2]), 1024)
#     grid = (matrix.shape[0] * matrix.shape[1], triton.cdiv(input.shape[2], block_c))
#     _csr_bmm_kernel[grid](
#         matrix,
#         input,
#         row_ptr,
#         col_idx,
#         output,
#         matrix.shape[1],
#         input.shape[2],
#         *matrix.stride(),
#         *input.stride(),
#         *output.stride(),
#         MAX_NNZ=max_nnz,
#         BLOCK_C=block_c,
#     )
#     return output


# def _triton_csr_matrix_grad(input, grad_output, matrix, row_ptr, col_idx, max_nnz):
#     grad_matrix = torch.zeros_like(matrix)
#     block_c = triton.next_power_of_2(input.shape[2])
#     grid = (matrix.shape[0] * matrix.shape[1], max_nnz)
#     _csr_matrix_grad_kernel[grid](
#         input,
#         grad_output,
#         row_ptr,
#         col_idx,
#         grad_matrix,
#         matrix.shape[1],
#         input.shape[2],
#         *input.stride(),
#         *grad_output.stride(),
#         *grad_matrix.stride(),
#         MAX_NNZ=max_nnz,
#         BLOCK_C=block_c,
#     )
#     return grad_matrix


# class _FusedSparseWignerBMM(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         matrix,
#         input,
#         row_ptr,
#         col_idx,
#         transpose_row_ptr,
#         transpose_col_idx,
#         max_nnz,
#         transpose_max_nnz,
#     ):
#         ctx.save_for_backward(
#             matrix,
#             input,
#             row_ptr,
#             col_idx,
#             transpose_row_ptr,
#             transpose_col_idx,
#         )
#         ctx.max_nnz = max_nnz
#         ctx.transpose_max_nnz = transpose_max_nnz
#         return _triton_csr_bmm(matrix, input, row_ptr, col_idx, max_nnz)

#     @staticmethod
#     def backward(ctx, grad_output):
#         (
#             matrix,
#             input,
#             row_ptr,
#             col_idx,
#             transpose_row_ptr,
#             transpose_col_idx,
#         ) = ctx.saved_tensors
#         grad_matrix = grad_input = None
#         if ctx.needs_input_grad[0]:
#             grad_matrix = _triton_csr_matrix_grad(
#                 input,
#                 grad_output,
#                 matrix,
#                 row_ptr,
#                 col_idx,
#                 ctx.max_nnz,
#             )
#         if ctx.needs_input_grad[1]:
#             grad_input = _triton_csr_bmm(
#                 matrix.transpose(1, 2),
#                 grad_output,
#                 transpose_row_ptr,
#                 transpose_col_idx,
#                 ctx.transpose_max_nnz,
#             )
#         return grad_matrix, grad_input, None, None, None, None, None, None


# def fused_sparse_wigner_bmm(
#     matrix,
#     input,
#     row_ptr,
#     col_idx,
#     transpose_row_ptr,
#     transpose_col_idx,
#     max_nnz,
#     transpose_max_nnz,
# ):
#     if (
#         not TRITON_AVAILABLE
#         or matrix.device.type != "cuda"
#         or matrix.dtype not in (torch.float16, torch.bfloat16, torch.float32)
#         or input.shape[-1] > 1024
#     ):
#         return torch.bmm(matrix, input)
#     return _FusedSparseWignerBMM.apply(
#         matrix,
#         input,
#         row_ptr,
#         col_idx,
#         transpose_row_ptr,
#         transpose_col_idx,
#         max_nnz,
#         transpose_max_nnz,
#     )
