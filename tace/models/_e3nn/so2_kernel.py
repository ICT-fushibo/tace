import math
from typing import NamedTuple, List, Dict, Tuple, Any, Callable
import torch
from torch_geometric.utils import segment, scatter

import triton
import triton.language as tl

from tace.models._eqt.equitorch.irreps import check_irreps


def extract_batch_segments(keys: List[List[int]]):
    r"""
    Process sorted integer key lists to generate batch indices, boundary pointers, and key values.

    Parameters
    ----------
    keys : List[List[int]]
        A list of sorted integer key lists. All lists must have the same length.

    Returns
    -------
    batch : List[int]
        A list where each element indicates the batch index it belongs to.
    seg : List[int]
        A list of boundary pointers indicating the start and end of each batch.
    val : List[List[int]]
        A list of lists containing the key values at the boundary points for each key list.

    Notes
    -----
    - The input key lists must be sorted in ascending order.
    - If the input is empty, the function returns empty lists for `batch`, `seg`, and `val`.

    Examples
    --------
    >>> keys = [
    ...     [1, 1, 2, 2],
    ...     [1, 1, 2, 2]
    ... ]
    >>> extract_batch_seg_native(keys)
    ([0, 0, 1, 1], [0, 2, 4], [[1, 2], [1, 2]])

    >>> keys = [
    ...     [5, 5, 5],
    ...     [5, 5, 5]
    ... ]
    >>> extract_batch_seg_native(keys)
    ([0, 0, 0], [0, 3], [[5], [5]])

    >>> keys = [
    ...     [1, 1, 2, 3, 3],
    ...     [1, 2, 2, 3, 3]
    ... ]
    >>> extract_batch_seg_native(keys)
    ([0, 1, 2, 3, 3], [0, 1, 2, 3, 5], [[1, 1, 2, 3], [1, 2, 2, 3]])
    """
    if not keys or not keys[0]:
        return [], [], []
    
    length = len(keys[0])
    seg = [0]  # 初始化分界指针
    
    # 生成分界指针
    for i in range(length):
        last_idx = seg[-1]
        # 检查所有键在当前索引i处是否与上一个分界点的值不同
        if any(key[i] != key[last_idx] for key in keys):
            seg.append(i)
    
    seg.append(length)  # 添加最终边界
    
    # 生成批次索引
    batch = [0] * length
    for batch_idx in range(1, len(seg)):
        start = seg[batch_idx-1]
        end = seg[batch_idx]
        for i in range(start, end):
            batch[i] = batch_idx - 1
    
    # 提取分界点键值
    val = [
        [key[boundary] for boundary in seg[:-1]]  # 排除最后一个边界
        for key in keys
    ]
    
    return batch, seg, val


def sort_by_column_key(to_sort: List[List[Any]], key: List[List[Any]] = None) -> List[List[Any]]:
    """
    Sort the columns of the first 2D list based on the column-wise lexicographical order of the key 2D list.

    Parameters
    ----------
    to_sort : List[List[Any]]
        The first 2D list whose columns are to be sorted.
    key : List[List[Any]]
        The key 2D list used to determine the sorting order of columns.

    Returns
    -------
    List[List[Any]]
        The first 2D list with columns sorted according to the column-wise lexicographical order of the key.

    Raises
    ------
    ValueError
        If either `to_sort` or `key` is empty, or if their lengths do not match.

    Examples
    --------
    >>> to_sort = [[1, 2, 3], 
    ...            [4, 5, 6]]
    >>> key = [[2, 1, 3], 
    ...        [1, 3, 2]]
    >>> sort_by_column_key(to_sort, key)
    [[2, 1, 3], 
     [5, 4, 6]]
    """

    if key is None:
        key = to_sort

    # 将两个列表转置为列优先的形式
    to_sort_transposed = list(zip(*to_sort))
    key_transposed = list(zip(*key))
    # 将转置后的列表组合成 [(to_sort_col, key_col)] 的形式
    combined = list(zip(to_sort_transposed, key_transposed))
    # 根据 key 的列字典序进行排序
    sorted_combined = sorted(combined, key=lambda x: x[1])
    # 提取排序后的 to_sort 的列
    sorted_to_sort_transposed = [item[0] for item in sorted_combined]
    # 将转置后的结果还原为原始形式
    sorted_to_sort = list(zip(*sorted_to_sort_transposed))
    # 将元组转换为列表
    sorted_to_sort = [list(row) for row in sorted_to_sort]

    return sorted_to_sort


def sparse_product_info(index1=None, index2=None, index=None, scale=None, out_size=None):

    assert index1 is not None or index2 is not None or index is not None, "At least one of the indices should be not None"
    # assert index is None or index_out is None, "index and index_out cannot be both not None " 

    inter_size = len(index1 or index2 or index)
    index1 = index1 or list(range(inter_size))
    index2 = index2 or list(range(inter_size))
    index = index or list(range(inter_size))
    if scale is not None:
        index_MM1M2, index1_MM1M2, index2_MM1M2, scale_MM1M2 = sort_by_column_key(
            [index, index1, index2, scale])
        batch_M_MM1M2, seg_M_MM1M2, (index_M,) = extract_batch_segments(
            [index_MM1M2]
        )
    else:
        index_MM1M2, index1_MM1M2, index2_MM1M2 = sort_by_column_key(
            [index, index1, index2])
        batch_M_MM1M2, seg_M_MM1M2, (index_M,) = extract_batch_segments(
            [index_MM1M2]
        )
        scale_MM1M2 = None

    if out_size is None:
        out_size = len(index_M)

    if index_M[0] != 0:
        seg_M_MM1M2 = [0] * index_M[0] + seg_M_MM1M2
    if index_M[-1] != out_size:
        seg_M_MM1M2 = seg_M_MM1M2 + [inter_size] * (out_size-index_M[-1]-1)


    if index1_MM1M2 == list(range(inter_size)):
        index1_MM1M2 = None
    if index2_MM1M2 == list(range(inter_size)):
        index2_MM1M2 = None

    seg_new = [0]
    out_count = 0
    for out_M in range(out_size):
        if out_count < len(index_M) and index_M[out_count] == out_M:
            seg_new.append(seg_M_MM1M2[out_count+1])
            out_count+=1
        else:
            seg_new.append(seg_new[-1])

    if seg_new == list(range(out_size+1)):
        seg_new = None
    # if len(seg_M_MM1M2) == len(batch_M_MM1M2)+1:
    #     seg_M_MM1M2 = None
    # if index_M == list(range(out_size)):
    index_M = None
            
    return SparseProductInfo(
        scale=torch.tensor(scale_MM1M2) if scale_MM1M2 is not None else None,
        index1=torch.tensor(index1_MM1M2) if index1_MM1M2 is not None else None,
        index2=torch.tensor(index2_MM1M2) if index2_MM1M2 is not None else None,
        seg_out=torch.tensor(seg_new) if seg_new is not None else None,
        # seg_out=tensor(seg_M_MM1M2) if seg_M_MM1M2 is not None else None,
        index_out=torch.tensor(index_M) if index_M is not None else None,
        out_size=out_size
    )


def sparse_product_infos(index1=None, index2=None, index=None, scale=None, out_size=None, in1_size=None, in2_size=None):
    return (
        sparse_product_info(index1,index2,index,scale,out_size),
        sparse_product_info(index2,index,index1,scale,in1_size),
        sparse_product_info(index,index1,index2,scale,in2_size),
    )


def prepare_so2_linear(
        irreps_out, 
        irreps_in, 
        path=None, 
        path_norm=True, 
        channel_norm=False, 
        channel_scale=1.0
    ):
    if path is None:
        path = [(k, i) for k in range(len(irreps_out)) for i in range(len(irreps_in))]
    # 1. Create mapping from M -> (irrep_idx, l, m) and (irrep_idx, l, m) -> M
    m_to_ilm_in: Dict[int, Tuple[int, int, int]] = {}
    ilm_to_M_in: Dict[Tuple[int, int, int], int] = {}
    current_m_index = 0
    max_l = 0
    for irrep_idx, irrep in enumerate(irreps_in):
        l = irrep.l
        max_l = max(max_l, l)
        for m in range(-l, l + 1):
            m_to_ilm_in[current_m_index] = (irrep_idx, l, m)
            # Use irrep_idx from the original list as part of the key
            ilm_to_M_in[(irrep_idx, l, m)] = current_m_index
            current_m_index += 1

    m_to_klm_out: Dict[int, Tuple[int, int, int]] = {}
    klm_to_M_out: Dict[Tuple[int, int, int], int] = {}
    current_m_index = 0
    max_l = 0
    for irrep_idx, irrep in enumerate(irreps_out):
        l = irrep.l
        max_l = max(max_l, l)
        for m in range(-l, l + 1):
            m_to_klm_out[current_m_index] = (irrep_idx, l, m)
            # Use irrep_idx from the original list as part of the key
            klm_to_M_out[(irrep_idx, l, m)] = current_m_index
            current_m_index += 1

    indices1: List[int] = [] # Index into x (input1)
    indices2: List[int] = [] # Index into weights (input2)
    indices_out: List[int] = [] # Output index M
    scales: List[float] = [] # Scale factor: 

    # 2. Iterate through output indices M and generate sparse interactions
    kim_to_w_idx: Dict[Tuple[int, int, int], int] = {}
    w_idx_to_kim: Dict[int, Tuple[int, int, int]] = {}
    path_count_km_out: Dict[Tuple[int, int], int] = {}
    # for ir_idx_out, ir_out in enumerate(irreps_out):
    #     for ir_idx_in, ir_in in enumerate(irreps_in):
    for ir_idx_out, ir_idx_in in path:
        ir_out = irreps_out[ir_idx_out]
        ir_in = irreps_in[ir_idx_in]
        for m in range(0, min(ir_out.l, ir_in.l)+1):
            if m == 0:
                w_idx_to_kim[len(w_idx_to_kim)] = (ir_idx_out, ir_idx_in, m)
                kim_to_w_idx[(ir_idx_out, ir_idx_in, m)] = len(w_idx_to_kim) - 1
                path_count_km_out[(ir_idx_out, m)] = path_count_km_out.get((ir_idx_out, m),0)+1
            else:
                w_idx_to_kim[len(w_idx_to_kim)] = (ir_idx_out, ir_idx_in, -m)
                kim_to_w_idx[(ir_idx_out, ir_idx_in, -m)] = len(w_idx_to_kim) - 1
                path_count_km_out[(ir_idx_out, m)] = path_count_km_out.get((ir_idx_out, -m),0)+1
                w_idx_to_kim[len(w_idx_to_kim)] = (ir_idx_out, ir_idx_in, m)
                kim_to_w_idx[(ir_idx_out, ir_idx_in, m)] = len(w_idx_to_kim) - 1
                path_count_km_out[(ir_idx_out, m)] = path_count_km_out.get((ir_idx_out, -m),0)+1
    
    for ir_idx_out, ir_idx_in in path:
        ir_out = irreps_out[ir_idx_out]
        ir_in = irreps_in[ir_idx_in]
        for m in range(0, min(ir_out.l, ir_in.l)+1):
            if m == 0:
                if path_norm:
                    scales.append(path_count_km_out[(ir_idx_out, m)] ** (-0.5))
                else:
                    scales.append(1.0)  
                indices_out.append(klm_to_M_out[(ir_idx_out, ir_out.l, m)])
                indices1.append(ilm_to_M_in[(ir_idx_in, ir_in.l, m)])
                indices2.append(kim_to_w_idx[(ir_idx_out, ir_idx_in, m)])
            else:
                if path_norm:
                    scales.append((path_count_km_out[(ir_idx_out, m)]*2) ** (-0.5))
                else:
                    scales.append(2**(-0.5))
                indices_out.append(klm_to_M_out[(ir_idx_out, ir_out.l, m)])
                indices1.append(ilm_to_M_in[(ir_idx_in, ir_in.l, m)])         
                indices2.append(kim_to_w_idx[(ir_idx_out, ir_idx_in, m)])         

                if path_norm:
                    scales.append((path_count_km_out[(ir_idx_out, m)]*2) ** (-0.5))
                else:
                    scales.append(2**(-0.5))
                indices_out.append(klm_to_M_out[(ir_idx_out, ir_out.l, -m)])
                indices1.append(ilm_to_M_in[(ir_idx_in, ir_in.l, -m)])         
                indices2.append(kim_to_w_idx[(ir_idx_out, ir_idx_in, m)])         

                if path_norm:
                    scales.append(-(path_count_km_out[(ir_idx_out, m)]*2) ** (-0.5))
                else:
                    scales.append(-2**(-0.5))    
                indices_out.append(klm_to_M_out[(ir_idx_out, ir_out.l, m)])
                indices1.append(ilm_to_M_in[(ir_idx_in, ir_in.l, -m)])         
                indices2.append(kim_to_w_idx[(ir_idx_out, ir_idx_in, -m)])         

                if path_norm:
                    scales.append((path_count_km_out[(ir_idx_out, m)]*2) ** (-0.5))
                else:
                    scales.append(2**(-0.5))
                indices_out.append(klm_to_M_out[(ir_idx_out, ir_out.l, -m)])
                indices1.append(ilm_to_M_in[(ir_idx_in, ir_in.l, m)])         
                indices2.append(kim_to_w_idx[(ir_idx_out, ir_idx_in, -m)])         
    if channel_norm:
        scales = [s * channel_scale for s in scales]
    num_weights = len(w_idx_to_kim)
    return indices1, indices2, indices_out, scales, num_weights


def so2_linear_info(irreps_out, irreps_in, path=None, path_norm=True, channel_norm=False, channel_scale=1.0):
    indices1, indices2, indices_out, scales, num_weights = prepare_so2_linear(irreps_out, irreps_in, path=path, path_norm=path_norm, channel_norm=channel_norm, channel_scale=channel_scale)
    return sparse_product_info(indices1, indices2, indices_out, scales, irreps_out.dim), num_weights


def so2_linear_infos(irreps_out, irreps_in, path=None, path_norm=True, channel_norm=False, channel_scale=1.0):
    indices1, indices2, indices_out, scales, num_weights = prepare_so2_linear(
        irreps_out, 
        irreps_in, 
        path=path, 
        path_norm=path_norm, 
        channel_norm=channel_norm, 
        channel_scale=channel_scale
    )
    return *sparse_product_infos(indices1, indices2, indices_out, scales, irreps_out.dim, irreps_in.dim, num_weights), num_weights


def add_operation_methods(cls):
    # """类装饰器：为 NamedTuple 动态添加 to/cuda/cpu 等方法"""
    def _apply(self, func: Callable[[Any], Any]):
        processed = []
        for field in self._fields:
            value = getattr(self, field)
            # 递归处理 Tensor 或同类型实例
            if isinstance(value, torch.Tensor):
                processed.append(func(value))
            elif isinstance(value, self.__class__):
                processed.append(func(value))  # 递归处理同类型字段
            elif hasattr(value, '_apply'):
                processed.append(value._apply(func))
            else:
                processed.append(value)
        return self.__class__(*processed)
    
    def to(self, *args, **kwargs):
        # 解析参数
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        
        # 定义转换函数
        def convert(t):
            if isinstance(t, torch.Tensor):
                # 只对浮点张量应用dtype转换
                target_dtype = dtype if (dtype is not None and t.is_floating_point()) else None
                return t.to(device=device, dtype=target_dtype, non_blocking=non_blocking)
            return t
        
        return self._apply(convert)

    def cuda(self, *args, **kwargs):
        return self._apply(lambda x: x.cuda(*args, **kwargs))

    def cpu(self, *args, **kwargs):
        return self._apply(lambda x: x.cpu(*args, **kwargs))

    cls._apply = _apply
    cls.to = to
    cls.cuda = cuda
    cls.cpu = cpu
    return cls


@add_operation_methods
class SparseProductInfo(NamedTuple):
    '''
        z_M = sum_{t in Ind*[M]} s_t * x_Ind1[t] * y_Ind2[t]

        or

        z_M = sum_{M1M2} s_{MM1M2} x_M1 * y_M2
    '''
    scale: torch.Tensor | None = None # (num_t,), floating
    index1: torch.Tensor | None = None # (num_t,), int in [0, num_M1)
    index2: torch.Tensor | None = None # (num_t,), int in [0, num_M2)
    seg_out: torch.Tensor | None = None # (num_M_nonzero+1,), increasing int in [0, num_t]
    gather_index: torch.Tensor | None = None # (num_M_nonzero,) int in [0, num_t)
    index_out: torch.Tensor | None = None # (num_M_nonzero,), int in [0, num_M)
    out_size: int | None = None # num_M


def indexed_mul_scale_gather_torch(
        input1: torch.Tensor, input2: torch.Tensor, 
        scale=None, index1=None, index2=None,
        seg=None, gather_index=None, 
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None):
    r"""
    Torch implementation of indexed_mul_scale_gather.
    """
    # print("input1:", None if input1 is None else input1.shape)
    # print("input2:", None if input2 is None else input2.shape)
    # print("scale:", None if scale is None else scale.shape)
    # print(scale)
    # print("index1:", None if index1 is None else index1.shape)
    # print("index2:", None if index2 is None else index2.shape)
    # print("seg:", None if seg is None else seg.shape)
    # print("gather_index:", None if gather_index is None else gather_index.shape)
    # print("index_out:", None if index_out is None else index_out.shape)
    # print("out:", None if out is None else out.shape)
    # Handle input indexing
    if index1 is not None:
        input1 = input1.index_select(-2, index1)
    if index2 is not None:
        input2 = input2.index_select(-2, index2)
    
    # Core multiplication
    inter = input1 * input2
    
    # Apply scaling if provided
    if scale is not None:
        inter = inter * scale.unsqueeze(-1)
    
    # Handle segmentation/gathering
    if seg is not None:
        if gather_index is not None:
            # Gather then segment
            gathered = inter.index_select(-2, gather_index)
            inter = segment(gathered, seg.unsqueeze(0))
  
        else:
            # Direct segment
            inter = segment(inter, seg.unsqueeze(0)) # hidden => sum_path 

    # Handle output indexing
    if index_out is not None:
        inter = scatter(inter, index_out, dim=-2, dim_size=out_size)
    
    # Handle accumulation
    if out_accumulated:
        inter = inter.sum(dim=0)
    
    return inter


def indexed_mul_scale_gather(
        input1, input2, 
        scale=None, index1=None, index2=None,
        seg=None, gather_index=None, 
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None,
        ):

    return indexed_mul_scale_gather_torch(
        input1, input2, scale, index1, index2,
        seg, gather_index, index_out, out,
        out_accumulated, out_size)


class SparseMul(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, 
        input1: torch.Tensor, 
        input2: torch.Tensor,  
        info_fwd: SparseProductInfo, 
        info_bwd1: SparseProductInfo | None = None, 
        info_bwd2: SparseProductInfo | None  = None, 
        out_accumulated: bool = False
    ) -> torch.Tensor:
        ret = indexed_mul_scale_gather(
            input1, input2,
            info_fwd.scale,
            info_fwd.index1,
            info_fwd.index2,
            info_fwd.seg_out,
            info_fwd.gather_index,
            info_fwd.index_out,
            out_accumulated=out_accumulated,
            out_size=info_fwd.out_size,
            )
        ctx.save_for_backward(input1 if input2.requires_grad else None, input2 if input1.requires_grad else None)
        ctx.infos = (info_fwd, info_bwd1, info_bwd2)
        # Determine shared status based on input dimensions heuristic
        ctx.shared1 = input1.ndim < 3
        ctx.shared2 = input2.ndim < 3
        ctx.out_accumulated = out_accumulated # Save for backward logic
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None, None, None
            
        grad = grad_output
        input1, input2 = ctx.saved_tensors
        info_fwd, info_bwd1, info_bwd2 = ctx.infos
        left_shared_fwd = ctx.shared1
        right_shared_fwd = ctx.shared2

        grad1, grad2 = None, None
        if ctx.needs_input_grad[0]:
            # gx = op_bx(y, gz) -> out_accumulated_bx = left_shared_fwd
            out_accumulated_bwd1 = left_shared_fwd
            grad1 = SparseMul.apply(input2, grad, info_bwd1, info_bwd2, info_fwd, out_accumulated_bwd1)

        if ctx.needs_input_grad[1]:
            # gy = op_by(gz, x) -> out_accumulated_by = right_shared_fwd
            out_accumulated_bwd2 = right_shared_fwd
            grad2 = SparseMul.apply(grad, input1, info_bwd2, info_fwd, info_bwd1, out_accumulated_bwd2)
        else:
            grad2 = None

        # Return grads corresponding to input1, input2, out_accumulated, info_fwd, info_bwd1, info_bwd2
        return grad1, grad2, None, None, None, None


def sparse_mul(
        input1: torch.Tensor, 
        input2: torch.Tensor, 
        info_fwd: SparseProductInfo, 
        info_bwd1: SparseProductInfo | None = None, 
        info_bwd2: SparseProductInfo | None  = None, 
        out_accumulated: bool = False,
    ) -> torch.Tensor:
    r"""
    Computes sparse element-wise multiplication using indexed operations.

    This function performs an element-wise product of two input tensors, ``input1`` and ``input2``,
    based on the indexing and scaling information provided in ``info_fwd``.
    The operation is "sparse" in the sense that it uses predefined indexing schemes
    to select elements for multiplication, rather than a dense matrix multiplication.

    The backward pass information can be optionally provided via ``info_bwd1`` and ``info_bwd2``
    for custom gradient calculations if needed.

    Args:
        input1 (torch.Tensor): The first input tensor.
        input2 (torch.Tensor): The second input tensor.
        info_fwd (SparseProductInfo): Contains scaling factors, indices for ``input1`` and ``input2``,
            output segmentation, gather indices, and output indices for the forward pass.
        info_bwd1 (SparseProductInfo, optional): Information for the backward pass with respect to ``input1``.
            Defaults to None.
        info_bwd2 (SparseProductInfo, optional): Information for the backward pass with respect to ``input2``.
            Defaults to None.
        out_accumulated (bool, optional): If ``True``, the output is accumulated into an existing tensor
            (not fully supported by all underlying ops, behavior might vary). Defaults to ``False``.

    Returns:
        torch.Tensor: The result of the sparse element-wise multiplication.
    """
    return SparseMul.apply(input1, input2, info_fwd, info_bwd1, info_bwd2, out_accumulated)


class SO2TensorProduct(torch.nn.Module):
    r"""
    SO(2) equivariant linear layer using tensor products.

    This layer applies an SO(2) equivariant linear transformation, as proposed in `Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs <https://arxiv.org/abs/2302.03655>`_.

    - ``'uu'``: Depthwise/elementwise linear layer.

      - Input shape: ``(..., irreps_in.dim, channels)``
      - Weight shape: ``(num_weights, channels_out)``
      - Output shape: ``(..., irreps_out.dim, channels_out)``

    Args:
        irreps_in (Irreps or str): Irreducible representations of the input tensor.
        irreps_out (Irreps or str): Irreducible representations of the output tensor.
        channels_in (int, optional): Number of channels for the input.
            Required if ``internal_weights=True``.
        channels_out (int, optional): Number of channels for the output.
            Required if ``internal_weights=True``.
        internal_weights (bool, optional): If ``True``, the module manages its own weight parameter.
            If ``False``, weights must be provided during the forward pass. Defaults to ``True``.
        feature_mode (str, optional): Controls the type of linear operation: ``{'uu', 'uv'}``.
            Defaults to ``'uu'``.

            - ``'uu'``: Depthwise/elementwise linear. Assumes ``channels_in == channels_out``.
            - ``'uv'``: Fully connected linear.
        path_norm (bool, optional): Whether to apply path normalization to the weights.
            Normalizes by the square root of the number of paths to each output irrep. Defaults to ``True``.
        channel_norm (bool, optional): Whether to apply channel normalization (specific to ``'uv'`` mode).
            Divides weights by \(\sqrt{\text{channels_in}}\). Note: This interacts with ``path_norm``.
            Defaults to ``False``.
        path (list, optional): Manually specify the coupling paths.
            If ``None``, all allowed paths are used. Defaults to ``None``.

    Attributes:
        weight (torch.nn.Parameter or None): The learnable weights of the module if ``internal_weights=True``.
            Shape depends on ``feature_mode``.
        info_forward (SparseProductInfo): Constant information for the forward pass computation.
        info_backward1 (SparseProductInfo): Constant information for the first backward pass.
        info_backward2 (SparseProductInfo): Constant information for the second backward pass.
        num_paths (int): Number of coupling paths determined by the irreps.
        weight_numel (int): Total number of elements in the weight tensor.
    """

    def __init__(self,
                 irreps_in, irreps_out,
                 channels_in=None, 
                 channels_out=None,
                 internal_weights=True,
                 path_norm=True,
                 channel_norm=False, 
                 path=None):

        super().__init__()

        self.irreps_in = check_irreps(irreps_in)
        self.irreps_out = check_irreps(irreps_out)
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.internal_weights = internal_weights
        self.path_norm = path_norm
        self.channel_norm = channel_norm

        assert not internal_weights or (
            channels_in is not None and
            channels_out is not None
        )

        (self.info_forward, 
         self.info_backward1,
         self.info_backward2,
         self.num_weights) = so2_linear_infos(
            self.irreps_out,
            self.irreps_in, path=path, path_norm=path_norm,
            channel_norm=self.channel_norm, channel_scale=1.0
        )

        self.weight_shape = (self.num_weights, self.channels_out)
        self.weight_numel = math.prod(self.weight_shape)

        if internal_weights:
            self.weight = torch.nn.Parameter(torch.empty(*self.weight_shape))
            a = 3 ** 0.5
            torch.nn.init.uniform_(self.weight, -a, a)
        else:
            self.weight = None

    def forward(self, input: torch.Tensor, weight: torch.Tensor | None = None, edge_index: torch.Tensor | None = None):
        assert input.shape[-2] == self.irreps_in.dim
        assert input.shape[-1] == self.channels_in

        if self.internal_weights:
            assert weight is None
            weight = self.weight
        else:
            assert weight is not None
            if weight.numel() > self.weight_numel:
                weight = weight.view(-1, *self.weight_shape)
            else:
                weight = weight.view(*self.weight_shape)

        return sparse_mul(
            input, weight,
            self.info_forward,
            self.info_backward1,
            self.info_backward2
        )

    def _apply(self, *args, **kwargs):
        lin = super()._apply(*args, **kwargs)
        lin.info_forward = self.info_forward._apply(*args, **kwargs)
        lin.info_backward1 = self.info_backward1._apply(*args, **kwargs)
        lin.info_backward2 = self.info_backward2._apply(*args, **kwargs)
        return lin
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"irreps_in={self.irreps_in.short_repr()}, "
            f"irreps_out={self.irreps_out.short_repr()}, "
            f"channels={self.channels_in}, "
            f"path_norm={self.path_norm}, "
            f"channel_norm={self.channel_norm}, "
            f"internal_weights={self.internal_weights}, "
            f"num_weights={self.num_weights}"
            ")"
        )


