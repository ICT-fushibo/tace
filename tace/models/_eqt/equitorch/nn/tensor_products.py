import math
import torch
from torch import Tensor
from torch.autograd import Function

from ..irreps import check_irreps
from ..structs import TensorProductInfo, tp_infos
from .sparse_product import sparse_mat_t_vec, sparse_mul, sparse_outer, sparse_vecmat


class TensorProductUUUDummy(Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo,
                tp_info_backward2: TensorProductInfo):
        
        inter = sparse_mul(input1, input2, tp_info_forward.info_Mij_fwd)
        ret = sparse_mul(inter, weight, tp_info_forward.info_M_fwd)

        grad_weight = weight.requires_grad
        grad_input1 = input1.requires_grad
        grad_input2 = input2.requires_grad
        
        ctx.inter_shape = inter.shape
        ctx.inter_dtype = inter.dtype
        ctx.inter_device = inter.device
        ctx.weight_ndim = weight.ndim
        if not grad_weight:
            inter = None
        if (not grad_input1) and (not grad_input2):
            weight = None
        if not grad_input1:
            input2 = None
        if not grad_input2:
            input1 = None
        ctx.save_for_backward(input1, input2, weight, inter)
        ctx.tp_info = (tp_info_forward, tp_info_backward1, tp_info_backward2)
        return ret, inter

    @staticmethod
    def backward(ctx, grad_output, grad_inter):
        grad = grad_output
        input1, input2, weight, inter = ctx.saved_tensors
        tp_info_forward, tp_info_backward1, tp_info_backward2 = ctx.tp_info

        # Ensure grad_inter has correct shape for higher-order derivatives
        if grad_inter is not None:
            grad_inter = torch.broadcast_to(grad_inter, ctx.inter_shape)
        else:
            grad_inter = torch.zeros(ctx.inter_shape, dtype=ctx.inter_dtype, device=ctx.inter_device)
        if ctx.needs_input_grad[0]:
            grad1 = tensor_product_uuu(
                input2, grad, weight, 
                tp_info_backward1, tp_info_backward2, tp_info_forward)
            # if grad1.requires_grad:
            grad1 = grad1 + sparse_mul(input2, grad_inter,
                                    tp_info_forward.info_Mij_bwd1,
                                    tp_info_forward.info_Mij_bwd2,
                                    tp_info_forward.info_Mij_fwd)
        else:
            grad1 = None
            
        if ctx.needs_input_grad[1]:
            grad2 = tensor_product_uuu(
                grad, input1, weight, 
                tp_info_backward2, tp_info_forward, tp_info_backward1)
            # if grad2.requires_grad:
            grad2 = grad2 + sparse_mul(grad_inter, input1,
                                    tp_info_forward.info_Mij_bwd2,
                                    tp_info_forward.info_Mij_fwd,
                                    tp_info_forward.info_Mij_bwd1)
        else:
            grad2 = None

        if ctx.needs_input_grad[2]:
            grad_W = sparse_mul(grad_output, inter,
                                tp_info_forward.info_M_bwd2,
                                tp_info_forward.info_M_fwd,
                                tp_info_forward.info_M_bwd1,
                                out_accumulated=ctx.weight_ndim == 2)
        else:
            grad_W = None
        return grad1, grad2, grad_W, None, None, None


def tensor_product_uuu(input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo = None,
                tp_info_backward2: TensorProductInfo = None):
    ret, _ =  TensorProductUUUDummy.apply(input1, input2, weight,
                                       tp_info_forward,
                                       tp_info_backward1,
                                       tp_info_backward2)
    return ret


class TensorProductUVWDummy(Function):

    @staticmethod
    def forward(ctx, input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo,
                tp_info_backward2: TensorProductInfo):
        
        inter = sparse_outer(input1, input2, tp_info_forward.info_Mij_fwd)
        ret = sparse_vecmat(inter.flatten(-2), weight.flatten(-3,-2), tp_info_forward.info_M_fwd)

        grad_weight = weight.requires_grad
        grad_input1 = input1.requires_grad
        grad_input2 = input2.requires_grad

        ctx.inter_shape = inter.shape
        ctx.inter_dtype = inter.dtype
        ctx.inter_device = inter.device  
        ctx.weight_ndim = weight.ndim
        if not grad_weight:
            inter = None
        if (not grad_input1) and (not grad_input2):
            weight = None
        if not grad_input1:
            input2 = None
        if not grad_input2:
            input1 = None
        ctx.save_for_backward(input1, input2, weight, inter)
        ctx.tp_info = (tp_info_forward, tp_info_backward1, tp_info_backward2)
        return ret, inter

    @staticmethod
    def backward(ctx, grad_output, grad_inter):
        grad = grad_output
        input1, input2, weight, inter = ctx.saved_tensors
        tp_info_forward, tp_info_backward1, tp_info_backward2 = ctx.tp_info

        grad_inter = torch.broadcast_to(grad_inter, ctx.inter_shape)
        if ctx.needs_input_grad[0]:
            grad1 = tensor_product_uvw(
                input2, grad, 
                weight.permute(*range(0,weight.ndim-3), -2, -1, -3).contiguous(),
                tp_info_backward1, tp_info_backward2, tp_info_forward)
            # if grad1.requires_grad:
            grad1 = grad1 + sparse_vecmat(input2, grad_inter.transpose(-1,-2).contiguous(),
                                    tp_info_forward.info_Mij_bwd1,
                                    tp_info_forward.info_Mij_bwd2,
                                    tp_info_forward.info_Mij_fwd)
        else:
            grad1 = None
            
        if ctx.needs_input_grad[1]:
            grad2 = tensor_product_uvw(
                grad, input1,
                weight.permute(*range(0,weight.ndim-3), -1, -3, -2).contiguous(),
                tp_info_backward2, tp_info_forward, tp_info_backward1)
            # if grad2.requires_grad:
            grad2 = grad2 + sparse_mat_t_vec(grad_inter, input1,
                                    tp_info_forward.info_Mij_bwd2,
                                    tp_info_forward.info_Mij_fwd,
                                    tp_info_forward.info_Mij_bwd1)
        else:
            grad2 = None

        if ctx.needs_input_grad[2]:
            grad_W = sparse_outer(grad_output, inter.flatten(-2),
                                tp_info_forward.info_M_bwd2,
                                tp_info_forward.info_M_fwd,
                                tp_info_forward.info_M_bwd1,
                                out_accumulated=ctx.weight_ndim == 4).transpose(-1,-2).contiguous()
            grad_W = grad_W.unflatten(-2, weight.shape[-3:-1])
        else:
            grad_W = None
        return grad1, grad2, grad_W, None, None, None


def tensor_product_uvw(input1: Tensor, input2: Tensor, weight: Tensor,
                tp_info_forward: TensorProductInfo,
                tp_info_backward1: TensorProductInfo = None,
                tp_info_backward2: TensorProductInfo = None):
    ret, _ = TensorProductUVWDummy.apply(input1, input2, weight,
                                       tp_info_forward,
                                       tp_info_backward1,
                                       tp_info_backward2)
    return ret


def initialize_tensor_product(weight, feature_mode, gain=1, channel_normed=False):
    r"""Initialize weights for tensor product operations.
    
    This function initializes weights for tensor product operations with different
    feature modes. The initialization uses a uniform distribution with bounds
    calculated based on the feature mode and whether channel normalization is used.
    
    For 'uvw' mode:
    
    .. math::
        a = \begin{cases}
        \sqrt{3} \cdot \text{gain}, & \text{if channel_normed} = \text{True} \\
        \sqrt{\frac{3}{\text{fan_in}}} \cdot \text{gain}, & \text{otherwise}
        \end{cases}
    
    where :math:`\text{fan_in} = \text{weight.shape[-2]} \cdot \text{weight.shape[-3]}`
    
    For 'uuu' mode:
    
    .. math::
        a = \sqrt{3} \cdot \text{gain}
    
    Args:
        weight (torch.Tensor): The weight tensor to initialize.
        feature_mode (str): The feature mode for initialization. Must be one of ['uvw', 'uuu'].
        gain (float, optional): The gain factor to apply. Default is 1.
        channel_normed (bool, optional): Whether channel normalization is used. Default is False.
    
    Raises:
        ValueError: If an unknown feature_mode is provided.
    """

    if feature_mode == 'uvw':
        fan_out = weight.shape[-1]
        fan_in = weight.shape[-2] * weight.shape[-3]
        # Skip fan_in normalization if channel_normed is True
        a = (3 ** 0.5 * gain) if channel_normed else (3.0 / fan_in) ** 0.5 * gain
    elif feature_mode == 'uuu':
        # 'uuu' mode doesn't have channel normalization based on fan_in
        a = 3 ** 0.5 * gain
    else:
        raise ValueError(f"Unknown feature_mode '{feature_mode}' for initialize_tensor_product")

    torch.nn.init.uniform_(weight, -a, a)

 
class TensorProduct(torch.nn.Module):
    r"""Computes the tensor product of two equivariant feature tensors.

    Supports two main modes controlled by ``feature_mode``:

    - ``'uvw'``: Fully connected tensor product.

      - Input1 shape: ``(..., irreps_in1.dim, channels_in1)``
      - Input2 shape: ``(..., irreps_in2.dim, channels_in2)``
      - Weight shape: ``(num_paths, channels_in1, channels_in2, channels_out)``
      - Output shape: ``(..., irreps_out.dim, channels_out)``

    - ``'uuu'``: Depthwise/elementwise tensor product.
      with ``uuu`` instructions (often used for self-interaction).

      - Input1 shape: ``(..., irreps_in1.dim, channels)``
      - Input2 shape: ``(..., irreps_in2.dim, channels)``
      - Weight shape: ``(num_paths, channels_out)`` (where ``channels_out`` usually equals ``channels``)
      - Output shape: ``(..., irreps_out.dim, channels_out)``

    Args:
        irreps_in1 (Irreps): Irreducible representations of the first input tensor.
        irreps_in2 (Irreps): Irreducible representations of the second input tensor.
        irreps_out (Irreps): Irreducible representations of the output tensor.
        channels_in1 (int, optional): Number of channels for the first input.
            Required if ``internal_weights=True`` or ``feature_mode='uvw'``.
        channels_in2 (int, optional): Number of channels for the second input.
            Required if ``internal_weights=True`` or ``feature_mode='uvw'``.
        channels_out (int, optional): Number of channels for the output.
            Required if ``internal_weights=True``.
        internal_weights (bool, default=True): If ``True``, the module manages its own weight parameter.
            If ``False``, weights must be provided during the forward pass.
        feature_mode ({'uuu', 'uvw'}, default='uuu'): Controls the type of tensor product:

            - ``'uuu'``: Depthwise/elementwise product. Assumes ``channels_in1 == channels_in2 == channels_out``.
            - ``'uvw'``: Fully connected product.
        path_norm (bool, default=True): Whether to apply path normalization to the weights.
        channel_norm (bool, default=False): Whether to apply channel normalization (specific to ``'uvw'`` mode).
            Divides weights by :math:`\sqrt{\text{channels_in1} \times \text{channels_in2}}`.
        path (list, optional): Manually specify the coupling paths.
            If ``None``, all allowed paths are used.

    Attributes:
        weight (torch.nn.Parameter or None): The learnable weights of the module if ``internal_weights=True``.
            Shape depends on ``feature_mode``.
        tp_info_forward (TensorProductInfo): Constant information for the forward pass computation.
        tp_info_backward1 (TensorProductInfo): Constant information for the backward pass w.r.t. input1.
        tp_info_backward2 (TensorProductInfo): Constant information for the backward pass w.r.t. input2.
        num_paths (int): Number of coupling paths determined by the irreps.
        weight_numel (int): Total number of elements in the weight tensor.
    """
    tp_info_forward: TensorProductInfo
    tp_info_backward1: TensorProductInfo
    tp_info_backward2: TensorProductInfo

    def __init__(self, 
                 irreps_in1,
                 irreps_in2, 
                 irreps_out, 
                 channels_in1=None,
                 channels_in2=None,
                 channels_out=None,
                 internal_weights=True,
                 feature_mode='uuu',
                 path_norm=True,
                 channel_norm=False, # Add channel_norm parameter
                 trainable: bool = True,
                 path=None):

        super().__init__()

        self.irreps_in1 = check_irreps(irreps_in1)
        self.irreps_in2 = check_irreps(irreps_in2)
        self.irreps_out = check_irreps(irreps_out)
        self.irreps_in1_dim = self.irreps_in1.dim
        self.irreps_in2_dim = self.irreps_in2.dim
        self.irreps_out_dim = self.irreps_out.dim
        self.trainable = trainable

        assert not internal_weights or (
            channels_in1 is not None and
            channels_in2 is not None and
            channels_out is not None
        )
        self.channels_in1 = channels_in1
        self.channels_in2 = channels_in2
        self.channels_out = channels_out
        self.feature_mode = feature_mode
        self.path_norm = path_norm
        self.channel_norm = channel_norm and (self.feature_mode == 'uvw') # Apply only for uvw

        # Calculate fan_in for channel normalization
        fan_in = self.channels_in1 * self.channels_in2 if self.feature_mode == 'uvw' else 1

        (self.tp_info_forward, 
          self.tp_info_backward1,
          self.tp_info_backward2,
          self.num_paths) = tp_infos(
              self.irreps_out,
              self.irreps_in1,
              self.irreps_in2, path=path, path_norm=path_norm,
              channel_norm=self.channel_norm, channel_scale=fan_in**(-0.5) # Pass channel_norm and fan_in
          )

        if self.feature_mode == 'uvw':
            self.weight_shape = (self.num_paths, self.channels_in1, self.channels_in2, self.channels_out)
        elif self.feature_mode == 'uuu':
            self.weight_shape = (self.num_paths, self.channels_out)
        else:
            raise ValueError(f'Feature_mode should be in ["uvw", "uuu"], got {feature_mode}.')
        self.weight_numel = math.prod(self.weight_shape)

        self.internal_weights = internal_weights
        if internal_weights:
            if self.trainable:
                self.weight = torch.nn.Parameter(torch.empty(*self.weight_shape))
                # Pass channel_norm status to initializer
                initialize_tensor_product(self.weight, self.feature_mode, channel_normed=self.channel_norm)
            else:
                self.register_buffer('weight', torch.ones(*self.weight_shape), persistent=False)
        else:
            self.weight = None

    def forward(self, input1: torch.Tensor, input2: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        # Input shape checks
        assert input1.shape[-2] == self.irreps_in1_dim, f"Input1 spherical dim mismatch: expected {self.irreps_in1_dim}, got {input1.shape[-2]}"
        assert input2.shape[-2] == self.irreps_in2_dim, f"Input2 spherical dim mismatch: expected {self.irreps_in2_dim}, got {input2.shape[-2]}"
        if self.feature_mode == 'uvw':
            assert input1.shape[-1] == self.channels_in1, f"Input1 channel dim mismatch: expected {self.channels_in1}, got {input1.shape[-1]}"
            assert input2.shape[-1] == self.channels_in2, f"Input2 channel dim mismatch: expected {self.channels_in2}, got {input2.shape[-1]}"
        elif self.feature_mode == 'uuu':
            assert input1.shape[-1] == input2.shape[-1] == self.channels_in1, f"Input1 channel dim mismatch: expected {self.channels_in1}, got {input1.shape[-1]}"

        if self.internal_weights:
            assert weight is None, 'Do not pass the weight when self.internal_weights is True.'
            weight = self.weight
        else:
            assert weight is not None, 'Please pass the weight when self.internal_weights is False.'
            if weight.numel() > self.weight_numel:
                weight = weight.view(-1, *self.weight_shape)
            else:
                weight = weight.view(*self.weight_shape)

        # Prepare args for functional call
        args = (input1, input2, weight,
                self.tp_info_forward,
                self.tp_info_backward1,
                self.tp_info_backward2)

        # Call functional implementation
        if self.feature_mode == 'uuu':
            output = tensor_product_uuu(*args)
        elif self.feature_mode == 'uvw':
            output = tensor_product_uvw(*args)
        else:
             # This case is already handled in __init__, but added for safety
             raise ValueError(f"Invalid feature_mode: {self.feature_mode}")
        return output

    def _apply(self, *args, **kwargs):
        tp = super()._apply(*args, **kwargs)
        tp.tp_info_forward = self.tp_info_forward._apply(*args, **kwargs)
        tp.tp_info_backward1 = self.tp_info_backward1._apply(*args, **kwargs)
        tp.tp_info_backward2 = self.tp_info_backward2._apply(*args, **kwargs)
        return tp

    # def __repr__(self):
    #     channels_repr = ""
    #     if self.feature_mode == 'uvw':
    #         channels_repr = f"channels_in1={self.channels_in1}, channels_in2={self.channels_in2}, channels_out={self.channels_out}"
    #     elif self.feature_mode == 'uuu':
    #          # Assuming channels_in1 == channels_in2 == channels_out for uuu mode based on docstring
    #          channels_repr = f"channels={self.channels_in1}" # Use channels_in1 as the representative channel count
        
        # return (f"{self.__class__.__name__}("
        #         f"irreps_in1={self.irreps_in1.short_repr()}, "
        #         f"irreps_in2={self.irreps_in2.short_repr()}, "
        #         f"irreps_out={self.irreps_out.short_repr()}, "
        #         f"{channels_repr}, "
        #         f"feature_mode={self.feature_mode}, "
        #         f"path_norm={self.path_norm}, "
        #         f"channel_norm={self.channel_norm}, "
        #         f"internal_weights={self.internal_weights}, "
        #         f"num_paths={self.num_paths}"
        #         ")")

    def __repr__(self):

        def to_beautiful(irreps, channel):
            return "+".join(f"{channel}x{irrep}" for irrep in irreps.short_repr().split('+'))

        return (
            f"{self.__class__.__name__}("
            f"{to_beautiful(self.irreps_in1.simplified(), self.channels_in1)} x "
            f"{to_beautiful(self.irreps_in2.simplified(), self.channels_in2)} -> "
            f"{to_beautiful(self.irreps_out.simplified(), self.channels_out)})"
        )
