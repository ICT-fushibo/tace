import math
from typing import Optional


import torch


from ..irreps import check_irreps
from ..structs import so2_linear_infos
from .sparse_product import sparse_mul, sparse_vecmat


def initialize_so3_so2_linear(weight, feature_mode, gain=1, channel_normed=False):
    r"""Initialize weights for SO(3) or SO(2) linear operations.
    
    This function initializes weights for SO(3) or SO(2) linear operations with different
    feature modes. The initialization uses a uniform distribution with bounds
    calculated based on the feature mode and whether channel normalization is used.
    
    For 'uv' mode:
    
    .. math::
        a = \begin{cases}
        \sqrt{3} \cdot \text{gain}, & \text{if channel_normed} = \text{True} \\
        \sqrt{\frac{3}{\text{fan_in}}} \cdot \text{gain}, & \text{otherwise}
        \end{cases}
    
    where :math:`\text{fan_in} = \text{weight.shape[-2]}`
    
    For 'uu' mode:
    
    .. math::
        a = \sqrt{3} \cdot \text{gain}
    
    Args:
        weight (torch.Tensor): The weight tensor to initialize.
        feature_mode (str): The feature mode for initialization. Must be one of ['uv', 'uu'].
        gain (float, optional): The gain factor to apply. Default is 1.
        channel_normed (bool, optional): Whether channel normalization is used. Default is False.
    
    Raises:
        ValueError: If an unknown feature_mode is provided.
    """

    if feature_mode == 'uv':
        fan_out = weight.shape[-1]
        fan_in = weight.shape[-2]
        # Skip fan_in normalization if channel_normed is True
        a = (3 ** 0.5 * gain) if channel_normed else (3.0 / fan_in) ** 0.5 * gain
    elif feature_mode == 'uu':
        # 'uu' mode doesn't have channel normalization based on fan_in
        a = 3 ** 0.5 * gain
    else:
        raise ValueError(f"Unknown feature_mode '{feature_mode}' for initialize_so3_so2_linear")

    torch.nn.init.uniform_(weight, -a, a)


class SO2TensorProduct(torch.nn.Module):
    r"""
    SO(2) equivariant linear layer using tensor products.

    This layer applies an SO(2) equivariant linear transformation, as proposed in `Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs <https://arxiv.org/abs/2302.03655>`_.
    It supports two main modes controlled by ``feature_mode``:

    - ``'uv'``: Fully connected linear layer.

      - Input shape: ``(..., irreps_in.dim, channels_in)``
      - Weight shape: ``(num_weights, channels_in, channels_out)``
      - Output shape: ``(..., irreps_out.dim, channels_out)``

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
                 feature_mode='uu',
                 path_norm=True,
                 channel_norm=False, 
                 path=None):

        super().__init__()

        self.irreps_in = check_irreps(irreps_in)
        self.irreps_out = check_irreps(irreps_out)
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.internal_weights = internal_weights
        self.feature_mode = feature_mode
        self.path_norm = path_norm
        self.channel_norm = channel_norm

        assert not internal_weights or (
            channels_in is not None and
            channels_out is not None
        )
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.feature_mode = feature_mode
        self.path_norm = path_norm
        self.channel_norm = channel_norm and (self.feature_mode == 'uv') # Apply only for uv

        fan_in = self.channels_in if self.feature_mode == 'uv' else 1
        fan_out = self.channels_out if self.feature_mode == 'uv' else 1

        channel_scale = fan_in**(-0.5) if self.channel_norm and self.feature_mode =='uv' else 1.0

        (self.info_forward, 
         self.info_backward1,
         self.info_backward2,
         self.num_weights) = so2_linear_infos(
            self.irreps_out,
            self.irreps_in, path=path, path_norm=path_norm,
            channel_norm=self.channel_norm, channel_scale=channel_scale
        )

        if self.feature_mode == 'uv':
            self.weight_shape = (self.num_weights, self.channels_in, self.channels_out)
        elif self.feature_mode == 'uu':
            self.weight_shape = (self.num_weights, self.channels_out)
        else:
            raise ValueError(f'Feature_mode should be in ["uv", "uu"], got {feature_mode}.')

        self.weight_numel = math.prod(self.weight_shape)

        if internal_weights:
            self.weight = torch.nn.Parameter(torch.empty(*self.weight_shape))
            # Pass channel_norm status to initializer
            initialize_so3_so2_linear(self.weight, self.feature_mode, channel_normed=self.channel_norm)
        else:
            self.weight = None

    def forward(self, input: torch.Tensor, weight:Optional[torch.Tensor] = None):
        # Shape checks
        assert input.shape[-2] == self.irreps_in.dim, f"Input spherical dim mismatch: expected {self.irreps_in.dim}, got {input.shape[-2]}"
        assert input.shape[-1] == self.channels_in, f"Input channel dim mismatch: expected {self.channels_in}, got {input.shape[-1]}"
        if self.internal_weights:
            assert weight is None, 'Do not pass the weight when self.internal_weights is True.'
            weight = self.weight
        else:
            assert weight is not None, 'Please pass the weight when self.internal_weights is False.'
            if weight.numel() > self.weight_numel:
                weight = weight.view(-1, *self.weight_shape)
            else:
                weight = weight.view(*self.weight_shape)

        if self.feature_mode == 'uu':
            output = sparse_mul(
                input, weight,
                self.info_forward, self.info_backward1, self.info_backward2
            )
        elif self.feature_mode == 'uv':
            output = sparse_vecmat(
                input, weight,
                self.info_forward, self.info_backward1, self.info_backward2
            )

        return output

    def _apply(self, *args, **kwargs):
        lin = super()._apply(*args, **kwargs)
        lin.info_forward = self.info_forward._apply(*args, **kwargs)
        lin.info_backward1 = self.info_backward1._apply(*args, **kwargs)
        lin.info_backward2 = self.info_backward2._apply(*args, **kwargs)
        return lin
    
    def __repr__(self):
        channels_repr = f"channels_in={self.channels_in}, channels_out={self.channels_out}"
        if self.feature_mode == 'uu':
             channels_repr = f"channels={self.channels_in}"
        return (f"{self.__class__.__name__}("
                f"irreps_in={self.irreps_in.short_repr()}, "
                f"irreps_out={self.irreps_out.short_repr()}, "
                f"{channels_repr}, "
                f"feature_mode={self.feature_mode}, "
                f"path_norm={self.path_norm}, "
                f"channel_norm={self.channel_norm}, "
                f"internal_weights={self.internal_weights}, "
                f"num_weights={self.num_weights}"
                ")")
