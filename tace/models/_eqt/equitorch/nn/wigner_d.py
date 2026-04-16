from typing import Optional


import torch
from torch import Tensor
import torch.nn as nn

from ..irreps import check_irreps, Irreps
from ..structs import WignerRotationInfo, wigner_d_info
from .sparse_product import sparse_vecsca
from .sparse_scale import sparse_scale


def xyz_to_spherical(
    xyz: Tensor,
    normalize_input: bool = True,
    with_r: bool = False,
    eps: float = 1e-14,
    dim: int = -1
) -> tuple[Tensor, ...] | Tensor:
    """
    Computes spherical coordinates (r, theta, phi) from Cartesian coordinates (x, y, z).

    Args:
        xyz (Tensor): Input tensor with Cartesian coordinates. Assumed to have shape (..., 3).
                      The last dimension (specified by `dim`) should contain x, y, z.
        normalize_input (bool, optional): If True, normalizes the input xyz vector before
                                     computing angles, effectively setting r=1 for angle calculation.
                                     The returned r will still be the original magnitude unless
                                     with_r is False and normalized is True. Defaults to True.
        with_r (bool, optional): If True, returns r along with theta and phi.
                                 If False, returns only theta and phi. Defaults to False.
        eps (float, optional): Small epsilon value to avoid division by zero or acos/atan2
                               instabilities. Defaults to 1e-14.
        dim (int, optional): The dimension along which x, y, z are stored. Defaults to -1.

    Returns:
        tuple[Tensor, ...] | Tensor:
            If with_r is True: (theta, phi, r)
            If with_r is False: (theta, phi)
            - theta: Polar angle (inclination) from z-axis. Range [0, pi]. Shape (...)
            - phi: Azimuthal angle in x-y plane from x-axis. Range [-pi, pi]. Shape (...)
            - r: Radius. Shape (...)
    """
    if xyz.shape[dim] != 3:
        raise ValueError(
            f"Input `xyz` is expected to have 3 components along dimension `dim`={dim}, "
            f"but got {xyz.shape[dim]}."
        )

    # Extract x, y, z components
    x, y, z = torch.unbind(xyz, dim=dim)
    
    # Compute radius if needed
    r = torch.sqrt(xyz.pow(2).sum(dim=-1).clamp_min(eps))
    
    if normalize_input:
        # Normalize coordinates for angle calculations
        x_norm = x / r
        y_norm = y / r
        z_norm = z / r
    else:
        x_norm, y_norm, z_norm = x, y, z
    
    # Compute angles
    theta = torch.acos(torch.clamp(z_norm, -1.0 + eps, 1.0 - eps))
    phi = torch.atan2(y_norm, x_norm + eps)
    
    if with_r:
        return theta, phi, r
    return theta, phi


def sincos(angle: Tensor, max_m: int, with_ones=True, component_normalize=False):
    r"""Prepares the sin/cos tensor for z-rotation.

    This is the functional version of the :class:`~equitorch.nn.angular.SinCos` module.
    See :class:`~equitorch.nn.angular.SinCos` for more details.

    Args:
        angle (torch.Tensor): Input angles.
        max_m (int): The maximum multiple of the angle to compute.
        with_ones (bool, optional): Whether to include the leading 1.0. Defaults to ``True``.
        component_normalize (bool, optional): If ``True``, normalizes sin/cos components by :math:`\sqrt{2}`.
                                         Defaults to ``False``.

    Returns:
        torch.Tensor: The computed sin/cos tensor.
    """
    if max_m == 0:
        # Only scalar irreps, rotation is identity
        return torch.ones_like(angle).unsqueeze(-1)
    m = torch.arange(1, max_m+1, dtype=angle.dtype, device=angle.device)
    m_angle = angle.unsqueeze(-1) * m
    sin_m = torch.sin(m_angle) # sin(|m|*angle)
    cos_m = torch.cos(m_angle) # cos(|m|*angle)
    if component_normalize:
        sin_m = sin_m * (2**0.5)
        cos_m = cos_m * (2**0.5)
    if with_ones:
        ones = torch.ones_like(angle).unsqueeze(-1)
        # [1.0, sin(1a), cos(1a), sin(2a), cos(2a), ...]
        return torch.cat([ones, torch.stack([sin_m, cos_m], dim=-1).flatten(-2, -1)], dim=-1)
    else:
        return torch.stack([sin_m, cos_m], dim=-1).flatten(-2, -1)


def sparse_wigner_rotation(
    input: Tensor,
    sincos_alpha: Optional[Tensor],
    sincos_beta: Optional[Tensor],
    sincos_gamma: Optional[Tensor],
    info: WignerRotationInfo
) -> Tensor:
    """Applies sparse Wigner D-matrix rotation :math:`D(\alpha, \beta, \gamma) = D_z(\alpha)D_y(\beta)D_z(\gamma)`.

    Functional version of :class:`~equitorch.nn.wigner_d.SparseWignerRotation`.
    See the class for more details.

    Args:
        input (torch.Tensor): Input features.
        sincos_alpha (Optional[torch.Tensor]): Precomputed sin/cos of Euler angle :math:`\alpha`.
        sincos_beta (Optional[torch.Tensor]): Precomputed sin/cos of Euler angle :math:`\beta`.
        sincos_gamma (Optional[torch.Tensor]): Precomputed sin/cos of Euler angle :math:`\gamma`.
        info (WignerRotationInfo): Precomputed sparse rotation information.

    Returns:
        torch.Tensor: Rotated features.
    """

    x = input
    if sincos_gamma is not None:
        x = sparse_vecsca(x, sincos_gamma, 
                           info.rotate_z_info_fwd, 
                           info.rotate_z_info_bwd_input,
                           info.rotate_z_info_bwd_cs)
    if sincos_beta is not None:
        x = sparse_scale(x, info.j_matrix_info, info.j_matrix_info)
        x = sparse_vecsca(x, sincos_beta,
                        info.rotate_z_info_fwd, 
                        info.rotate_z_info_bwd_input,
                        info.rotate_z_info_bwd_cs)
        x = sparse_scale(x, info.j_matrix_info, info.j_matrix_info)
    if sincos_alpha is not None:
        x = sparse_vecsca(x, sincos_alpha,
                          info.rotate_z_info_fwd, 
                          info.rotate_z_info_bwd_input,
                          info.rotate_z_info_bwd_cs)

    return x


def dense_wigner_rotation(input: Tensor, wigner_d: Tensor):
    r"""Applies a precomputed dense Wigner D-matrix to input features.

    Functional version of :class:`~equitorch.nn.wigner_d.DenseWignerRotation`.
    See the class for more details.

    Args:
        input (torch.Tensor): Input features of shape ``(batch_size, irreps.dim, channels)``.
        wigner_d (torch.Tensor): Dense Wigner D-matrix of shape ``(batch_size, irreps.dim, irreps.dim)``.

    Returns:
        torch.Tensor: Rotated features.
    """
    return wigner_d @ input


def wigner_d_matrix(
    eye: Tensor, # Input identity matrix (dim, dim)
    alpha: Optional[Tensor] = None,
    beta: Optional[Tensor] = None,
    gamma: Optional[Tensor] = None,
    sincos_alpha: Optional[Tensor] = None,
    sincos_beta: Optional[Tensor] = None,
    sincos_gamma: Optional[Tensor] = None,
    info: Optional[WignerRotationInfo] = None
) -> Tensor:
    r"""
    Computes the dense Wigner D-matrix by applying sparse rotation operations
    to a provided identity matrix.

    Args:
        eye (Tensor): Identity matrix of shape (dim, dim). Device and dtype are inferred.
        alpha (Optional[Tensor]): Alpha Euler angle.
        beta (Optional[Tensor]): Beta Euler angle.
        gamma (Optional[Tensor]): Gamma Euler angle.
        sincos_alpha (Optional[Tensor]): Precomputed sin/cos for alpha angle.
        sincos_beta (Optional[Tensor]): Precomputed sin/cos for beta angle.
        sincos_gamma (Optional[Tensor]): Precomputed sin/cos for gamma angle.
        info (WignerRotationInfo): Precomputed sparse rotation info.

    Returns:
        Tensor: The dense Wigner D-matrix of shape (batch, dim, dim)
                or (dim, dim) if angles are not batched.
    """
    assert info is not None
    # Precompute sincos if angles are given
    if sincos_alpha is None and alpha is not None:
        sincos_alpha = sincos(alpha, info.max_m)
    if sincos_beta is None and beta is not None:
        sincos_beta = sincos(beta, info.max_m)
    if sincos_gamma is None and gamma is not None:
        sincos_gamma = sincos(gamma, info.max_m)

    dim = eye.shape[0]
    if dim == 0:
        # Handle empty irreps case based on eye matrix
        batch_size = 1 # Default if no angles
        if sincos_alpha is not None: batch_size = sincos_alpha.shape[0]
        elif sincos_beta is not None: batch_size = sincos_beta.shape[0]
        elif sincos_gamma is not None: batch_size = sincos_gamma.shape[0]

        return torch.empty((batch_size, 0, 0), device=eye.device, dtype=eye.dtype)

    # Reshape eye to (N, M, C) where N=1, M=dim, C=dim
    # The rotation acts on the M dimension, treating C as channels.

    # Apply sparse rotation
    # Assume info is already on the correct device (handled by the Module wrapper)
    rotated_eye = sparse_wigner_rotation(
        eye, sincos_alpha, sincos_beta, sincos_gamma, info
    ) # Output shape (N, dim, dim) where N is batch size from angles or 1

    # Determine if the operation was batched based on angle inputs
    is_batched = (sincos_alpha is not None and sincos_alpha.ndim > 1 and sincos_alpha.shape[0] > 1) or \
                 (sincos_beta is not None and sincos_beta.ndim > 1 and sincos_beta.shape[0] > 1) or \
                 (sincos_gamma is not None and sincos_gamma.ndim > 1 and sincos_gamma.shape[0] > 1)

    # If not batched and output has batch dim 1, squeeze it
    if not is_batched and rotated_eye.shape[0] == 1:
        return rotated_eye.squeeze(0) # Shape (dim, dim)
    else:
        return rotated_eye # Shape (batch, dim, dim)


def align_to_z_wigner_d(
    eye: Tensor,
    xyz: Tensor,
    max_m: int,
    info: WignerRotationInfo,
    normalized: bool = True,
    eps: float = 1e-14
) -> Tensor:
    r"""
    Computes Wigner D-matrix :math:`D(R_{align})` that rotates vector :math:`\vec{v}` to z-axis.

    Functional version of :class:`~equitorch.nn.wigner_d.AlignToZWignerD`.

    See the class for more details.

    The rotation :math:`R_{align}` is :math:`(0, -\theta, -\phi)` where :math:`\theta, \phi` are polar and azimuthal angles of :math:`\vec{v}`.

    Args:
        eye (torch.Tensor): Identity matrix of shape ``(dim, dim)``.
        xyz (torch.Tensor): Input Cartesian coordinates, shape ``(..., 3)``.
        max_m (int): Maximum m value for sincos calculation, derived from irreps.
        info (WignerRotationInfo): Precomputed sparse rotation info for the irreps.
        normalized (bool, optional): Whether ``xyz`` is already normalized. Defaults to ``True``.
        eps (float, optional): Epsilon for numerical stability. Defaults to ``1e-14``.

    Returns:
        torch.Tensor: Dense Wigner D-matrix for alignment, shape ``(..., dim, dim)``.
    """
    # 1. Calculate theta and phi from xyz
    theta, phi = xyz_to_spherical(
        xyz,
        normalize_input=normalized,
        with_r=False, # We don't need r
        eps=eps,
        dim=-1 # Assume xyz is in the last dimension
    ) # Output shapes (..., 1)

    # 2. Squeeze angles and calculate sincos embeddings
    theta_squeezed = theta.squeeze(-1)
    phi_squeezed = phi.squeeze(-1)

    sincos_neg_theta = sincos(-theta_squeezed, max_m)
    sincos_neg_phi = sincos(-phi_squeezed, max_m)

    # Infer batch shape from xyz (excluding the last dim)
    batch_shape = xyz.shape[:-1]
    # Create identity matrix without batch dim first
    # Expand eye matrix to match the batch shape of xyz/angles
    # Example: if xyz is (B, N, 3), batch_shape is (B, N)
    # We need eye to be broadcastable to (B, N, dim, dim) for wigner_d_matrix if angles are batched.
    # However, wigner_d_matrix expects eye (dim, dim) and handles batching internally based on angles.

    # 4. Compute Wigner D matrix D(alpha=phi, beta=theta, gamma=0)
    # gamma=0 means sincos_gamma=None
    wigner_d = wigner_d_matrix(
        eye=eye,
        sincos_alpha=None, # Gamma rotation is zero
        sincos_beta=sincos_neg_theta,
        sincos_gamma=sincos_neg_phi,
        info=info
    )

    return wigner_d


class SparseWignerRotation(nn.Module):
    r"""
    Applies a sparse Wigner D-matrix rotation to input features.

    This module computes the rotation based on Euler angles (:math:`\alpha, \beta, \gamma`)
    provided as precomputed sin/cos tensors. It utilizes sparse matrix operations for the rotation.

    .. warning::
        It is currently suggested to use :class:`DenseWignerRotation` or :class:`WignerD` 
        for applying rotations, at least when gradients with respect to angles are not required.

    The Wigner D-matrix :math:`D^l_{m'm}(R)` transforms spherical tensors under rotation :math:`R`.
    This module applies such transformations for a given :class:`~equitorch.irreps.Irreps`.

    Args:
        irreps (Irreps): The irreducible representations defining the input and output feature space.
    """

    info: WignerRotationInfo

    def __init__(self, irreps: Irreps):
        super().__init__()

        self.irreps = check_irreps(irreps)
        self.info = wigner_d_info(self.irreps)
        self.max_m = max(ir.l for ir in irreps)

    def forward(self, input: Tensor, 
                sincos_alpha: Optional[Tensor], 
                sincos_beta: Optional[Tensor],
                sincos_gamma: Optional[Tensor]) -> Tensor:
        r"""
        Applies the sparse Wigner rotation.

        Args:
            input (torch.Tensor): Input features of shape ``(batch_size, irreps.dim, channels)``.
            sincos_alpha (Optional[torch.Tensor]): Precomputed :math:`\sin` and :math:`\cos` of Euler angle :math:`\alpha`.
            sincos_beta (Optional[torch.Tensor]): Precomputed :math:`\sin` and :math:`\cos` of Euler angle :math:`\beta`.
            sincos_gamma (Optional[torch.Tensor]): Precomputed :math:`\sin` and :math:`\cos` of Euler angle :math:`\gamma`.

        Returns:
            torch.Tensor: Rotated features of shape ``(batch_size, irreps.dim, channels)``.
        """
        return sparse_wigner_rotation(input, sincos_alpha,
                        sincos_beta, sincos_gamma,
                        self.info)
    
    
    def _apply(self, *args, **kwargs):
        # Ensure info objects are moved to the correct device/dtype
        wig = super()._apply(*args, **kwargs)
        # Apply to the WignerDRotationInfo NamedTuple fields
        wig.info = wig.info._apply(*args, **kwargs)
        return wig

    def extra_repr(self) -> str:
        return f'irreps={self.irreps}, max_m={self.max_m}'


class DenseWignerRotation(nn.Module):
    r"""
    Applies a dense Wigner D-matrix rotation to input features.

    This module takes a precomputed dense Wigner D-matrix and applies it to the input features.
    The Wigner D-matrix :math:`D(R)` itself should be computed separately, for example, using the :class:`WignerD` module.

    Args:
        irreps (Irreps): The irreducible representations defining the input and output feature space.
                         This is used for validation and representation purposes.
    """
    def __init__(self, irreps: Irreps):
        super().__init__()

        self.irreps = check_irreps(irreps)

    def forward(self, input: Tensor, wigner_d: Tensor) -> Tensor:
        r"""
        Applies the dense Wigner rotation.

        The operation performed is effectively a batched matrix multiplication:
        `output = wigner_d @ input`

        Args:
            input (Tensor): Input features of shape ``(batch_size, irreps.dim, channels)``.
            wigner_d (Tensor): Dense Wigner D-matrix of shape ``(batch_size, irreps.dim, irreps.dim)``.

        Returns:
            Tensor: Rotated features of shape ``(batch_size, irreps.dim, channels)``.
        """
        # Using einsum for clarity on dimensions, equivalent to batched matmul
        # D_bij, F_bjk -> R_bik (b=batch, i=output_dim, j=input_dim, k=channels)
        # For Wigner D, input_dim and output_dim are the same (self.irreps.dim)
        return wigner_d @ input # [N, dim, dim] @ [N, dim, channels] -> [N, dim, channels]

    
    def extra_repr(self) -> str:
        return f'irreps={self.irreps}'


class WignerD(nn.Module):
    r"""
    Computes the dense Wigner D-matrix :math:`D(R)` for given :class:`~equitorch.irreps.Irreps` and Euler angles :math:`(\alpha, \beta, \gamma)`.

    The Wigner D-matrix is constructed based on the ZYZ Euler angle convention:

    .. math::

        D(\alpha, \beta, \gamma) = D_z(\alpha) D_y(\beta) D_z(\gamma)

    This module caches the necessary sparse rotation information and an identity matrix
    to efficiently compute the dense D-matrix using the :func:`~equitorch.nn.functional.wigner_d.wigner_d_matrix` functional.

    Args:
        irreps (Irreps): The irreducible representations for which to compute the D-matrix.
                         The resulting D-matrix will have dimensions ``(irreps.dim, irreps.dim)``.
    """
    info: WignerRotationInfo

    def __init__(self, irreps: Irreps):
        super().__init__()
        self.irreps = check_irreps(irreps)
        self.max_m = max((ir.l for ir in self.irreps), default=0)

        # Prepare and store sparse rotation info
        self.info = wigner_d_info(self.irreps)

        # Prepare and register identity matrix as a buffer
        dim = self.irreps.dim
        self.register_buffer('identity', torch.eye(dim))

    def forward(self,
                alpha: Optional[Tensor] = None,
                beta: Optional[Tensor] = None,
                gamma: Optional[Tensor] = None,
                sincos_alpha: Optional[Tensor] = None,
                sincos_beta: Optional[Tensor] = None,
                sincos_gamma: Optional[Tensor] = None) -> Tensor:
        r"""
        Computes the Wigner D-matrix.

        Provide either the angles (alpha, beta, gamma) or the precomputed
        sin/cos tensors (sincos_alpha, sincos_beta, sincos_gamma).

        Args:
            alpha (Optional[torch.Tensor]): Euler angle alpha.
            beta (Optional[torch.Tensor]): Euler angle beta.
            gamma (Optional[torch.Tensor]): Euler angle gamma.
            sincos_alpha (Optional[torch.Tensor]): Precomputed sin/cos for alpha.
            sincos_beta (Optional[torch.Tensor]): Precomputed sin/cos for beta.
            sincos_gamma (Optional[torch.Tensor]): Precomputed :math:`\sin` and :math:`\cos` of Euler angle :math:`\gamma`.

        Returns:
            torch.Tensor: The dense Wigner D-matrix of shape ``(batch_size, irreps.dim, irreps.dim)``
                          if batch size is 1 and input angles are unbatched.
        """
        return wigner_d_matrix(
            self.identity, # Use cached identity matrix
            alpha,
            beta,
            gamma,
            sincos_alpha,
            sincos_beta,
            sincos_gamma,
            self.info # Use cached info
        )

    def _apply(self, *args, **kwargs):
        # Ensure info objects are moved to the correct device/dtype
        wig = super()._apply(*args, **kwargs)
        # Apply to the WignerDRotationInfo NamedTuple fields
        wig.info = wig.info._apply(*args, **kwargs)
        # Identity matrix buffer is handled automatically by PyTorch
        return wig

    def extra_repr(self) -> str:
        return f'irreps={self.irreps}'


class AlignToZWignerD(nn.Module):
    r"""
    Computes the Wigner D-matrix :math:`D(R_{align})` that rotates a given vector :math:`\vec{v} = (x, y, z)` onto the z-axis.

    The rotation :math:`R_{align}` is defined by Euler angles :math:`(0, -\theta, -\phi)`, where :math:`\theta` and :math:`\phi` are the
    polar and azimuthal angles of the vector :math:`\vec{v}`, respectively. This means:

    .. math::

        R_{align} \vec{v} = ||\vec{v}|| \hat{z}

    The Wigner D-matrix is then :math:`D(0, -\theta, -\phi)`.

    This module caches the necessary sparse rotation information and an identity matrix.
    It utilizes the :func:`~equitorch.nn.functional.wigner_d.align_to_z_wigner_d` functional.

    Args:
        irreps (Irreps): The irreducible representations for which to compute the D-matrix.
        normalized (bool, optional): Whether to normalize the input ``xyz`` vector
            before calculating angles for rotation. If ``True``, effectively rotates :math:`\hat{v}`.
            Defaults to ``True``.
        eps (float, optional): Small :math:`\epsilon` value for numerical stability in angle calculation.
            Defaults to ``1e-14``.
    """
    info: WignerRotationInfo

    def __init__(self,
                 irreps: Irreps,
                 normalized: bool = True,
                 eps: float = 1e-14):
        super().__init__()
        self.irreps = check_irreps(irreps)
        self.normalized = normalized
        self.eps = eps
        self.max_m = max((ir.l for ir in self.irreps), default=0)

        # Prepare and store sparse rotation info
        self.info = wigner_d_info(self.irreps)

        # Prepare and register identity matrix as a buffer
        dim = self.irreps.dim
        self.register_buffer('identity', torch.eye(dim))

    def forward(self, xyz: Tensor) -> Tensor:
        r"""
        Computes the alignment Wigner D-matrix.

        Args:
            xyz (Tensor): Input Cartesian coordinates, shape (..., 3).

        Returns:
            Tensor: The dense Wigner D-matrix for alignment.
                    Shape (..., irreps.dim, irreps.dim).
        """
        # Import functional align_to_z_wigner_d here to avoid circular dependency at top level
        # if functional layer imports from nn layer (though unlikely)
        
        return align_to_z_wigner_d(
            eye=self.identity,
            xyz=xyz,
            max_m=self.max_m,
            info=self.info,
            normalized=self.normalized,
            eps=self.eps
        )

    def _apply(self, *args, **kwargs):
        # Ensure info objects are moved to the correct device/dtype
        wig = super()._apply(*args, **kwargs)
        # Apply to the WignerDRotationInfo NamedTuple fields
        wig.info = wig.info._apply(*args, **kwargs)
        # Identity matrix buffer is handled automatically by PyTorch
        return wig

    def extra_repr(self) -> str:
        return (f'irreps={self.irreps}, normalized={self.normalized}, '
                f'eps={self.eps}')
