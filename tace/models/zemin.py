################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from pathlib import Path
from typing import Optional, Tuple


import torch


from .ictd import ICTD


CARTNN_CACHE_DIR = Path.home() / ".cache" / "cartnn"


def torch_get_default_tensor_type() -> str:
    return torch.empty(0).type()


def torch_get_default_dtype() -> torch.dtype:
    """A torchscript-compatible version of torch.get_default_dtype()"""
    return torch.empty(0).dtype


def torch_get_default_device() -> torch.device:
    return torch.empty(0).device


def explicit_default_types(
        dtype: Optional[torch.dtype], device: Optional[torch.device]
    ) -> Tuple[torch.dtype, torch.device]:
    """A torchscript-compatible type resolver"""
    if dtype is None:
        dtype = torch_get_default_dtype()
    if device is None:
        device = torch_get_default_device()
    return dtype, device


def _cartesian_3j(l1: int, l2: int, l3: int) -> torch.Tensor:     
    with torch.no_grad():
        P1S, D1S, C1S, S1S = ICTD(l1+l2, l3, decomposition=False)
        P2S, D2S, C2S, S2S = ICTD(l3, l3, decomposition=False)
        Z = C1S[-1] @ S2S[0]
        del P1S, D1S, C1S, S1S, P2S, D2S, C2S, S2S
        Z = Z.view(3**l1, 3**l2, 3**l3)
    return Z / torch.norm(Z)


def cartesian_3j(l1: int, l2: int, l3: int, dtype=None, device=None) -> torch.Tensor:
    '''
    In practical atomistic machine learning models, very high-order Cartesian tensors 
    are typically not required. However, if one needs to compute cartesian_nj, caching 
    partial results becomes necessary.
    '''
    assert abs(l2 - l3) <= l1 <= l2 + l3
    assert isinstance(l1, int) and isinstance(l2, int) and isinstance(l3, int)

    # === cache directory ===
    cache_dir = CARTNN_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{l1}_{l2}_{l3}.pt"
    path = cache_dir / filename

    dtype, device = explicit_default_types(dtype, device)

    # === try to load === 
    Z = None
    if path.exists():
        try:
            Z = torch.load(path, weights_only=False)
        except Exception as e:
            print(f"[cartnn] Warning: Failed to load cache {path}: {e}")
            Z = None

    # === fallback: compute manually ===
    if Z is None:
        Z = _cartesian_3j(l1, l2, l3)
        if not path.exists():
            try:
                torch.save(Z, path)
                file_size = path.stat().st_size / (1024 ** 3)  # GB
                if file_size > 5:
                    print(f"[cartnn] Cache {path} is {file_size:.2f} GB > 5GB, removing.")
                    path.unlink(missing_ok=True)
            except Exception as e:
                print(f"[cartnn] Warning: Failed to save cache {path}: {e}")

    return Z.to(dtype=dtype, device=device, copy=True, memory_format=torch.contiguous_format)
