# ################################################################################
# # Authors: Zemin Xu
# # License: MIT, see LICENSE.md
# ################################################################################

# from typing import Callable, TypeVar, Union

# import torch

# from ..utils._global import DTYPE


# PrecisionDType = Union[str, torch.dtype]
# T = TypeVar("T")


# class MinimumPrecision:
#     def __init__(self, dtype: PrecisionDType = "float32") -> None:
#         if dtype not in DTYPE or DTYPE[dtype] is None:
#             raise ValueError(f"Unsupported numerical dtype: {dtype!r}")
#         resolved = DTYPE[dtype]
#         if resolved in (torch.float16, torch.bfloat16):
#             resolved = torch.float32
#         if resolved not in (torch.float32, torch.float64):
#             raise ValueError(f"Unsupported numerical dtype: {dtype!r}")
#         self.dtype = resolved

#     def compute_dtype(self, *values) -> torch.dtype:
#         if self.dtype == torch.float64:
#             return torch.float64
#         for value in values:
#             if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
#                 return torch.float64
#         return torch.float32

#     @staticmethod
#     def context(reference: torch.Tensor) -> torch.autocast:
#         return torch.autocast(device_type=reference.device.type, enabled=False)

#     def cast(self, value, *, dtype: torch.dtype):
#         if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
#             return value.to(dtype=dtype)
#         return value

#     def call(
#         self,
#         fn: Callable[..., T],
#         *args,
#         reference: Union[torch.Tensor, None] = None,
#         **kwargs,
#     ) -> T:
#         if reference is None:
#             reference = next(
#                 value for value in (*args, *kwargs.values())
#                 if isinstance(value, torch.Tensor)
#             )
#         dtype = self.compute_dtype(*args, *kwargs.values())
#         args = tuple(self.cast(value, dtype=dtype) for value in args)
#         kwargs = {
#             key: self.cast(value, dtype=dtype)
#             for key, value in kwargs.items()
#         }
#         with self.context(reference):
#             return fn(*args, **kwargs)

#     def bmm(
#         self,
#         left: torch.Tensor,
#         right: torch.Tensor,
#     ) -> torch.Tensor:
#         dtype = self.compute_dtype(left, right)
#         with self.context(right):
#             return torch.bmm(
#                 left.to(device=right.device, dtype=dtype),
#                 right.to(dtype=dtype),
#             )

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}(dtype={self.dtype})"


# DEFAULT_MINIMUM_PRECISION = MinimumPrecision()
