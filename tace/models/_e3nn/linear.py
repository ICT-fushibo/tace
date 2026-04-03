################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Optional


import torch
from e3nn import o3


class Linear(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        bias: bool = True,
        internal_weights: bool = True,
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        self.linear = o3.Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            internal_weights=False,
            shared_weights=False,
        )
        self.weight_numel = self.linear.weight_numel

        self._0e_muls = []
        self._0e_slices = []
        self._bias_slices = []
        acc = 0
        bias_acc = 0
        for mul, ir in self.irreps_out:
            dim = mul * ir.dim
            if ir.l == 0 and ir.p == 1:
                self._0e_muls.append(mul)
                self._0e_slices.append(slice(acc, acc + dim))
                self._bias_slices.append(slice(bias_acc, bias_acc + dim))
                bias_acc += dim
            acc += dim

        if internal_weights:
            self.weight = torch.nn.Parameter(
                torch.empty(self.weight_numel)
            )
        else:
            self.register_parameter("weight", None)

        if bias and bias_acc > 0:
            self.bias = torch.nn.Parameter(
                torch.empty(bias_acc)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, weight = None) -> torch.Tensor:
        if self.weight is not None:
            weight = self.weight
        out = self.linear(x, weight)
        if self.bias is not None:
            for sl, bias_sl in zip(self._0e_slices, self._bias_slices):
                out[:, sl] = out[:, sl] + self.bias[bias_sl].unsqueeze(0)
        return out

    def __repr__(self):
        return repr(self.linear) + f"(bias={self.bias is not None})"


class ElementLinear(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        bias: bool = True,
        num_elements: int,
    ):
        super().__init__()

        self.num_elements = num_elements

        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)

        self.linear = o3.Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            internal_weights=False,
            shared_weights=False,
        )

        self.weight = torch.nn.Parameter(
            torch.empty(num_elements, self.linear.weight_numel)
        )

        self._0e_muls = []
        self._0e_slices = []
        self._bias_slices = []
        acc = 0
        bias_acc = 0
        for mul, ir in irreps_out:
            dim = mul * ir.dim
            if ir.l == 0 and ir.p == 1:
                self._0e_muls.append(mul)
                self._0e_slices.append(slice(acc, acc + dim))
                self._bias_slices.append(slice(bias_acc, bias_acc + dim))
                bias_acc += dim
            acc += dim

        if bias and bias_acc > 0:
            self.bias = torch.nn.Parameter(
                torch.empty(num_elements, bias_acc)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        weight = torch.einsum("bz,zi->bi", y, self.weight)
        out = self.linear(x, weight)
        if self.bias is not None:
            bias = torch.einsum("bz,zi->bi", y, self.bias)
            for sl, bias_sl in zip(self._0e_slices, self._bias_slices):
                out[:, sl] = out[:, sl] + bias[:, bias_sl]
        return out

    def __repr__(self):
        return repr(self.linear) + f"(bias={self.bias is not None})"
    


# class IdentityLinear(torch.nn.Module):
#     def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
#         super().__init__()

#         self.linear = o3.Linear(
#             irreps_in=irreps_in,
#             irreps_out=irreps_out,
#             internal_weights=False,
#             shared_weights=True,
#         )

#         weight = torch.zeros(self.linear.weight_numel)

#         offset = 0
#         for ins in self.linear.instructions:
#             size = math.prod(ins.path_shape)
#             if ins.i_in == -1:
#                 weight[offset:offset + size] = 0.0

#             else:
#                 mul_in, mul_out = ins.path_shape
#                 k = min(mul_in, mul_out)

#                 block = torch.zeros(mul_out, mul_in)
#                 block[:k, :k] = torch.eye(k)
#                 block = block / ins.path_weight

#                 weight[offset:offset + size] = block.reshape(-1)

#             offset += size

#         self.register_buffer("weight", weight, persistent=False)

#     def forward(self, x):
#         return self.linear(x, self.weight)

#     def forward(self, x):
#         return self.linear(x, self.weight)


# class NormGatedLinearUnit(torch.nn.Module):
#     def __init__(
#         self,
#         irreps_in: o3.Irreps,
#         irreps_out: o3.Irreps,
#         *,
#         bias: bool = True,
#         num_elements: int,
#         activation: torch.nn.Module = torch.nn.Sigmoid(),
#     ):
#         super().__init__()

#         self.nonlinear = NormLinearUnit(irreps_out, activation)

#         self.num_elements = num_elements

#         irreps_in = o3.Irreps(irreps_in)
#         irreps_out = o3.Irreps(irreps_out)

#         self.linear = o3.Linear(
#             irreps_in=irreps_in,
#             irreps_out=irreps_out,
#             internal_weights=False,
#             shared_weights=False,
#         )

#         self.weight1 = torch.nn.Parameter(
#             torch.empty(self.linear.weight_numel)
#         )
#         self.weight2 = torch.nn.Parameter(
#             torch.empty(num_elements, self.linear.weight_numel)
#         )

#         self._0e_muls = []
#         self._0e_slices = []
#         self._bias_slices = []
#         acc = 0
#         bias_acc = 0
#         for mul, ir in irreps_out:
#             dim = mul * ir.dim
#             if ir.l == 0 and ir.p == 1:
#                 self._0e_muls.append(mul)
#                 self._0e_slices.append(slice(acc, acc + dim))
#                 self._bias_slices.append(slice(bias_acc, bias_acc + dim))
#                 bias_acc += dim
#             acc += dim

#         if bias and bias_acc > 0:
#             self.bias = torch.nn.Parameter(
#                 torch.empty(num_elements, bias_acc)
#             )
#         else:
#             self.register_parameter("bias", None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         torch.nn.init.normal_(self.weight1)
#         torch.nn.init.normal_(self.weight2)
#         if self.bias is not None:
#             torch.nn.init.zeros_(self.bias)

#     def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

#         gate = self.linear(x, self.weight1)
#         weight2 = torch.einsum("bz,zi->bi", y, self.weight2)
#         gated = self.linear(x, weight2)
#         out = self.nonlinear(gate, gated)
        
#         if self.bias is not None:
#             bias = torch.einsum("bz,zi->bi", y, self.bias)
#             for sl, bias_sl in zip(self._0e_slices, self._bias_slices):
#                 out[:, sl] += bias[:, bias_sl]
#         return out

#     def __repr__(self):
#         return (
#             f"{self.__class__.__name__}("
#             f"linear1={self.linear}, "
#             f"linear2={self.linear}, "
#             f"bias={self.bias is not None})"
#         )
    

# class NormGatedLinearUnit(torch.nn.Module):
#     def __init__(
#         self,
#         irreps_in: o3.Irreps,
#         irreps_out: o3.Irreps,
#         *,
#         bias: bool = True,
#         num_elements: int,
#         activation: torch.nn.Module = torch.nn.Sigmoid(),
#     ):
#         super().__init__()

#         self.nonlinear = NormLinearUnit(irreps_out, activation)

#         self.num_elements = num_elements

#         irreps_in = o3.Irreps(irreps_in)
#         irreps_out = o3.Irreps(irreps_out)

#         self.linear = o3.Linear(
#             irreps_in=irreps_in,
#             irreps_out=irreps_out,
#             internal_weights=False,
#             shared_weights=False,
#         )

#         self.weight1 = torch.nn.Parameter(
#             torch.empty(self.linear.weight_numel)
#         )
#         self.weight2 = torch.nn.Parameter(
#             torch.empty(num_elements, self.linear.weight_numel)
#         )

#         self._0e_muls = []
#         self._0e_slices = []
#         self._bias_slices = []
#         acc = 0
#         bias_acc = 0
#         for mul, ir in irreps_out:
#             dim = mul * ir.dim
#             if ir.l == 0 and ir.p == 1:
#                 self._0e_muls.append(mul)
#                 self._0e_slices.append(slice(acc, acc + dim))
#                 self._bias_slices.append(slice(bias_acc, bias_acc + dim))
#                 bias_acc += dim
#             acc += dim

#         if bias and bias_acc > 0:
#             self.bias = torch.nn.Parameter(
#                 torch.empty(num_elements, bias_acc)
#             )
#         else:
#             self.register_parameter("bias", None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         torch.nn.init.normal_(self.weight1)
#         torch.nn.init.normal_(self.weight2)
#         if self.bias is not None:
#             torch.nn.init.zeros_(self.bias)

#     def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

#         gate = self.linear(x, self.weight1)
#         weight2 = torch.einsum("bz,zi->bi", y, self.weight2)
#         gated = self.linear(x, weight2)
#         out = self.nonlinear(gate, gated)
        
#         if self.bias is not None:
#             bias = torch.einsum("bz,zi->bi", y, self.bias)
#             for sl, bias_sl in zip(self._0e_slices, self._bias_slices):
#                 out[:, sl] += bias[:, bias_sl]
#         return out

#     def __repr__(self):
#         return (
#             f"{self.__class__.__name__}("
#             f"linear1={self.linear}, "
#             f"linear2={self.linear}, "
#             f"bias={self.bias is not None})"
#         )

# class OneEGatedLinearUnit(torch.nn.Module):
#     def __init__(
#         self,
#         irreps_in: o3.Irreps,
#         irreps_out: o3.Irreps,
#         *,
#         bias: bool = True,
#         num_elements: int,
#         activation: torch.nn.Module = torch.nn.Sigmoid(),
#     ):
#         super().__init__()

#         irreps_gated = irreps_out
#         irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in irreps_out)
#         self.mul = o3.ElementwiseTensorProduct(irreps_gated, irreps_gates)
#         self.act = activation
#         self.num_elements = num_elements

#         irreps_in = o3.Irreps(irreps_in)
#         irreps_out = o3.Irreps(irreps_out)

#         self.gate_linear = o3.Linear(
#             irreps_in=irreps_in,
#             irreps_out=irreps_gates.regroup(),
#             internal_weights=False,
#             shared_weights=False,
#         )
#         self.linear = o3.Linear(
#             irreps_in=irreps_in,
#             irreps_out=irreps_out,
#             internal_weights=False,
#             shared_weights=False,
#         )

#         self.weight1 = torch.nn.Parameter(
#             torch.empty(self.gate_linear.weight_numel)
#         )
#         self.weight2 = torch.nn.Parameter(
#             torch.empty(num_elements, self.linear.weight_numel)
#         )

#         self._0e_muls = []
#         self._0e_slices = []
#         self._bias_slices = []
#         acc = 0
#         bias_acc = 0
#         for mul, ir in irreps_out:
#             dim = mul * ir.dim
#             if ir.l == 0 and ir.p == 1:
#                 self._0e_muls.append(mul)
#                 self._0e_slices.append(slice(acc, acc + dim))
#                 self._bias_slices.append(slice(bias_acc, bias_acc + dim))
#                 bias_acc += dim
#             acc += dim

#         if bias and bias_acc > 0:
#             self.bias = torch.nn.Parameter(
#                 torch.empty(num_elements, bias_acc)
#             )
#         else:
#             self.register_parameter("bias", None)
    
#         self.reset_parameters()

#     def reset_parameters(self):
#         torch.nn.init.normal_(self.weight1)
#         torch.nn.init.normal_(self.weight2)
#         if self.bias is not None:
#             torch.nn.init.zeros_(self.bias)

#     def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

#         gate = self.act(self.gate_linear(x, self.weight1))
#         weight2 = torch.einsum("bz,zi->bi", y, self.weight2)
#         gated = self.linear(x, weight2)
#         out = self.mul(gated, gate)
        
#         if self.bias is not None:
#             bias = torch.einsum("bz,zi->bi", y, self.bias)
#             for sl, bias_sl in zip(self._0e_slices, self._bias_slices):
#                 out[:, sl] += bias[:, bias_sl]
#         return out

#     def __repr__(self):
#         return (
#             f"{self.__class__.__name__}("
#             f"gate={self.gate_linear}, "
#             f"linear={self.linear}, "
#             f"bias={self.bias is not None})"
#         )
    
