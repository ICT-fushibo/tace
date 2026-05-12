################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import torch
from e3nn import o3
from e3nn.nn import Activation


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
        node_type = y.argmax(dim=-1)
        weight = self.weight[node_type]
        # weight = torch.einsum("bz,zi->bi", y, self.weight)
        out = self.linear(x, weight)
        if self.bias is not None:
            bias = torch.einsum("bz,zi->bi", y, self.bias)
            for sl, bias_sl in zip(self._0e_slices, self._bias_slices):
                out[:, sl] = out[:, sl] + bias[:, bias_sl]
        return out

    def __repr__(self):
        return "Element" + repr(self.linear) + f"(bias={self.bias is not None})"
    

class GatedLinearUnit(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        bias: bool = True,
        num_elements: int,
        activation: torch.nn.Module = torch.nn.Sigmoid(),
    ):
        super().__init__()

        irreps_gated = irreps_out
        irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in irreps_out)
        self.mul = o3.ElementwiseTensorProduct(irreps_gated, irreps_gates)
        self.act = Activation(irreps_gates, [activation])
        self.num_elements = num_elements

        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)

        self.gate_linear = o3.Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_gates.regroup(),
            internal_weights=False,
            shared_weights=False,
        )
        self.linear = o3.Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            internal_weights=False,
            shared_weights=False,
        )

        self.weight1 = torch.nn.Parameter(
            torch.empty(self.gate_linear.weight_numel)
        )
        self.weight2 = torch.nn.Parameter(
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
        torch.nn.init.normal_(self.weight1)
        torch.nn.init.normal_(self.weight2)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        gate = self.act(self.gate_linear(x, self.weight1))
        weight2 = torch.einsum("bz,zi->bi", y, self.weight2)
        gated = self.linear(x, weight2)
        out = self.mul(gated, gate)
        
        if self.bias is not None:
            bias = torch.einsum("bz,zi->bi", y, self.bias)
            for sl, bias_sl in zip(self._0e_slices, self._bias_slices):
                out[:, sl] += bias[:, bias_sl]
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"gate={self.gate_linear}, "
            f"linear={self.linear}, "
            f"bias={self.bias is not None})"
        )