################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math


import torch
import torch.nn.functional as F


from e3nn import o3



class torchLinear(torch.nn.Module):

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.randn((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.alpha = 1.0 / math.sqrt(in_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight * self.alpha, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class mlpLinear(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        alpha: float = 1.0,
        bias: bool = False,
        std: float = math.sqrt(3),
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.weight = torch.nn.Parameter(torch.empty((in_dim, out_dim)))
        torch.nn.init.uniform_(self.weight, -std, std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.alpha
        if self.bias is None:
            return torch.mm(input, weight)
        else:
            return torch.addmm(self.bias, input, weight)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim} bias={ self.bias is not None})"


class e3nnLinear(torch.nn.Module):
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


class e3nnElementLinear(torch.nn.Module):
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
    

