################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math


import torch
import torch.nn.functional as F
from e3nn import o3


from tace.utils.env import get_tace_use_matrix_weight


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
        use_matrix_weight: bool = get_tace_use_matrix_weight()
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.use_matrix_weight = use_matrix_weight == '1'

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

        if self.use_matrix_weight:
            if internal_weights:
                self.weight = torch.nn.ParameterList()
                for ins in self.linear.instructions:
                    self.weight.append(
                        torch.nn.Parameter(
                            torch.randn(ins.path_shape)
                        )
                    )
            else:
                self.register_parameter("weight", None)
        else:
            if internal_weights:
                self.weight = torch.nn.Parameter(torch.empty(self.weight_numel))
                torch.nn.init.normal_(self.weight)
            else:
                self.register_parameter("weight", None)

        if bias and bias_acc > 0:
            self.bias = torch.nn.Parameter(torch.zeros(bias_acc))
        else:
            self.register_parameter("bias", None)


    def forward(self, x: torch.Tensor, weight = None) -> torch.Tensor:
        if self.weight is not None:
            weight = self.weight

        if self.use_matrix_weight:
            weight = torch.cat([w.view(-1) for w in weight], dim=-1)

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
        use_matrix_weight: bool = get_tace_use_matrix_weight()
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_elements = num_elements
        self.use_matrix_weight = use_matrix_weight == '1'

        self.linear = o3.Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            internal_weights=False,
            shared_weights=False,
        )
        weight_numel = self.linear.weight_numel

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

        if self.use_matrix_weight:
            self.weight = torch.nn.ParameterList()
            for ins in self.linear.instructions:
                self.weight.append(
                    torch.nn.Parameter(
                        torch.randn(num_elements, *ins.path_shape)
                    )
                )
            if bias and bias_acc > 0:
                self.bias = torch.nn.Parameter(torch.zeros(num_elements * bias_acc))
            else:
                self.register_parameter("bias", None)
        else:
            self.weight = torch.nn.Parameter(torch.empty(num_elements, weight_numel))
            torch.nn.init.normal_(self.weight)
            if bias and bias_acc > 0:
                self.bias = torch.nn.Parameter(torch.zeros(num_elements, bias_acc))
            else:
                self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        if self.use_matrix_weight:
            weight = torch.cat([w.view(self.num_elements, -1) for w in self.weight], dim=-1)
            bias = self.bias.view(self.num_elements, -1)
        else:
            weight = self.weight
            bias = self.bias

        node_type = y.argmax(dim=-1)
        weight = weight[node_type]
        # weight = torch.einsum("bz,zi->bi", y, self.weight)
        out = self.linear(x, weight)
        if bias is not None:
            bias = torch.einsum("bz,zi->bi", y, bias)
            for sl, bias_sl in zip(self._0e_slices, self._bias_slices):
                out[:, sl] = out[:, sl] + bias[:, bias_sl]
        return out

    def __repr__(self):
        return "Element" + repr(self.linear) + f"(bias={self.bias is not None})"
    



def switch_e3nn_weight_layout(model: torch.nn.Module, target: str = "toggle") -> torch.nn.Module:

    assert target in {"toggle", "flat", "matrix"}

    def is_supported_module(m: torch.nn.Module) -> bool:
        return (
            hasattr(m, "linear")
            and hasattr(m.linear, "instructions")
            and hasattr(m, "weight")
            and hasattr(m, "use_matrix_weight")
        )

    def is_element_linear(m: torch.nn.Module) -> bool:
        return hasattr(m, "num_elements")

    def weight_is_matrix_layout(m: torch.nn.Module) -> bool:
        return isinstance(m.weight, torch.nn.ParameterList)

    def set_weight(m: torch.nn.Module, new_weight):
        if hasattr(m, "weight"):
            delattr(m, "weight")
        m.weight = new_weight

    def flat_to_matrix(m: torch.nn.Module):
        if m.weight is None:
            return

        old_weight = m.weight
        requires_grad = old_weight.requires_grad

        matrix_weights = torch.nn.ParameterList()
        offset = 0

        with torch.no_grad():
            if is_element_linear(m):
                # old_weight: [num_elements, weight_numel]
                for ins in m.linear.instructions:
                    size = int(torch.tensor(ins.path_shape).prod().item())
                    chunk = old_weight[:, offset:offset + size]
                    chunk = chunk.reshape(m.num_elements, *ins.path_shape)
                    matrix_weights.append(torch.nn.Parameter(chunk.clone(), requires_grad=requires_grad))
                    offset += size

                # bias:
                # flat layout:   [num_elements, bias_dim]
                # matrix layout: [num_elements * bias_dim]
                if m.bias is not None and m.bias.dim() == 2:
                    m.bias = torch.nn.Parameter(
                        m.bias.reshape(-1).clone(),
                        requires_grad=m.bias.requires_grad,
                    )

            else:
                # old_weight: [weight_numel]
                for ins in m.linear.instructions:
                    size = int(torch.tensor(ins.path_shape).prod().item())
                    chunk = old_weight[offset:offset + size]
                    chunk = chunk.reshape(*ins.path_shape)
                    matrix_weights.append(torch.nn.Parameter(chunk.clone(), requires_grad=requires_grad))
                    offset += size

        set_weight(m, matrix_weights)
        m.use_matrix_weight = True

    def matrix_to_flat(m: torch.nn.Module):
        if m.weight is None:
            return

        old_weights = list(m.weight)
        requires_grad = any(w.requires_grad for w in old_weights)

        with torch.no_grad():
            if is_element_linear(m):
                # 每个 w: [num_elements, *path_shape]
                chunks = [w.reshape(m.num_elements, -1) for w in old_weights]
                flat_weight = torch.cat(chunks, dim=-1)

                # bias:
                # matrix layout: [num_elements * bias_dim]
                # flat layout:   [num_elements, bias_dim]
                if m.bias is not None and m.bias.dim() == 1:
                    m.bias = torch.nn.Parameter(
                        m.bias.reshape(m.num_elements, -1).clone(),
                        requires_grad=m.bias.requires_grad,
                    )
            else:
                chunks = [w.reshape(-1) for w in old_weights]
                flat_weight = torch.cat(chunks, dim=-1)

        set_weight(m, torch.nn.Parameter(flat_weight.clone(), requires_grad=requires_grad))
        m.use_matrix_weight = False

    def convert_module(m: torch.nn.Module):
        if not is_supported_module(m):
            return

        currently_matrix = weight_is_matrix_layout(m)

        if target == "toggle":
            to_matrix = not currently_matrix
        else:
            to_matrix = target == "matrix"

        if to_matrix and not currently_matrix:
            flat_to_matrix(m)
        elif not to_matrix and currently_matrix:
            matrix_to_flat(m)

    for module in model.modules():
        convert_module(module)

    return model