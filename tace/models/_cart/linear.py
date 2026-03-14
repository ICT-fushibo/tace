################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import string
from math import sqrt
from typing import Dict, List, Optional


import torch


from ..env import TACE_WEIGHT_INIT

class _Linear(torch.nn.Module):
    def __init__(
        self,
        l: int,
        in_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.l = l
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = 1.0 / sqrt(in_dim)
        letters = [letter for letter in list(string.ascii_letters)[3:] if letter != 'C']
        in1 = 'b' + ''.join(letters[:self.l]) + 'c'
        in2 = 'Cc'
        out = 'b' + ''.join(letters[:self.l]) + 'C'
        self.expr = f'{in1}, {in2} -> {out}'

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = weight * self.alpha
        weight = weight.view(self.out_dim, self.in_dim)
        x = torch.einsum(self.expr, x, weight)
        if self.l == 0 and bias is not None:
            x = x + bias.unsqueeze(0)
        return x


class _ElementLinear(torch.nn.Module):
    def __init__(
        self,
        l: int,
        in_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.l = l
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = 1.0 / sqrt(in_dim)
        letters = [letter for letter in list(string.ascii_letters)[3:] if letter != 'C']
        in1 = 'b' + ''.join(letters[:self.l]) + 'c'
        in2 = 'bCc'
        out = 'b' + ''.join(letters[:self.l]) + 'C'
        self.expr = f'{in1}, {in2} -> {out}'

    def forward(self, x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = weight * self.alpha
        weight = torch.einsum('bz, zi -> bi', y, weight)
        weight = weight.view(-1, self.out_dim, self.in_dim)
        x = torch.einsum(self.expr, x, weight)
        if self.l == 0 and bias is not None:
            bias = torch.einsum('bz, zi -> bi', y, bias)
            x = x + bias
        return x
    
class Linear(torch.nn.Module):

    weight_numel: int

    def __init__(
        self,
        ls: List[int],
        channels_in: int | List[int],
        channels_out: int | List[int],
        bias: bool,
    ) -> None:
        super().__init__()
        """
        assume l in ls have been sorted and unique
        """
        assert isinstance(ls, List)
        self.ls = ls
        if isinstance(channels_in, int):
            channels_in = [channels_in] * len(ls)
        if isinstance(channels_out, int):
            channels_out = [channels_out] * len(ls)
        self.channels_in = channels_in
        self.channels_out = channels_out
        assert len(self.channels_out) == len(self.ls), \
            f"channels_out ({len(self.channels_out)}) != ls ({len(self.ls)})"
        assert len(self.channels_in) == len(self.ls), \
            f"channels_in ({len(self.channels_in)}) != ls ({len(self.ls)})"

        bias_idx = next((i for i, l in enumerate(self.ls) if l == 0), None)

        weight_numel = 0
        self.weight_slice = []
        for c_in, c_out in zip(self.channels_in, self.channels_out):
            self.weight_slice.append(slice(weight_numel, weight_numel+c_in*c_out))
            weight_numel += c_in*c_out
        self.weight_numel = weight_numel
        self.weight = torch.nn.Parameter(torch.empty(self.weight_numel))
        if bias and bias_idx is not None:
            self.bias = torch.nn.Parameter(torch.empty(self.channels_out[bias_idx]))
        else:
            self.register_parameter("bias", None)

        self.linears = torch.nn.ModuleList(
                _Linear(l, c_in, c_out) 
                for l, c_in, c_out in zip(self.ls, self.channels_in, self.channels_out)
        )

        self.reset_parameters()

    def forward(self, in_dict: Dict[int, torch.Tensor], y: Optional[torch.Tensor] = None) -> Dict[int, torch.Tensor]:            
        out_dict = {}
        for idx, l in enumerate(self.ls):
            out_dict[l] = self.linears[idx](
                in_dict[l], 
                self.weight[self.weight_slice[idx]], 
                self.bias,
            )
        return out_dict

    def reset_parameters(self):
        if TACE_WEIGHT_INIT == 'uniform':
            torch.nn.init.uniform_(self.weight, -sqrt(3), sqrt(3))
        else:
            torch.nn.init.normal_(self.weight)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
            
    def __repr__(self):
        in_list = []
        out_list = []
        for l, c_in, c_out in zip(self.ls, self.channels_in, self.channels_out):
            in_list.append(f"{c_in}x{l}e")
            out_list.append(f"{c_out}x{l}e")
        repr_str = "+".join(in_list) + " -> "  + "+".join(out_list)
        return f"{self.__class__.__name__}({repr_str})"
    

class ElementLinear(torch.nn.Module):

    weight_numel: int

    def __init__(
        self,
        ls: List[int],
        channels_in: int | List[int],
        channels_out: int | List[int],
        bias: bool,
        num_elements: int,
    ) -> None:
        super().__init__()
        """
        assume l in ls have been sorted and unique
        """
        assert isinstance(ls, List)
        self.ls = ls
        if isinstance(channels_in, int):
            channels_in = [channels_in] * len(ls)
        if isinstance(channels_out, int):
            channels_out = [channels_out] * len(ls)
        self.channels_in = channels_in
        self.channels_out = channels_out
        assert len(self.channels_out) == len(self.ls), \
            f"channels_out ({len(self.channels_out)}) != ls ({len(self.ls)})"
        assert len(self.channels_in) == len(self.ls), \
            f"channels_in ({len(self.channels_in)}) != ls ({len(self.ls)})"

        bias_idx = next((i for i, l in enumerate(self.ls) if l == 0), None)

        weight_numel = 0
        self.weight_slice = []
        for c_in, c_out in zip(self.channels_in, self.channels_out):
            self.weight_slice.append(slice(weight_numel, weight_numel+c_in*c_out))
            weight_numel += c_in*c_out
        self.weight_numel = weight_numel
        self.weight = torch.nn.Parameter(torch.empty(num_elements, self.weight_numel))
        if bias and bias_idx is not None:
            self.bias = torch.nn.Parameter(torch.empty(num_elements, self.channels_out[bias_idx]))
        else:
            self.register_parameter("bias", None)

        self.linears = torch.nn.ModuleList(
                _ElementLinear(l, c_in, c_out) 
                for l, c_in, c_out in zip(self.ls, self.channels_in, self.channels_out)
        )

        self.reset_parameters()

    def reset_parameters(self):
        if TACE_WEIGHT_INIT == 'uniform':
            torch.nn.init.uniform_(self.weight, -sqrt(3), sqrt(3))
        else:
            torch.nn.init.normal_(self.weight)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, in_dict: Dict[int, torch.Tensor], node_attrs: torch.Tensor) -> Dict[int, torch.Tensor]:
        out_dict = {}
        for idx, l in enumerate(self.ls):
            out_dict[l] = self.linears[idx](
                in_dict[l], 
                node_attrs,
                self.weight[:, self.weight_slice[idx]], 
                self.bias,
            )
        return out_dict
    
    def __repr__(self):
        in_list = []
        out_list = []
        for l, c_in, c_out in zip(self.ls, self.channels_in, self.channels_out):
            in_list.append(f"{c_in}x{l}e")
            out_list.append(f"{c_out}x{l}e")
        repr_str = "+".join(in_list) + " -> "  + "+".join(out_list)
        return f"{self.__class__.__name__}({repr_str})"
    

# class _Shortcut(torch.nn.Module):
#     def __init__(
#             self, 
#             ls_in: List[int],
#             ls_out: List[int],
#             channels_in: int | List[int],
#             channels_out: int | List[int],
#         ) -> None:
#         super().__init__()
        
#         def _to_list(x: int | List[int]) -> List[int]:
#             if isinstance(x, int):
#                 return [x]
#             return x
        
#         self.ls_in = ls_in
#         self.ls_out = ls_out
#         self.channels_in = _to_list(channels_in)


#         self.weight = torch.nn.Parameter(
#             torch.randn((self.lmax + 1), out_features, in_features)
#         )
#         self.alpha = 1 / math.sqrt(self.in_features)

#         self.bias = torch.nn.Parameter(torch.zeros(out_features))

#         expand_index = torch.zeros([(lmax + 1) ** 2]).long()
#         for lval in range(lmax + 1):
#             start_idx = lval**2
#             length = 2 * lval + 1
#             expand_index[start_idx : (start_idx + length)] = lval
#         self.register_buffer("expand_index", expand_index, persistent=False)

#     def reset_parameters(self):
#         if TACE_WEIGHT_INIT == 'uniform':
#             torch.nn.init.uniform_(self.weight, -math.sqrt(3), math.sqrt(3))
#         else:
#             torch.nn.init.normal_(self.weight)

#         if self.bias is not None:
#             torch.nn.init.zeros_(self.bias)

#     def forward(self, x):
#         weight = self.weight * self.alpha
#         weight = torch.index_select(
#             self.weight, dim=0, index=self.expand_index
#         ) 
#         out = torch.einsum(
#             "bmi, moi -> bmo", x, weight
#         )  # [N, (L_max + 1) ** 2, C_out]
#         bias = self.bias.view(1, 1, self.out_features)
#         out[:, 0:1, :] = out.narrow(1, 0, 1) + bias
#         return out

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax})"