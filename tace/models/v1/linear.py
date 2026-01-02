################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import string
from math import sqrt
from typing import Dict, List, Optional


import torch


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        l: int = -1,
        **kwargs,
    ) -> None:
        super().__init__()
        assert l >= 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = 1.0 / sqrt(in_dim)
        self.l = l
        self.weight = torch.nn.Parameter(torch.empty((in_dim, out_dim)))
        torch.nn.init.uniform_(self.weight, -sqrt(3), sqrt(3))
        if bias and l == 0:
            self.bias = torch.nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

        letters = [letter for letter in list(string.ascii_letters)[3:] if letter != 'C']
        in1 = 'bc' + ''.join(letters[:self.l])
        in2 = 'cC'
        out = 'bC' + ''.join(letters[:self.l])
        self.expr = f'{in1}, {in2} -> {out}'
        
    def forward(self, t: torch.Tensor, node_attrs: Optional[torch.Tensor] = None) -> torch.Tensor:
        W = self.weight 
        W = W * self.alpha
        t = torch.einsum(self.expr, t, W)
        if self.bias is not None:
            t = t + self.bias.unsqueeze(0)
        return t
    
    # def forward(
    #         self, 
    #         t: torch.Tensor, 
    #         node_attrs: Optional[torch.Tensor] = None, 
    #         s: Optional[slice] = None,
    #         bias: bool = True,
    #     ) -> torch.Tensor:
    #     W = self.weight if s is None else self.weight[s, :]
    #     W = W * self.alpha
    #     t = torch.einsum(self.expr, t, W)
    #     if self.bias is not None and bias:
    #         t = t + self.bias.unsqueeze(0)
    #     return t

    def __repr__(self):
        return f"{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim}, bias={self.bias is not None}, l={self.l})"


class ElementLinear(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        l: int = -1,
        atomic_numbers: List[int] = None,
        groups: Optional[List[List[int]]] = None,
        **kwargs,
    ):
        super().__init__()
        assert l >= 0
        assert atomic_numbers is not None
        num_elements = len(atomic_numbers)
        self.register_buffer(
            "num_elements", torch.tensor(num_elements, dtype=torch.int64)
        )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = 1.0 / sqrt(in_dim)
        self.l = l
        self.atomic_numbers = atomic_numbers
        self.groups = groups
        self.num_groups = len(groups) if groups is not None else None
        # if self.groups is None:
        self.weights = torch.nn.Parameter(
            torch.empty(num_elements, out_dim, in_dim)
        )
        torch.nn.init.uniform_(self.weights, -sqrt(3), sqrt(3))
        if bias and l == 0:
            self.bias = torch.nn.Parameter(torch.empty(num_elements, out_dim))
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)
        
        letters = [c for c in string.ascii_letters[3:] if c not in ['C', 'z']]
        self.expr = (
            f'bz, zCc, bc{"".join(letters[:self.l])} -> '
            f'bC{"".join(letters[:self.l])}'
        )
        # else:
        #     assert sum(len(g) for g in groups) == num_elements
        #     elem2gid = {}
        #     for gid, g in enumerate(groups):
        #         for elem in g:
        #             elem2gid[elem] = gid
        #     gid_map = [elem2gid[z] for z in atomic_numbers]
        #     self.register_buffer("elem2gid", torch.tensor(gid_map, dtype=torch.int64))
        #     self.weights = nn.Parameter(
        #         torch.empty(self.num_groups, out_dim, in_dim)
        #     )
        #     torch.nn.init.uniform_(self.weights, -sqrt(3), sqrt(3))
        #     if bias:
        #         self.bias = nn.Parameter(torch.empty(self.num_groups, out_dim))
        #         torch.nn.init.zeros_(self.bias)
        #     else:
        #         self.register_parameter("bias", None)


    def forward(self, t: torch.Tensor, node_attrs: Optional[torch.Tensor] = None) -> torch.Tensor:
        W = self.weights * self.alpha
        t = torch.einsum(self.expr, node_attrs, W, t)
        if self.bias is not None:
            b = torch.einsum('bz, zC -> bC', node_attrs, self.bias)
            t = t + b
        return t

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_dim={self.in_dim}, "
                    f"out_dim={self.out_dim}, bias={self.bias is not None}, l={self.l})")


# only for prod
class CWLinear(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        l: int = -1,
        **kwargs,
    ):
        super().__init__()
        assert l >= 0
        self.l = l
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channel = out_dim
        num_path = int(in_dim / out_dim)
        self.num_path = num_path
        self.alpha = 1.0 / sqrt(num_path)
        self.weight = torch.nn.Parameter(torch.empty(out_dim, num_path))
        torch.nn.init.uniform_(self.weight, -sqrt(3), sqrt(3))
        if bias and self.l == 0:
            self.bias = torch.nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)
        letters = list(string.ascii_letters[3:])
        self.expr = (
            f'abc{"".join(letters[:self.l])}, ca -> '
            f'bc{"".join(letters[:self.l])}'
        )

    def forward(self, t: torch.Tensor, node_attrs: Optional[torch.Tensor] = None) -> torch.Tensor:
        W = self.weight * self.alpha
        t = torch.einsum(self.expr, t, W)
        if self.bias is not None:
            t = t + self.bias.unsqueeze(0)
        return t

    def __repr__(self):
        return f"{self.__class__.__name__}(in_dim={self.num_path}, out_dim={self.channel}, bias={self.bias is not None}, l={self.l})"


# only for prod
class ElementCWLinear(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        l: int = -1,
        atomic_numbers: List[int] = None,
        **kwargs,
    ):
        super().__init__()
        assert l >= 0
        assert atomic_numbers is not None
        num_elements = len(atomic_numbers)
        self.register_buffer(
            "num_elements", torch.tensor(num_elements, dtype=torch.int64)
        )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channel = out_dim
        num_path = int(in_dim / out_dim)
        self.num_path = num_path
        self.atomic_numbers = atomic_numbers
        self.alpha = 1.0 / sqrt(num_path)
        self.l = l
        self.weights = torch.nn.Parameter(torch.empty(num_elements, out_dim, num_path))
        torch.nn.init.uniform_(self.weights, -sqrt(3), sqrt(3))
        if bias and l == 0:
            self.bias = torch.nn.Parameter(torch.zeros(num_elements, out_dim))
        else:
            self.register_parameter("bias", None)
        letters = [c for c in string.ascii_letters[3:] if c != 'z']
        self.expr = (
            f'bz, zca, abc{"".join(letters[:self.l])} -> '
            f'bc{"".join(letters[:self.l])}'
        )

    def forward(self, t: torch.Tensor, node_attrs: Optional[torch.Tensor] = None) -> torch.Tensor:
        W = self.weights * self.alpha
        t = torch.einsum(self.expr, node_attrs, W, t)
        if self.bias is not None:
            b = torch.einsum('bz, zC -> bC')
            t = t + b
        return t

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_dim={self.num_path}, "
                    f"out_dim=1, bias={self.bias is not None}, l={self.l})")


class SelfInteraction(torch.nn.Module):

    ls: List[int]

    def __init__(
        self,
        in_channel: int | List[int],
        out_channel: int | List[int],
        ls: List[int],
        bias: bool = False,
        atomic_numbers: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.ls = ls

        if isinstance(in_channel, int):
            in_channel = [in_channel] *len(ls)

        if isinstance(out_channel, int):
            out_channel = [out_channel] *len(ls)

        if atomic_numbers is None:
            self.linears = torch.nn.ModuleDict(
                {
                    str(l): Linear(
                        in_channel[idx],
                        out_channel[idx],
                        bias=(l == 0 and bias),
                        l=l,
                    )
                    for idx, l in enumerate(ls)
                }
            )
        else:
            self.linears = torch.nn.ModuleDict(
                {
                    str(l): ElementLinear(
                        in_channel[idx],
                        out_channel[idx],
                        bias=(l == 0 and bias),
                        l=l,
                        atomic_numbers=atomic_numbers,
                    )
                    for idx, l in enumerate(ls)
                }
            )

    def forward(
            self, 
            in_dict: Dict[int, torch.Tensor],
            node_attrs: Optional[torch.Tensor] = None,
        ) -> Dict[int, torch.Tensor]:
        out_dict = {}
        for l in self.ls:
            out_dict[l] = self.linears[str(l)](in_dict[l], node_attrs)
        return out_dict


class LoRALinear(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        l: int = -1,
        lora_r: int = 8,
        lora_alpha: float = 8.0,
        **kwargs,
    ):
        super().__init__()
        assert l >= 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.l = l
        self.alpha = 1.0 / sqrt(in_dim)
        self.use_lora = True
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_r
        self.weight = torch.nn.Parameter(torch.empty(in_dim, out_dim))
        torch.nn.init.uniform_(self.weight, -sqrt(3), sqrt(3))
        self.weight.requires_grad_(False)
        if bias and l == 0:
            self.bias = torch.nn.Parameter(torch.zeros(out_dim))
            self.bias.requires_grad_(False)
        else:
            self.register_parameter("bias", None)
        self.lora_A = torch.nn.Parameter(torch.zeros(in_dim, lora_r))
        self.lora_B = torch.nn.Parameter(torch.zeros(lora_r, out_dim))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=sqrt(5))
        torch.nn.init.zeros_(self.lora_B)

        letters = [c for c in string.ascii_letters[3:] if c != 'C']
        in1 = 'bc' + ''.join(letters[:self.l])
        in2 = 'cC'
        out = 'bC' + ''.join(letters[:self.l])
        self.expr = f'{in1}, {in2} -> {out}'

    def forward(self, t: torch.Tensor, node_attrs: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.use_lora:
            delta_w = self.lora_A @ (self.lora_B * self.scaling)
            W = self.weight + delta_w
        else:
            W = self.weight

        W = W * self.alpha
        t = torch.einsum(self.expr, t, W)

        if self.bias is not None:
            t = t + self.bias.unsqueeze(0)

        return t
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  in_dim      = {self.in_dim},\n"
            f"  out_dim     = {self.out_dim},\n"
            f"  bias        = {self.bias is not None},\n"
            f"  alpha       = {self.alpha:.2f},\n"
            f"  lora_r      = {self.lora_r},\n"
            f"  lora_alpha  = {self.lora_alpha},\n"
            f")"
        )


class LoRAElementLinear(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        l: int = -1,
        atomic_numbers: List[int] = None,
        lora_r: int = 8,
        lora_alpha: float = 8.0,
        element_aware: bool = True, 
        **kwargs,
    ):
        super().__init__()
        assert l >= 0
        assert atomic_numbers is not None
        num_elements = len(atomic_numbers)
        self.register_buffer("num_elements", torch.tensor(num_elements))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.l = l
        self.alpha = 1.0 / sqrt(in_dim)
        self.atomic_numbers = atomic_numbers
        self.use_lora = True
        self.element_aware = element_aware
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_r
        self.weights = torch.nn.Parameter(
            torch.empty(num_elements, out_dim, in_dim)
        )
        torch.nn.init.uniform_(self.weights, -sqrt(3), sqrt(3))
        self.weights.requires_grad_(False)
        if bias and l == 0:
            self.bias = torch.nn.Parameter(torch.zeros(num_elements, out_dim))
            self.bias.requires_grad_(False)
        else:
            self.register_parameter("bias", None)
        if self.element_aware:
            self.lora_A = torch.nn.Parameter(torch.zeros(num_elements, lora_r, in_dim))
            self.lora_B = torch.nn.Parameter(torch.zeros(num_elements, out_dim, lora_r))
        else:
            self.lora_A = torch.nn.Parameter(torch.zeros(lora_r, in_dim))
            self.lora_B = torch.nn.Parameter(torch.zeros(out_dim, lora_r))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
        letters = [c for c in string.ascii_letters[3:] if c not in ['C', 'z']]
        self.expr = (
            f'bz, zCc, bc{"".join(letters[:self.l])} -> '
            f'bC{"".join(letters[:self.l])}'
        )

    def forward(self, t: torch.Tensor, node_attrs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_lora:
            if self.element_aware:
                delta_w = torch.einsum(
                    'zri, zor -> zoi',
                    self.lora_A,
                    self.lora_B * self.scaling,
                )
            else:
                delta_w = (self.lora_B * self.scaling) @ self.lora_A 
                delta_w = delta_w.unsqueeze(0)
            W = self.weights + delta_w
        else:
            W = self.weights
        W = W * self.alpha
        t = torch.einsum(self.expr, node_attrs, W, t)
        if self.bias is not None:
            b = torch.einsum('bz, zC -> bC', node_attrs, self.bias)
            t = t + b
        return t

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  in_dim             = {self.in_dim},\n"
            f"  out_dim            = {self.out_dim},\n"
            f"  bias               = {self.bias is not None},\n"
            f"  alpha              = {self.alpha:.2f},\n"
            f"  lora_r             = {self.lora_r},\n"
            f"  lora_alpha         = {self.lora_alpha},\n"
            f"  lora_element_aware = {self.element_aware},\n"
            f")"
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim} bias={self.bias is not None}, alpha={self.alpha:.2f}, lora_r={self.lora_r}, lora_alpha={self.lora_alpha}, ) "


# only for prod
class LoRACWLinear(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        l: int = -1,
        lora_r: int = 8,
        lora_alpha: float = 8.0,
        **kwargs,
    ):
        super().__init__()
        assert l >= 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channel = out_dim
        self.num_path = int(in_dim / out_dim)
        self.l = l
        self.alpha = 1.0 / sqrt(self.num_path)
        self.use_lora = True
        self.lora_alpha = lora_alpha
        self.lora_r = lora_r
        self.scaling = lora_alpha / lora_r
        self.weight = torch.nn.Parameter(torch.empty(out_dim, self.num_path))
        torch.nn.init.uniform_(self.weight, -sqrt(3), sqrt(3))
        self.weight.requires_grad_(False)
        if bias and l == 0:
            self.bias = torch.nn.Parameter(torch.zeros(out_dim))
            self.bias.requires_grad_(False)
        else:
            self.register_parameter("bias", None)
        self.lora_A = torch.nn.Parameter(torch.zeros(self.num_path, lora_r))
        self.lora_B = torch.nn.Parameter(torch.zeros(lora_r, out_dim))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
        letters = list(string.ascii_letters[3:])
        self.expr = (
            f'abc{"".join(letters[:self.l])}, ca -> '
            f'bc{"".join(letters[:self.l])}'
        )

    def forward(self, t: torch.Tensor, node_attrs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_lora:
            delta_w = (self.lora_A @ self.lora_B.T) * self.scaling
            W = self.weight + delta_w
        else:
            W = self.weight
        W = W * self.alpha
        t = torch.einsum(self.expr, t, W)
        if self.bias is not None:
            t = t + self.bias.unsqueeze(0)
        return t

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  in_dim      = {self.in_dim},\n"
            f"  out_dim     = {self.out_dim},\n"
            f"  bias        = {self.bias is not None},\n"
            f"  alpha       = {self.alpha:.2f},\n"
            f"  lora_r      = {self.lora_r},\n"
            f"  lora_alpha  = {self.lora_alpha},\n"
            f")"
        )


# only for prod
class LoRAElementCWLinear(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        l: int = -1,
        atomic_numbers: List[int] = None,
        lora_r: int = 8,
        lora_alpha: float = 8.0,
        **kwargs,
    ):
        super().__init__()
        assert l >= 0
        assert atomic_numbers is not None
        self.in_dim = in_dim
        self.out_dim = out_dim
        num_elements = len(atomic_numbers)
        self.register_buffer("num_elements", torch.tensor(num_elements))
        self.channel = out_dim
        self.num_path = int(in_dim / out_dim)
        self.l = l
        self.alpha = 1.0 / sqrt(self.num_path)
        self.atomic_numbers = atomic_numbers
        self.use_lora = True
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_r
        self.weights = torch.nn.Parameter(
            torch.empty(num_elements, out_dim, self.num_path)
        )
        torch.nn.init.uniform_(self.weights, -sqrt(3), sqrt(3))
        self.weights.requires_grad_(False)
        if bias and l == 0:
            self.bias = torch.nn.Parameter(torch.zeros(num_elements, out_dim))
            self.bias.requires_grad_(False)
        else:
            self.register_parameter("bias", None)

        self.lora_A = torch.nn.Parameter(torch.zeros(self.num_path, lora_r))
        self.lora_B = torch.nn.Parameter(torch.zeros(lora_r, out_dim))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
        letters = [c for c in string.ascii_letters[3:] if c != 'z']
        self.expr = (
            f'bz, zca, abc{"".join(letters[:self.l])} -> '
            f'bc{"".join(letters[:self.l])}'
        )

    def forward(self, t: torch.Tensor, node_attrs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_lora:
            delta_w = (self.lora_B.T @ self.lora_A.T) * self.scaling
            W = self.weights + delta_w.unsqueeze(0)
        else:
            W = self.weights
        W = W * self.alpha
        t = torch.einsum(self.expr, node_attrs, W, t)
        if self.bias is not None:
            b = torch.einsum('bz, zC -> bC', node_attrs, self.bias)
            t = t + b
        return t

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  in_dim      = {self.in_dim},\n"
            f"  out_dim     = {self.out_dim},\n"
            f"  bias        = {self.bias is not None},\n"
            f"  alpha       = {self.alpha:.2f},\n"
            f"  lora_r      = {self.lora_r},\n"
            f"  lora_alpha  = {self.lora_alpha},\n"
            f")"
        )
    

# (elemenet-aware, channel-wise)
LinearDict = {
    (False, False): Linear,
    (True, False): ElementLinear,
    (False, True): CWLinear,
    (True, True): ElementCWLinear,
}

LoRALinearDict = {
    (False, False): LoRALinear,
    (True, False): LoRAElementLinear,
    (False, True): LoRACWLinear,
    (True, True): LoRAElementCWLinear,
}