################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch


ACTIVATION = {
    None: torch.nn.Identity,
    "none": torch.nn.Identity,
    "None": torch.nn.Identity,
    "null": torch.nn.Identity,
    "identity": torch.nn.Identity,
    "relu": torch.nn.ReLU,
    "leaky_relu": torch.nn.LeakyReLU,
    "prelu": torch.nn.PReLU,
    "elu": torch.nn.ELU,
    "selu": torch.nn.SELU,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,
    "swish": torch.nn.SiLU, 
    "mish": torch.nn.Mish,
    "softplus": torch.nn.Softplus,
    "softsign": torch.nn.Softsign,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "hardtanh": torch.nn.Hardtanh,
    "hardswish": torch.nn.Hardswish,
    "hardsigmoid": torch.nn.Hardsigmoid,
    "tanhshrink": torch.nn.Tanhshrink,
}
