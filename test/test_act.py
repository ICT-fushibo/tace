import math
import torch
from e3nn.nn import Activation

silu = torch.nn.SiLU()
e3nn_silu = Activation("16x0e", [silu])
x = torch.randn(1, 16).to(torch.float64)

print(e3nn_silu(x) / silu(x))

from e3nn.o3 import TensorProduct

def moment(f, n, dtype=None, device=None):
    r"""
    compute n th moment
    <f(z)^n> for z normal
    """
    gen = torch.Generator(device=device).manual_seed(0)
    z = torch.randn(1_000_000, generator=gen, dtype=torch.float64, device=device)
    return f(z).pow(n).mean()

cst = moment(silu, 2, dtype=torch.float64).pow(-0.5).item()

if abs(cst - 1) < 1e-4:
    cst = 1.0
else:
    cst = cst

print(cst)