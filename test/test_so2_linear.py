import math
import torch

torch.set_default_dtype(torch.float64)
torch.set_printoptions(sci_mode=False, precision=8)
torch.manual_seed(0)

from tace.models.so2.utils import num_so2_components, rotate_so2_features
from tace.models.so2 import uvSO2Linear

B = 8

mmax = 2
lmax = 4

Cin = 5
Cout = 7

theta = 0.731

layer = uvSO2Linear(
    mmax=mmax,
    lmax=lmax,
    num_channel_in=Cin,
    num_channel_out=Cout,
)

ncomp = num_so2_components(lmax, mmax)

x = torch.randn(
    B,
    ncomp,
    Cin,
)

Rx = rotate_so2_features(
    x,
    theta,
    lmax,
    mmax,
    Cin,
)

rlinear_in = layer(Rx)
linear_out = layer(x)

rlinear_out = rotate_so2_features(
    linear_out,
    theta,
    lmax,
    mmax,
    Cout,
)


abs_err = (rlinear_in - rlinear_out).abs().max()


print("========================================")
print("Equivariance error")
print(f"abs error  : {abs_err.item():.3e}")
print("========================================")

