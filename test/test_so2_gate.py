import math
import torch

torch.set_default_dtype(torch.float64)
torch.set_printoptions(sci_mode=False, precision=8)
torch.manual_seed(0)

from tace.models.so2.utils import num_uuu_so2_components, rotate_uuu_so2_features
from tace.models.so2 import SO2Gate


B = 8

lmax = 4
mmax = 3

C = 6

theta = 0.731

ncomp = num_uuu_so2_components(lmax, mmax)

gate = SO2Gate(
    mmax=mmax,
    lmax=lmax,
    num_channel=C,
    channel_wise=True,
)

x = torch.randn(
    B,
    ncomp,
    C,
)
g = torch.randn(
    B,
    (lmax+1) *(mmax+1),
    C,
)
print(x.shape)
print(g.shape)
Rx = rotate_uuu_so2_features(
    x,
    theta,
    lmax,
    mmax,
    C,
)

rx_in = gate(Rx, g)
x_out = gate(x, g)

rx_out = rotate_uuu_so2_features(
    x_out,
    theta,
    lmax,
    mmax,
    C,
)


abs_err = (rx_in - rx_out).abs().max()


print("========================================")
print("Equivariance error")
print("========================================")
print(f"abs error  : {abs_err}")
print("========================================")

