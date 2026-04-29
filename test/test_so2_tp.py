import math
import torch
torch.set_default_dtype(torch.float64)
torch.set_printoptions(sci_mode=False, precision=8)
torch.manual_seed(0)

from tace.models.so2.utils import num_uuu_so2_components, rotate_uuu_so2_features
from tace.models.so2 import SO2TensorProduct

B = 4
mmax = 2
lmax = 3
C = 1

tp = SO2TensorProduct(
    mmax=mmax,
    lmax=lmax,
    num_channels=C,
    # m1m2='<='
)

ncomp = num_uuu_so2_components(lmax, mmax)

x = torch.randn(B, ncomp, C)
y = torch.randn(B, ncomp, C)

theta = 0.731
Rx = rotate_uuu_so2_features(
    x,
    theta,
    lmax,
    mmax,
    C,
)
Ry = rotate_uuu_so2_features(
    y,
    theta,
    lmax,
    mmax,
    C,
)

# TP(Rx,Ry)
rtp_in = tp(Rx, Ry)

# R TP(x,y)
tp = tp(x, y)
rtp_out = rotate_uuu_so2_features(
    tp,
    theta,
    lmax,
    mmax,
    C,
)


err = (rtp_in - rtp_out)


print("===================================")
# print(err)
print("equivariance error")
print(err.abs().max())
print("===================================")
