import math
import torch

torch.set_default_dtype(torch.float64)
torch.set_printoptions(sci_mode=False, precision=8)
torch.manual_seed(0)

def rot(theta, m):
    c = math.cos(m * theta)
    s = math.sin(m * theta)
    R = torch.tensor([
        [c, -s],
        [s,  c],
    ])
    return R

def complex_mul(x, y):
    a, b = x
    c, d = y
    return torch.tensor([
        a*c - b*d,
        a*d + b*c,
    ])

def complex_mul_conj(x, y):
    a, b = x
    c, d = y
    return torch.tensor([
        a*c + b*d,
        b*c - a*d,
    ])



theta = 0.73

m1 = 1
m2 = 2

R1 = rot(theta, m1)
R2 = rot(theta, m2)

R3_sum = rot(theta, m1+m2)
R3_diff = rot(theta, m1-m2)

x = torch.randn(2)
y = torch.randn(2)
Rx = R1 @ x
Ry = R2 @ y


z1_sum = R3_sum @ complex_mul(x, y) 
z1_diff = R3_diff @ complex_mul_conj(x, y) 

z2_sum = complex_mul(Rx, Ry)
z2_diff = complex_mul_conj(Rx, Ry)

print("sum")
print((z1_sum - z2_sum))


print("diff")
print((z1_diff - z2_diff))


# print(complex_mul_conj(x, y))
# print(complex_mul_conj(y, x))

# z1_diff_inv = R3_diff @ complex_mul_conj(y, x) 
# z2_diff_inv = complex_mul_conj(Rx, Ry)

# print('diff_inv')
# print((z1_diff_inv - z2_diff_inv ))




def num_components(lmax, mmax):
    total = lmax + 1
    for m in range(1, mmax + 1):
        total += 2 * (lmax + 1 - m)
    return total


def rotate_real_irrep(x, theta, m):
    """
    x:
        [B, 2, n, C], x_r, x_i

    rotation:
        [ cos -sin ]
        [ sin  cos ]
    """
    c = math.cos(m * theta)
    s = math.sin(m * theta)
    xr = x[:, 0]
    xi = x[:, 1]
    yr = c * xr - s * xi
    yi = s * xr + c * xi
    y = torch.stack([yr, yi], dim=1)

    return y


def rotate_so2_features(
    x,
    theta,
    lmax,
    mmax,
    num_channels,
):
    """
    x:
        [B, num_components, C]
    """
    B = x.shape[0]
    outputs = []
    offset = 0

    # m = 0
    n0 = lmax + 1
    x0 = x[:, offset:offset+n0]
    outputs.append(x0)
    offset += n0

    # m > 0
    for m in range(1, mmax + 1):
        n = lmax + 1 - m
        xm = x[:, offset:offset+2*n]
        xm = xm.view(B, 2, n, num_channels)
        xm = rotate_real_irrep(
            xm,
            theta,
            m,
        )
        xm = xm.reshape(B, 2*n, num_channels)
        outputs.append(xm)
        offset += 2 * n
    out = torch.cat(outputs, dim=1)
    return out




B = 4
mmax = 3
lmax = 4
C = 1
from tace.models.so2 import SO2TensorProduct

tp = SO2TensorProduct(
    mmax=mmax,
    lmax=lmax,
    num_channels=C,
    # m1m2='<='
)

ncomp = num_components(lmax, mmax)

x = torch.randn(B, ncomp, C)
y = torch.randn(B, ncomp, C)

theta = 0.731
Rx = rotate_so2_features(
    x,
    theta,
    lmax,
    mmax,
    C,
)
Ry = rotate_so2_features(
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
rtp_out = rotate_so2_features(
    tp,
    theta,
    lmax,
    mmax,
    C,
)


err = (rtp_in - rtp_out)

print()
print("===================================")
print(err)
print("equivariance error")
print(err.abs().max())
print()
print("===================================")
print()
