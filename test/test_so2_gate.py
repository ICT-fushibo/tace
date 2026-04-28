import math
import torch

torch.set_default_dtype(torch.float64)
torch.set_printoptions(sci_mode=False, precision=8)
torch.manual_seed(0)

def num_so2_components(
    lmax: int,
    mmax: int,
):
    total = lmax + 1
    for m in range(1, mmax + 1):
        total += 2 * (lmax + 1 - m)
    return total


def rotate_real_irrep(
    x: torch.Tensor,
    theta: float,
    m: int,
):
    """
    x:
        [B, 2, n, C]
    """

    c = math.cos(m * theta)
    s = math.sin(m * theta)

    xr = x[:, 0]
    xi = x[:, 1]

    yr = c * xr - s * xi
    yi = s * xr + c * xi

    out = torch.stack([yr, yi], dim=1)

    return out


def rotate_so2_features(
    x: torch.Tensor,
    theta: float,
    lmax: int,
    mmax: int,
    num_channels: int,
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




B = 8

lmax = 4
mmax = 3

C = 6

theta = 0.731

from tace.models.so2 import SO2Gate, SO2Linear

ncomp = num_so2_components(lmax, mmax)



linear = SO2Linear(
    mmax=mmax,
    lmax=lmax,
    num_in_channels=C,   
    num_out_channels=C,  
    extra_m0_out_channels=sum(lmax+1-m for m in range(mmax+1))*C,
)

gate = SO2Gate(
    mmax=mmax,
    lmax=lmax,
    num_channel=C,
)


x = torch.randn(
    B,
    ncomp,
    C,
)
x, g = linear(x)

Rx = rotate_so2_features(
    x,
    theta,
    lmax,
    mmax,
    C,
)

rx_in = gate(Rx, g)
x_out = gate(x, g)

rx_out = rotate_so2_features(
    x_out,
    theta,
    lmax,
    mmax,
    C,
)


abs_err = (rx_in - rx_out).abs().max()

print()
print("========================================")
print("SO2Gate Equivariance Test")
print("========================================")
print(f"abs error  : {abs_err}")
print("========================================")
print()
