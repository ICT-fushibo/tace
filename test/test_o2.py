import math

import torch


DTYPE = torch.float64
DEVICE = "cpu"
BATCH = 8
CHANNEL_IN = 5
CHANNEL_OUT = 4
M = 3
ANGLE = 0.731
SEED = 42
ATOL = 1e-10
RTOL = 1e-10


def so2_rotate(z: torch.Tensor, m: int, angle: float) -> torch.Tensor:
    c = math.cos(m * angle)
    s = math.sin(m * angle)
    zr = z[..., 0]
    zi = z[..., 1]
    return torch.stack((c * zr - s * zi, s * zr + c * zi), dim=-1)


def o2_reflect(z: torch.Tensor) -> torch.Tensor:
    zr = z[..., 0]
    zi = z[..., 1]
    return torch.stack((zr, -zi), dim=-1)


def complex_linear(z: torch.Tensor, wr: torch.Tensor, wi: torch.Tensor) -> torch.Tensor:
    zr = z[..., 0]
    zi = z[..., 1]
    yr = zr @ wr - zi @ wi
    yi = zr @ wi + zi @ wr
    return torch.stack((yr, yi), dim=-1)


def scalar_even_odd(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xr = x[..., 0]
    xi = x[..., 1]
    yr = y[..., 0]
    yi = y[..., 1]
    scalar_even = xr * yr + xi * yi
    scalar_odd = xr * yi - xi * yr
    return scalar_even, scalar_odd


def max_abs(x: torch.Tensor) -> float:
    return float(x.abs().max().item())


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    diff = max_abs(actual - expected)
    ok = torch.allclose(actual, expected, atol=ATOL, rtol=RTOL)
    print(f"{name:<45} max_abs={diff:.3e}  ok={ok}")
    if not ok:
        raise AssertionError(name)


def main() -> None:
    torch.manual_seed(SEED)

    z = torch.randn(BATCH, CHANNEL_IN, 2, device=DEVICE, dtype=DTYPE)
    y = torch.randn(BATCH, CHANNEL_IN, 2, device=DEVICE, dtype=DTYPE)

    wr = torch.randn(CHANNEL_IN, CHANNEL_OUT, device=DEVICE, dtype=DTYPE)
    wi = torch.randn(CHANNEL_IN, CHANNEL_OUT, device=DEVICE, dtype=DTYPE)
    zero_wi = torch.zeros_like(wi)

    # SO(2): complex linear with a complex weight commutes with rotation.
    lhs = complex_linear(so2_rotate(z, M, ANGLE), wr, wi)
    rhs = so2_rotate(complex_linear(z, wr, wi), M, ANGLE)
    assert_close("SO2 complex linear, complex weight", lhs, rhs)

    # O(2): reflection acts as complex conjugation. A general complex weight does
    # not commute with conjugation.
    lhs = complex_linear(o2_reflect(z), wr, wi)
    rhs = o2_reflect(complex_linear(z, wr, wi))
    diff = max_abs(lhs - rhs)
    print(f"{'O2 complex linear, complex weight':<45} max_abs={diff:.3e}  ok={torch.allclose(lhs, rhs, atol=ATOL, rtol=RTOL)}")

    # O(2): real-valued complex linear, i.e. wi=0, commutes with reflection.
    # lhs = complex_linear(o2_reflect(z), wr, zero_wi)
    # rhs = o2_reflect(complex_linear(z, wr, zero_wi))
    lhs = complex_linear(o2_reflect(z), wr, zero_wi)
    rhs = o2_reflect(complex_linear(z, wr, zero_wi))
    assert_close("O2 complex linear, real weight", lhs, rhs)

    # m=0 even and odd scalars from m x m.
    se, so = scalar_even_odd(z, y)

    z_rot = so2_rotate(z, M, ANGLE)
    y_rot = so2_rotate(y, M, ANGLE)
    se_rot, so_rot = scalar_even_odd(z_rot, y_rot)
    assert_close("SO2 0e invariant", se_rot, se)
    assert_close("SO2 0o invariant", so_rot, so)

    z_ref = o2_reflect(z)
    y_ref = o2_reflect(y)
    se_ref, so_ref = scalar_even_odd(z_ref, y_ref)
    assert_close("O2 0e reflection-even", se_ref, se)
    assert_close("O2 0o reflection-odd", so_ref, -so)


if __name__ == "__main__":
    main()
