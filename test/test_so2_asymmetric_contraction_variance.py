import torch
from tace.models._e3nn.edge_prod import SO2ASymmetricContraction  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
SEED = 2026

BATCH = 4096
MMAX = 3
LMAX = 3
CHANNELS = 32
CORRELATION = 3


def num_components(mmax: int, lmax: int) -> int:
    return (lmax + 1) * (1 + 2 * mmax)


def make_input() -> torch.Tensor:
    shape = (BATCH, num_components(MMAX, LMAX), CHANNELS)
    return torch.randn(shape, device=DEVICE, dtype=DTYPE)


def variance(x: torch.Tensor) -> float:
    return x.var(unbiased=False).item()


def per_m_variance(x: torch.Tensor) -> list[float]:
    out = []
    offset = 0
    n = LMAX + 1
    out.append(variance(x[:, offset : offset + n]))
    offset += n
    for _m in range(1, MMAX + 1):
        out.append(variance(x[:, offset : offset + 2 * n]))
        offset += 2 * n
    return out


def print_case(
    *,
    independent_orders: bool,
) -> None:
    torch.manual_seed(SEED)
    module = SO2ASymmetricContraction(
        mmax=MMAX,
        lmax=LMAX,
        num_channels=CHANNELS,
        correlation=CORRELATION,
        internal_weights=True,
    ).to(device=DEVICE, dtype=DTYPE)

    if independent_orders:
        x = [make_input() for _ in range(CORRELATION)]
        input_var = sum(variance(xi) for xi in x) / len(x)
    else:
        x0 = make_input()
        x = x0
        input_var = variance(x0)

    with torch.no_grad():
        y = module(x)

    per_m = ", ".join(f"{v:.3f}" for v in per_m_variance(y))
    print(
        f"independent={str(independent_orders):<5s} "
        f"in_var={input_var:.3f} "
        f"out_var={variance(y):.3f} "
        f"per_m=[{per_m}]"
    )


def main() -> None:
    print(f"device={DEVICE} dtype={DTYPE} seed={SEED}")
    print(
        f"B={BATCH} mmax={MMAX} lmax={LMAX} "
        f"channels={CHANNELS} correlation={CORRELATION}"
    )
    print()

    for independent_orders in (True, False):
        print_case(independent_orders=independent_orders)


if __name__ == "__main__":
    main()
