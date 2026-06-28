import torch
from e3nn import o3
from tace.models._e3nn.edge_prod import O3ASymmetricContraction 


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
SEED = 2026

BATCH = 4096
CHANNELS = 32
CORRELATION = 3
IRREPS = o3.Irreps("0e+1o+2e+3o")


def make_input() -> torch.Tensor:
    return torch.randn(
        BATCH,
        IRREPS.dim,
        CHANNELS,
        device=DEVICE,
        dtype=DTYPE,
    )


def variance(x: torch.Tensor) -> float:
    return x.var(unbiased=False).item()


def per_irrep_variance(x: torch.Tensor) -> list[tuple[str, float]]:
    out = []
    offset = 0
    for mul, ir in IRREPS:
        for _ in range(mul):
            block = x[:, offset : offset + ir.dim]
            out.append((str(ir), variance(block)))
            offset += ir.dim
    return out


def print_case(*, independent_orders: bool) -> None:
    torch.manual_seed(SEED)
    module = O3ASymmetricContraction(
        irreps_in=IRREPS,
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

    per_ir = ", ".join(
        f"{ir}:{var:.3f}" for ir, var in per_irrep_variance(y)
    )
    print(
        f"independent={str(independent_orders):<5s} "
        f"in_var={input_var:.3f} "
        f"out_var={variance(y):.3f} "
        f"per_irrep=[{per_ir}]"
    )
    print(f"num_paths={module.order_num_paths}")


def main() -> None:
    print(f"device={DEVICE} dtype={DTYPE} seed={SEED}")
    print(
        f"B={BATCH} irreps={IRREPS} dim={IRREPS.dim} "
        f"channels={CHANNELS} correlation={CORRELATION}"
    )
    print()

    for independent_orders in (True, False):
        print_case(independent_orders=independent_orders)


if __name__ == "__main__":
    main()
