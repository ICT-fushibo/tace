import torch

torch.set_printoptions(precision=6, sci_mode=True)
torch.set_default_dtype(torch.float64)

from tace.models.radial import C2PolynomialCutoff, C3PolynomialCutoff


def nth_derivative(y, x, n):
    """
    Compute n-th derivative of scalar y w.r.t scalar x
    """
    for _ in range(n):
        y = torch.autograd.grad(
            y,
            x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
        )[0]
    return y


def check_derivatives(
    cutoff_class,
    p=4,
    max_order=5,
    r_max=1.0,
    eps=1e-8,
):
    """
    Check derivatives near cutoff from inside: r = r_c - eps
    """
    r = torch.tensor(r_max - eps, requires_grad=True)

    # call static envelope
    y = cutoff_class.calculate_envelope(r, torch.tensor(r_max), torch.tensor(p))

    print(f"\n{cutoff_class.__name__}: p={p}")
    print(f"value at r≈cutoff: {y.item():.6e}")

    for n in range(1, max_order + 1):
        dn = nth_derivative(y, r, n)
        print(f"{n}-th derivative at r≈cutoff: {dn.item():.6e}")


def check_both(p=4, max_order=5, eps=1e-8):
    """
    Compare C2 and C3 behavior near cutoff
    """
    print("=" * 50)
    print(f"Testing near cutoff with p={p}, eps={eps}")

    check_derivatives(
        C2PolynomialCutoff,
        p=p,
        max_order=max_order,
        eps=eps,
    )

    check_derivatives(
        C3PolynomialCutoff,
        p=p,
        max_order=max_order,
        eps=eps,
    )


if __name__ == "__main__":
    check_both(p=4, max_order=6, eps=1e-10)
