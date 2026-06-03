import argparse
import math


import torch
torch.set_default_dtype(torch.float64)
generator = torch.Generator().manual_seed(42)
lmax = 6
normalization = "component" # ["component", "norm", "integral"]
convention = "i_minus_j" # ["i_minus_j", "j_minus_i"]
batch = 512
from e3nn import o3

def solid_harmonic(l: int, x: torch.Tensor, normalization: str) -> torch.Tensor:
    return o3.spherical_harmonics(
        l,
        x,
        normalize=False,
        normalization=normalization,
    )

def binomial_terms(
    l: int,
    a: torch.Tensor,
    b: torch.Tensor,
    normalization: str,
) -> torch.Tensor:
    terms = []
    for u in range(l + 1):
        v = l - u
        sign = -1.0 if (l - u) % 2 else 1.0
        coeff = sign * math.comb(l, u)
        x_u = solid_harmonic(u, a, normalization)
        x_v = solid_harmonic(v, b, normalization)
        tp = torch.einsum("bi, bj, ijk -> bk", x_u, x_v, o3.wigner_3j(u, v, l))
        terms.append(coeff * tp)
    return torch.stack(terms, dim=-1)

def fit_path_constants(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    A = A.reshape(-1, A.shape[-1]) # [batch, 2l+1, path]
    b = b.reshape(-1)              # [batch, 2l+1]
    return torch.linalg.lstsq(A, b).solution

def sample_pair(
    n: int,
    generator: torch.Generator,
    convention: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ri = torch.randn(n, 3, generator=generator)
    rj = torch.randn(n, 3, generator=generator)
    if convention == "i_minus_j":
        return ri, rj, ri - rj
    if convention == "j_minus_i":
        return rj, ri, rj - ri
    raise ValueError(f"unknown convention: {convention}")


def main() -> None:
    print("l | path constants")
    print("-" * 100)
    for l in range(1, lmax + 1):
        a, b, rel = sample_pair(batch, generator, convention)
        lhs = solid_harmonic(l, rel, normalization)
        terms = binomial_terms(l, a, b, normalization)
        constants = fit_path_constants(terms, lhs)
        a_test, b_test, rel_test = sample_pair(batch, generator, convention)
        lhs_test = solid_harmonic(l, rel_test, normalization)
        terms_test = binomial_terms(l, a_test, b_test, normalization)
        fitted_rhs = (terms_test * constants).sum(dim=-1)
        all_ok = torch.allclose(fitted_rhs, lhs_test)
        max_abs_err = (fitted_rhs - lhs_test).abs().max()
        coeff_text = ", ".join(f"{c:.9g}" for c in constants.detach().cpu().tolist())
        print(f"{l:3d} | [{coeff_text}]")
        print(f"allclose = {all_ok}")
        print(f"max_abs_err = {max_abs_err.item():.6e}")
        print()
        
if __name__ == "__main__":
    main()
