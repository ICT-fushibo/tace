import math
from functools import lru_cache

import torch
torch.set_default_dtype(torch.float64)
NORMALIZATION = "component"
ELL = 2       # edge     
FEATURE_L = 1 # node
OUT_L = 1
NEIGHBORS = 8
SAMPLES = 512
NUM_EDGES = SAMPLES * NEIGHBORS
SEED = 42
TOL = 5e-7
from e3nn import o3
from sympy import S
from sympy.physics import wigner

def solid_harmonic(l: int, x: torch.Tensor) -> torch.Tensor:
    return o3.spherical_harmonics(
        l,
        x,
        normalize=False,
        normalization=NORMALIZATION,
    )

def cgtp(
    x: torch.Tensor,
    y: torch.Tensor,
    l1: int,
    l2: int,
    lout: int,
) -> torch.Tensor:
    w3j = o3.wigner_3j(l1, l2, lout)
    return math.sqrt(2 * lout + 1) * torch.einsum("...i,...j,ijk->...k", x, y, w3j)


def allowed_intermediates(feature_l: int, v: int, u: int, out_l: int) -> list[int]:
    lo = abs(feature_l - v)
    hi = feature_l + v
    return [k for k in range(lo, hi + 1) if abs(u - k) <= out_l <= u + k]


def wigner_6j(
    l1: int, l2: int, l1l2: int,
    l3: int, L: int, l23: int,
) -> float:
    return (
        math.comb(l23, l2)
        * ((-1) ** l2)
        * ((-1) ** (l1 + l2 + l3 + L))
        * math.sqrt((2 * l1l2 + 1) * (2 * l23 + 1))
        * float(
            wigner.wigner_6j(
                S(l1), S(l2), S(l1l2),
                S(l3), S(L), S(l23),
            )
        )
    )


def solid_path_scales(edge_l: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7919 + edge_l)
    ri = torch.randn(512, 3, generator=generator)
    rj = torch.randn(512, 3, generator=generator)
    lhs = solid_harmonic(edge_l, ri - rj)
    terms = []
    for u in range(edge_l + 1):
        v = edge_l - u
        sign = -1.0 if v % 2 else 1.0
        coeff = sign * math.comb(edge_l, u)
        terms.append(
            coeff
            * cgtp(
                solid_harmonic(u, ri),
                solid_harmonic(v, rj),
                u,
                v,
                edge_l,
            )
        )

    basis = torch.stack(terms, dim=-1)
    return torch.linalg.lstsq(
        basis.reshape(-1, edge_l + 1),
        lhs.reshape(-1),
    ).solution.detach()


def sample_batch(
    batch: int,
    neighbors: int,
    feature_l: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ri = torch.randn(batch, 3, generator=generator)
    rj = torch.randn(batch, neighbors, 3, generator=generator)
    h = torch.randn(
        batch,
        neighbors,
        2 * feature_l + 1,
        generator=generator,
    )
    alpha = torch.randn(batch, neighbors, 1, generator=generator)
    return ri, rj, h, alpha


def edge_convolution(
    ri: torch.Tensor,
    rj: torch.Tensor,
    h: torch.Tensor,
    alpha: torch.Tensor,
    ell: int,
    feature_l: int,
    out_l: int,
) -> torch.Tensor:
    batch, neighbors, _ = rj.shape
    rel = ri[:, None, :] - rj # r_ij
    r_l = solid_harmonic(ell, rel.reshape(batch * neighbors, 3))
    h_flat = h.reshape(batch * neighbors, 2 * feature_l + 1)
    msg = cgtp(
        h_flat,
        r_l,
        feature_l,
        ell,
        out_l,
    ).reshape(batch, neighbors, 2 * out_l + 1)
    return (alpha * msg).sum(dim=1)


def recoupled_convolution(
    ri: torch.Tensor,
    rj: torch.Tensor,
    h: torch.Tensor,
    alpha: torch.Tensor,
    l23: int,   # solid harmonics
    l1: int,    # node_feats
    L: int,     # out
) -> tuple[torch.Tensor, list[tuple[str, float]]]:
    batch, neighbors, _ = rj.shape
    out = torch.zeros(batch, 2 * L + 1)
    paths: list[tuple[str, float]] = []
    scales = solid_path_scales(l23).to(device=ri.device, dtype=ri.dtype)

    # === h_i, r_i => expand => h_ij, r_IJ ===
    h_ij = h.reshape(NUM_EDGES, 2 * l1 + 1) # [edge, 2l1+1]
    r_ij = rj.reshape(NUM_EDGES, 3)

    for l3 in range(l23 + 1):
        l2 = l23 - l3
        r_u_i = solid_harmonic(l3, ri)      # [node]
        r_v_j = solid_harmonic(l2, r_ij) # [edge]
        for l12 in allowed_intermediates(l1, l2, l3, L):
            m_ij = cgtp(
                h_ij,
                r_v_j,
                l1,
                l2,
                l12,
            ).reshape(batch, neighbors, 2 * l12 + 1)
            m_i = (alpha * m_ij).sum(dim=1)

            outer = cgtp(
                m_i, r_u_i,
                l12, l3, L,
            )
            weight = float(scales[l3]) * wigner_6j(
                l1, l2, l12,
                l3, L,  l23,
            )
            out = out + weight * outer
            paths.append((f"l3={l3}, l2={l2}, l12={l12}", weight))

    return out, paths


def relative_rms(x: torch.Tensor, ref: torch.Tensor) -> float:
    num = (x - ref).square().mean().sqrt()
    den = ref.square().mean().sqrt().clamp_min(torch.finfo(ref.dtype).eps)
    return (num / den).item()


def main() -> None:
    generator = torch.Generator().manual_seed(SEED)

    batch = sample_batch(SAMPLES, NEIGHBORS, FEATURE_L, generator)
    lhs = edge_convolution(*batch, ELL, FEATURE_L, OUT_L)
    rhs, paths = recoupled_convolution(*batch, ELL, FEATURE_L, OUT_L)
    rel = relative_rms(rhs, lhs)
    max_abs = (rhs - lhs).abs().max().item()


    print(
        f"ell={ELL}, feature_l={FEATURE_L}, out_l={OUT_L}, "
        f"neighbors={NEIGHBORS}, samples={SAMPLES}"
    )
    print()
    print("solid path scales from Theorem 3.2 / e3nn convention:")
    print("  [" + ", ".join(f"{x:.12g}" for x in solid_path_scales(ELL).tolist()) + "]")
    print()
    print(f"number of analytic Wigner-6j paths: {len(paths)}")
    for label, weight in paths:
        print(f"  {label:16s} weight={weight:.12g}")
    print()
    print(f"relative RMS : {rel:.6e}")
    print(f"max abs      : {max_abs:.6e}")

    if rel > TOL:
        raise SystemExit(f"FAIL: relative RMS {rel:.6e} > tol {TOL:.6e}")
    print(f"PASS: relative RMS is below {TOL:g}.")


if __name__ == "__main__":
    main()
