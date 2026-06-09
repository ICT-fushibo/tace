# import argparse
# import math
# import sys
# from pathlib import Path

# import matplotlib

# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import torch


# from tace.models.radial import j0SphericalBesselBasis, ZernikeBasis


# def evaluate_bases(
#     cutoff: float, nmax: int, 
# ) -> tuple[torch.Tensor, list[torch.Tensor], list[list[str]]]:
#     r = torch.linspace(0.0, cutoff, 1000).unsqueeze(-1)
#     dummy_attrs = torch.empty(0)
#     dummy_edges = torch.empty((2, 0), dtype=torch.long)
#     # zernike_2d_module = ZernikeBasis(nmax=nmax, cutoff=cutoff)
#     zernike_2d_module = ZernikeBasis(m=0, num_basis=nmax, cutoff=cutoff)
#     num_basis = zernike_2d_module.num_basis
#     j0_basis = j0SphericalBesselBasis(cutoff=cutoff, num_basis=num_basis)
#     j0 = j0_basis(r, dummy_attrs, dummy_edges)
#     zernike_2d = zernike_2d_module(r, dummy_attrs, dummy_edges)
#     labels = [
#         [f"q={q + 1}" for q in range(num_basis)],
#         [f"(n,m)=({int(n)},{int(m)})" for n, m in zernike_2d_module.nm],
#     ]
#     return r.squeeze(-1), [j0, zernike_2d], labels


# def normalized_shape(values: torch.Tensor) -> torch.Tensor:
#     # return values / values.abs().amax(dim=0, keepdim=True).clamp_min(1.0e-12)
#     return values

# def plot_comparison(nmax: int, num_points: int):
#     r, basis_values, labels = evaluate_bases(nmax, num_points)
#     titles = [
#         r"$j_0$: $\sqrt{2}\sin(q\pi r)/r$",
#         rf"2D Zernike: all $R_n^m(r)$ with $n \leq {nmax}$",
#     ]

#     fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
#     for column, (values, curve_labels, title) in enumerate(
#         zip(basis_values, labels, titles)
#     ):
#         for row, plotted in enumerate((values, normalized_shape(values))):
#             ax = axes[row, column]
#             for channel, label in enumerate(curve_labels):
#                 ax.plot(r, plotted[:, channel], linewidth=1.7, label=label)
#             ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
#             ax.set_xlim(0.0, 1.0)
#             ax.grid(alpha=0.25)
#             ax.set_title(title)
#             ax.legend(fontsize=8, ncol=2)

#     axes[0, 0].set_ylabel("Original value")
#     axes[1, 0].set_ylabel("Per-channel max-normalized value")
#     for ax in axes[1]:
#         ax.set_xlabel(r"Normalized radius $r/r_c$")

#     fig.suptitle("Radial basis comparison on [0, 1]", fontsize=15)
#     fig.tight_layout()
#     fig.savefig("zernike.pdf", bbox_inches="tight")
#     plt.close(fig)


# if __name__ == "__main__":
#     nnmax = 8
#     cutoff = 6.0
#     plot_comparison(cutoff, nnmax)
