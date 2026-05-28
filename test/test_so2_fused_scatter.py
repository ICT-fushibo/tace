################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch

from tace.models._e3nn.fused import SO2ExternalLinear
from tace.models._triton.so2 import fused_so2_source_linear_scatter


def _reference(
    x,
    weight,
    src,
    dst,
    wigner,
    wigner_inv,
    linear,
    num_nodes,
):
    x_edge = torch.bmm(wigner, x[src])
    y_edge = linear(x_edge, weight)
    y_edge = torch.bmm(wigner_inv, y_edge)
    out = x.new_zeros(num_nodes, x.size(1), x.size(2))
    out.index_add_(0, dst, y_edge)
    return out


def _run(device="cpu", use_triton=False):
    torch.manual_seed(0)
    lmax = 3
    mmax = 2
    num_channel = 4
    num_nodes = 5
    num_edges = 7
    num_so3 = (lmax + 1) ** 2
    num_so2 = (lmax + 1) + sum(2 * (lmax + 1 - m) for m in range(1, mmax + 1))

    linear = SO2ExternalLinear(mmax, lmax, num_channel).to(device)
    x = torch.randn(num_nodes, num_so3, num_channel, device=device, requires_grad=True)
    weight = torch.randn(num_edges, linear.weight_numel, device=device, requires_grad=True)
    src = torch.tensor([0, 1, 2, 3, 4, 0, 1], device=device)
    dst = torch.tensor([1, 2, 3, 4, 0, 2, 3], device=device)
    wigner = torch.randn(num_edges, num_so2, num_so3, device=device, requires_grad=True)
    wigner_inv = torch.randn(num_edges, num_so3, num_so2, device=device, requires_grad=True)

    out_ref = _reference(x, weight, src, dst, wigner, wigner_inv, linear, num_nodes)
    out_fused = fused_so2_source_linear_scatter(
        x,
        weight,
        src,
        dst,
        wigner,
        wigner_inv,
        linear.path_out,
        linear.path_in,
        linear.path_weight,
        linear.path_scale,
        num_nodes,
        linear.num_weights,
        use_triton=use_triton,
    )
    torch.testing.assert_close(out_fused, out_ref, rtol=1e-5, atol=1e-5)

    grad = torch.randn_like(out_ref)
    ref_grads = torch.autograd.grad(out_ref, (x, weight, wigner, wigner_inv), grad)
    fused_grads = torch.autograd.grad(out_fused, (x, weight, wigner, wigner_inv), grad)
    for fused_grad, ref_grad in zip(fused_grads, ref_grads):
        torch.testing.assert_close(fused_grad, ref_grad, rtol=1e-5, atol=1e-5)


def test_so2_v4_fused_scatter_torch():
    _run(device="cpu", use_triton=False)


def test_so2_v4_fused_scatter_triton():
    _run(device="cuda", use_triton=True)


if __name__ == "__main__":
    test_so2_v4_fused_scatter_torch()
    test_so2_v4_fused_scatter_triton()

