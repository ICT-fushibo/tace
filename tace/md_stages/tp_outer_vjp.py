"""MD-only shared TP outer-product VJP; forward/GEMM stay compiler-native.

The native e3nn expression is einsum('edb,eca->ecdab', y, x), with
one spherical-harmonic channel. One backward kernel reads dOuter once and
reduces both feature and spherical-harmonic derivatives, without atomics.
"""
import copy
import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _both_vjp(G, X, Y, DX, DY, C: tl.constexpr, A: tl.constexpr,
                  B: tl.constexpr, SX: tl.constexpr, SY: tl.constexpr,
                  SG: tl.constexpr, R: tl.constexpr, K: tl.constexpr):
        e = tl.program_id(0)
        r = tl.arange(0, R)
        b = tl.arange(0, K)
        c, a = r // A, r % A
        x = tl.load(X+e*SX[0]+c*SX[1]+a*SX[2], r<C*A, 0)
        y = tl.load(Y+e*SY[0]+b*SY[2], b<B, 0)
        g = tl.load(G+e*SG[0]+c[:,None]*SG[1]+a[:,None]*SG[3]+b[None,:]*SG[4],
                    (r[:,None]<C*A)&(b[None,:]<B), 0)
        dx = tl.sum(g*y[None,:], axis=1)
        dy = tl.sum(g*x[:,None], axis=0)
        tl.store(DX+e*C*A+r, dx, r<C*A)
        tl.store(DY+e*B+b, dy, b<B)


@torch.library.custom_op('tace_opt4::outer_vjp', mutates_args=(), device_types='cuda')
def _vjp(grad: Tensor, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    if triton is None:
        raise RuntimeError('installed Triton is required for TACE outer VJP')
    e,c,a=x.shape
    b=y.shape[-1]
    if y.shape!=(e,1,b) or grad.shape!=(e,c,1,a,b):
        raise ValueError('unsupported TP outer-product shape')
    if x.dtype not in (torch.float32,torch.float64) or y.dtype!=x.dtype or grad.dtype!=x.dtype:
        raise ValueError('TP outer VJP requires same-dtype FP32/FP64 tensors')
    dx=torch.empty_like(x,memory_format=torch.contiguous_format)
    dy=torch.empty_like(y,memory_format=torch.contiguous_format)
    _both_vjp[(e,)](grad,x,y,dx,dy,c,a,b,x.stride(),y.stride(),grad.stride(),
                    triton.next_power_of_2(c*a),triton.next_power_of_2(b),
                    num_warps=8,enable_fp_fusion=False)
    return dx,dy


@_vjp.register_fake
def _(grad,x,y):
    return (torch.empty_like(x,memory_format=torch.contiguous_format),
            torch.empty_like(y,memory_format=torch.contiguous_format))


class _Outer(torch.autograd.Function):
    @staticmethod
    def forward(ctx,y,x):
        ctx.save_for_backward(x,y)
        return torch.einsum('edb,eca->ecdab',y,x)

    @staticmethod
    def backward(ctx,grad):
        x,y=ctx.saved_tensors
        dx,dy=_vjp(grad,x,y)
        return dy,dx


def outer(y,x):
    # Dynamo/AOT sees the original forward expression; only backward is opaque.
    return _Outer.apply(y,x)


def rewrite_shared_outer(tp):
    """Clone only this model's TP and replace shared explicit outer nodes."""
    candidate=copy.deepcopy(tp)
    gm=candidate._compiled_main_left_right
    if not isinstance(gm,torch.fx.GraphModule):
        raise TypeError('TACE outer VJP requires the native FX TP graph')
    replaced=[]
    for node in gm.graph.nodes:
        if (node.op=='call_function' and node.target in (torch.einsum,torch.functional.einsum)
                and node.args[0]=='edb,eca->ecdab' and len(node.users)>1):
            replaced.append({'node':node.name,'users':len(node.users)})
            node.target=outer
            node.args=node.args[1:]
    gm.graph.lint()
    gm.recompile()
    return candidate,replaced
