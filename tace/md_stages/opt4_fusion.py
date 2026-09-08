"""TACE-owned interaction rejector and ACE product candidates."""
import copy
import torch
from torch import nn
from md_benchmark.opt4_fx import CheckedRegion, install_tp_regions
from md_benchmark.opt4_registry import FusionSetupError, record


@torch.library.custom_op("tace_opt4::native_silu", mutates_args=())
def native_silu(x: torch.Tensor) -> torch.Tensor:
    """Opaque ATen boundary: preserve native activation rounding, not a fusion."""
    return torch.nn.functional.silu(x)


@native_silu.register_fake
def _native_silu_fake(x):
    return torch.empty_like(x)


@torch.library.custom_op("tace_opt4::native_silu_vjp", mutates_args=())
def native_silu_vjp(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.silu_backward.default(g, x)


@native_silu_vjp.register_fake
def _native_silu_vjp_fake(g, x):
    return torch.empty_like(x)


def _silu_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])


def _silu_backward(ctx, g):
    return native_silu_vjp(g, ctx.saved_tensors[0])


native_silu.register_autograd(_silu_backward, setup_context=_silu_setup)


class NativeSiLU(nn.Module):
    def forward(self, x):
        return native_silu(x)


class RadialCutoff(nn.Module):
    def __init__(self, edge_info):
        super().__init__()
        self.edge_info = edge_info

    def forward(self, features, cutoff):
        return self.edge_info(features) * cutoff


def build_radial_cutoff(edge_info, detail):
    # Only this candidate's private copy is changed; Opt3/model/checkpoint stay
    # untouched. Native calls execute at capture; replay uses recorded kernels.
    candidate = copy.deepcopy(edge_info)
    count = 0
    for path, child in list(candidate.named_modules()):
        if type(child) is nn.SiLU:
            if child.inplace or not path:
                raise FusionSetupError("TACE radial candidate requires out-of-place child SiLU")
            candidate.set_submodule(path, NativeSiLU())
            count += 1
    if not count:
        raise FusionSetupError("TACE radial candidate matched no native SiLU boundaries")
    detail.update(native_silu_boundaries=count, native_silu_is_fusion=False,
                  activation_policy="opaque-native-aten-forward-and-vjp")
    region = CheckedRegion(RadialCutoff(candidate), detail, backward_policy="aten")
    # Validate against the ORIGINAL native MLP, not the already adapted module.
    region.reference = RadialCutoff(edge_info)
    return region


def install(model, passes, report):
    install_tp_regions(model, passes, report,
        lambda path: "rejector.tp" in path or ".aces." in path or ".tp." in path,
        backward_policy="aten")
    if "radial_cutoff" in passes:
        modules = []
        for path, module in list(model.named_modules()):
            if type(module).__name__ == "CgtpInteraction" and hasattr(module, "edge_info"):
                detail = {"module": path, "validated_shapes": 0,"benchmark_requested":report.get("benchmark_boundaries",False)}
                module._opt4_radial_cutoff = build_radial_cutoff(module.edge_info, detail)
                modules.append(detail)
        record(report, "radial_cutoff", len(modules), "inductor-triton-epilogue", modules=modules,
               gemm="existing GEMM, no autotuned GEMM", backward_policy="aten", fusion_scope="forward-only",
               excluded_from_fusion=["SiLU forward", "SiLU VJP", "GEMM"],
               candidate_status="native-SiLU-boundary; CUDA correctness and fusion benefit pending")
