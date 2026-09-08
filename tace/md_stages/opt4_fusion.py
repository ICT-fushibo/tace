"""TACE-owned interaction rejector and ACE product candidates."""
from torch import nn
from md_benchmark.opt4_fx import CheckedRegion, install_tp_regions
from md_benchmark.opt4_registry import record


class RadialCutoff(nn.Module):
    def __init__(self, edge_info):
        super().__init__()
        self.edge_info = edge_info

    def forward(self, features, cutoff):
        return self.edge_info(features) * cutoff


def install(model, passes, report):
    install_tp_regions(model, passes, report,
        lambda path: "rejector.tp" in path or ".aces." in path or ".tp." in path)
    if "radial_cutoff" in passes:
        modules = []
        for path, module in list(model.named_modules()):
            if type(module).__name__ == "CgtpInteraction" and hasattr(module, "edge_info"):
                detail = {"module": path, "validated_shapes": 0,"benchmark_requested":report.get("benchmark_boundaries",False)}
                module._opt4_radial_cutoff = CheckedRegion(RadialCutoff(module.edge_info), detail)
                modules.append(detail)
        record(report, "radial_cutoff", len(modules), "inductor-triton-epilogue", modules=modules,
               gemm="original Linear, no autotuned GEMM")
