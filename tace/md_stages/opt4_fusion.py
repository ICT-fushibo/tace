"""FastEq-inspired complete CgtpInteraction rejector boundary.

Algorithmic adaptation of FastEq commit 40ba40e72bee769d74a869bb4a4ba820ee1c55c0
(MIT); the integration repository carries the complete third-party notice.
"""
from __future__ import annotations

import torch
from torch import nn

from md_benchmark.opt4_fx import CheckedRegion, assert_associative_sum_close
from md_benchmark.opt4_registry import FusionSetupError, fixed_csr_layout, record


class _Uniform1DRejector(nn.Module):
    def __init__(self, tp, edge_rows, rows, max_terms):
        super().__init__()
        object.__setattr__(self, "_tp", tp)
        self.register_buffer("edge_rows", edge_rows, persistent=False)
        self.rows = int(rows)
        self.max_terms = int(max_terms)

    def set_layout(self, edge_rows, rows, max_terms):
        self.edge_rows = edge_rows
        self.rows = int(rows)
        self.max_terms = int(max_terms)

    def forward(self, x, y, weight, edge_src):
        edge_features = self._tp(x.index_select(0, edge_src), y, weight)
        out = edge_features.new_zeros((self.rows, edge_features.shape[-1]))
        out.index_add_(0, self.edge_rows, edge_features)
        return out

    def validate_output(self, actual, expected, args, output_index):
        """Validate legal destination-reduction reorderings against float64."""

        if output_index != 0:
            return
        x, y, weight, edge_src = args
        with torch.no_grad():
            edge_features = self._tp(x.index_select(0, edge_src), y, weight)
        assert_associative_sum_close(
            actual,
            expected,
            edge_features,
            self.edge_rows,
            self.rows,
            self.max_terms,
        )


def _layout(options, parameter):
    row_ptr, edge_rows, max_row = fixed_csr_layout(options, parameter)
    return edge_rows, int(row_ptr.shape[0] - 1), int(max_row)


def refresh(model, options) -> None:
    edge_rows, rows, max_terms = _layout(options, next(model.parameters()))
    for module in model.modules():
        region = getattr(module, "_opt4_fasteq_uniform1d", None)
        if isinstance(region, CheckedRegion):
            module._opt4_edge_capacity = int(edge_rows.numel())
            region.reference.set_layout(edge_rows, rows, max_terms)
            region.signatures.clear()


def install(model, passes, report, options):
    if "fasteq_uniform1d_rejector" not in passes:
        return
    edge_rows, rows, max_terms = _layout(options, next(model.parameters()))
    modules = []
    for path, module in list(model.named_modules()):
        if type(module).__name__ != "O3ScatterTensorProduct" or "rejector" not in path:
            continue
        if hasattr(module, "fused_tp"):
            raise FusionSetupError(
                "FastEq rejector owns the full boundary and cannot wrap another fused TP"
            )
        detail = {
            "module": path,
            "validated_shapes": 0,
            "benchmark_requested": report.get("benchmark_boundaries", False),
        }
        boundary = _Uniform1DRejector(module.tp, edge_rows, rows, max_terms)
        module._opt4_fasteq_uniform1d = CheckedRegion(
            boundary,
            detail,
            output_validator=boundary.validate_output,
            vjp_atol=5e-6,
        )
        module._opt4_edge_capacity = int(edge_rows.numel())
        modules.append(detail)
    record(
        report,
        "fasteq_uniform1d_rejector",
        len(modules),
        "inductor-triton-full-boundary-aot-vjp",
        modules=modules,
        fused_boundaries=[
            "source-gather",
            "explicit-instruction-tp",
            "destination-reduce",
        ],
        ace_products="unchanged",
        gemm="original-e3nn",
        backward="aot-compiled-complete-input-vjp",
        replay_runtime_compile=False,
    )
