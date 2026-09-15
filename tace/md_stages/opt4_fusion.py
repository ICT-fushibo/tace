"""TACE Opt4: fixed-slot rejector TP destination reduction."""
from __future__ import annotations

from torch import nn

from md_benchmark.opt4_fx import CheckedRegion
from md_benchmark.opt4_ops import csr_segment_sum
from md_benchmark.opt4_registry import fixed_csr_layout, record


class _NativeScatter(nn.Module):
    def __init__(self, edge_rows, rows):
        super().__init__()
        self.register_buffer("edge_rows", edge_rows, persistent=False)
        self.rows = int(rows)

    def forward(self, values):
        out = values.new_zeros((self.rows, *values.shape[1:]))
        out.index_add_(0, self.edge_rows, values)
        return out


class _FixedCSR(nn.Module):
    def __init__(self, row_ptr, edge_rows, max_row):
        super().__init__()
        self.register_buffer("row_ptr", row_ptr, persistent=False)
        self.register_buffer("edge_rows", edge_rows, persistent=False)
        self.max_row = int(max_row)

    def set_layout(self, row_ptr, edge_rows, max_row):
        self.row_ptr = row_ptr
        self.edge_rows = edge_rows
        self.max_row = int(max_row)

    def forward(self, values):
        return csr_segment_sum(
            values.contiguous(), self.row_ptr, self.edge_rows, self.max_row
        )


def refresh(model, options):
    row_ptr, edge_rows, max_row = fixed_csr_layout(
        options, next(model.parameters())
    )
    for module in model.modules():
        region = getattr(module, "_opt4_rejector_csr", None)
        if isinstance(region, CheckedRegion):
            module._opt4_edge_capacity = int(edge_rows.numel())
            region.reference.edge_rows = edge_rows
            region.reference.rows = row_ptr.shape[0] - 1
            region.compiled.set_layout(row_ptr, edge_rows, max_row)
            region.signatures.clear()


def install(model, passes, report, options):
    if "rejector_tp_reduce_vjp" not in passes:
        return
    row_ptr, edge_rows, max_row = fixed_csr_layout(
        options, next(model.parameters())
    )
    modules = []
    for path, module in list(model.named_modules()):
        if type(module).__name__ != "O3ScatterTensorProduct" or "rejector" not in path:
            continue
        if hasattr(module, "fused_tp"):
            continue
        detail = {
            "module": path,
            "validated_shapes": 0,
            "benchmark_requested": report.get("benchmark_boundaries", False),
        }
        module._opt4_rejector_csr = CheckedRegion(
            _NativeScatter(edge_rows, row_ptr.shape[0] - 1),
            detail,
            _FixedCSR(row_ptr, edge_rows, max_row),
        )
        module._opt4_edge_capacity = int(edge_rows.numel())
        modules.append(detail)
    record(
        report,
        "rejector_tp_reduce_vjp",
        len(modules),
        "triton-fixed-csr-explicit-vjp",
        modules=modules,
        fused_boundaries=["rejector-tp-output", "destination-reduce"],
        tensor_product="native-e3nn-instructions-unchanged",
        ace_products="unchanged-first-candidate",
        backward_recomputes_reference=False,
        fusion_scope="forward-and-backward",
    )
