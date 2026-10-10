"""Experimental disjoint TP input views with a direct packed split VJP.

Generated e3nn TP slices disjoint feature, harmonic and weight blocks using
independent getitems. Their backward can materialize many full-width zeros.
An exact, exhaustive split has the same forward views and a single packed
backward. No TP arithmetic, GEMM, precision or reduction order is changed.
Not installed or selected by default; only the diagnostic driver uses this.
"""
import copy
import operator

import torch


def rewrite_disjoint_slices(tp):
    candidate = copy.deepcopy(tp)
    gm = candidate._compiled_main_left_right
    if not isinstance(gm, torch.fx.GraphModule):
        raise TypeError('TP partition VJP requires an FX-generated TP')
    groups = {}
    for node in gm.graph.nodes:
        if node.op != 'call_function' or node.target is not operator.getitem:
            continue
        source, index = node.args
        if not isinstance(index, tuple) or len(index) != 2 or index[0] != slice(None):
            continue
        part = index[1]
        if (not isinstance(part, slice) or type(part.start) is not int
                or type(part.stop) is not int or part.step not in (None, 1)):
            continue
        if (not isinstance(source, torch.fx.Node) or source.op != 'call_method'
                or source.target != 'reshape' or len(source.args) != 3
                or source.args[1] != -1 or type(source.args[2]) is not int):
            continue
        groups.setdefault(source, []).append((part.start, part.stop, node))
    replaced = []
    for source, parts in groups.items():
        ordered = sorted(parts, key=lambda part: part[0])
        end = 0
        valid = len(ordered) > 1
        for start, stop, _ in ordered:
            valid = valid and start == end and stop > start
            end = stop
        if not valid or end != source.args[2]:
            continue
        sizes = tuple(stop-start for start, stop, _ in ordered)
        # Use graph order, not channel order, for a dominating definition.
        with gm.graph.inserting_before(parts[0][2]):
            split = gm.graph.call_function(torch.split, args=(source, sizes), kwargs={'dim': 1})
        for index, (_, _, node) in enumerate(ordered):
            node.args = (split, index)
        replaced.append({'source':source.name, 'width':end, 'sizes':list(sizes)})
    gm.graph.lint()
    gm.recompile()
    return candidate, replaced
