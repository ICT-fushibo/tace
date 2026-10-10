"""Diagnostic shared-input Clebsch--Gordan GEMM grouping.

Use the same native tensordot/GEMM and coefficient values, concatenating
static CG output columns for instructions sharing one outer product. Its
backward uses one native GEMM instead of materializing and adding multiple
full outer-product gradients. No generic GEMM kernel is replaced.
"""
import copy
import operator

import torch


def _constant(graph, node):
    if not isinstance(node, torch.fx.Node):
        if type(node) in (int, float):
            return node
        raise ValueError('not a static coefficient')
    if node.op == 'get_attr':
        value = graph
        for name in node.target.split('.'):
            value = getattr(value, name)
        if not isinstance(value, torch.Tensor) or value.requires_grad:
            raise ValueError('CG grouping requires frozen coefficient buffers')
        return value
    if node.op == 'call_function' and node.target is operator.mul:
        return _constant(graph, node.args[0]) * _constant(graph, node.args[1])
    raise ValueError('not a static coefficient expression')


def rewrite_grouped_cg(tp):
    candidate = copy.deepcopy(tp)
    gm = candidate._compiled_main_left_right
    if not isinstance(gm, torch.fx.GraphModule):
        raise TypeError('CG grouping requires an FX-generated TP')
    groups = {}
    for node in gm.graph.nodes:
        if (node.op == 'call_function' and node.target is torch.tensordot
                and len(node.args) == 2 and node.kwargs.get('dims') == ((3, 4), (0, 1))
                and node.kwargs.get('out') is None):
            try:
                coefficient = _constant(gm, node.args[1])
            except ValueError:
                continue
            if isinstance(coefficient, torch.Tensor) and coefficient.ndim == 3:
                groups.setdefault(node.args[0], []).append((node, coefficient))
    replaced = []
    for source, group in groups.items():
        if len(group) < 2:
            continue
        coefficients = [coefficient for _, coefficient in group]
        if any(c.shape[:2] != coefficients[0].shape[:2] for c in coefficients):
            continue
        name = f'_opt4_grouped_cg_{len(replaced)}'
        gm.register_buffer(name, torch.cat(coefficients, dim=2).detach(), persistent=False)
        with gm.graph.inserting_before(group[0][0]):
            weight = gm.graph.get_attr(name)
            packed = gm.graph.call_function(torch.tensordot, args=(source, weight),
                                           kwargs={'dims':((3, 4), (0, 1))})
        offset = 0
        for node, coefficient in group:
            width = coefficient.shape[2]
            with gm.graph.inserting_before(node):
                value = gm.graph.call_function(operator.getitem,
                    args=(packed, (Ellipsis, slice(offset, offset+width))))
            node.replace_all_uses_with(value)
            gm.graph.erase_node(node)
            offset += width
        replaced.append({'source':source.name, 'instructions':len(group),
                         'input_degree_shape':list(coefficients[0].shape[:2]),
                         'output_widths':[c.shape[2] for c in coefficients]})
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return candidate, replaced
