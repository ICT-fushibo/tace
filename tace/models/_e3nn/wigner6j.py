################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Wigner-6j recoupling for magnetic O(3) interactions."""

import math
from dataclasses import dataclass
from typing import Union

import torch
from e3nn import o3

from tace.utils.torch_scatter import scatter_sum

from .fused import O3ScatterTensorProduct
from .paths import satisfy


def sympy_wigner_6j(
    l1: int,
    l2: int,
    l1l2: int,
    l3: int,
    L: int,
    l23: int,
) -> float:

    from sympy import S
    from sympy.physics import wigner

    return float(
        wigner.wigner_6j(
            S(l1),
            S(l2),
            S(l1l2),
            S(l3),
            S(L),
            S(l23),
        )
    )


def wigner_6j(
    l1: int,
    l2: int,
    l1l2: int,
    l3: int,
    L: int,
    l23: int,
) -> float:
    r"""For the magnetic interaction, the generic angular-momentum labels map to:

    - ``l1``: edge spherical harmonic
    - ``l2``: node feature
    - ``l1l2``: node-edge intermediate
    - ``l3``: magnetic moment
    - ``L``: output
    - ``l23``: node-magnetic intermediate
    """

    return (
        (-1) ** (l1 + l3 + l1l2 + l23)
        * math.sqrt((2 * l1l2 + 1) * (2 * l23 + 1))
        * sympy_wigner_6j(
            l1,
            l2,
            l1l2,
            l3,
            L,
            l23,
        )
    )


@dataclass(frozen=True)
class _CouplingPath:
    node_index: int
    edge_index: int
    pos_irrep: o3.Irrep
    out_irrep: o3.Irrep
    multiplicity: int
    weight_offset: int


class O3Wigner6jScatterTensorProduct(torch.nn.Module):
    r"""Recouple a position-first magnetic tensor product to magnetic-first.

    The reference tree is ``(node x edge_attrs) x magnetic_moments``. The
    executed tree is ``(node x magnetic_moments) x edge_attrs``. Every complete
    reference path remains separate, and all allowed magnetic-first
    intermediate irreps are summed with fixed Wigner-6j coefficients. Therefore
    the two trees are equal for arbitrary per-edge, per-path radial and magnetic
    weights.

    The magnetic input is the axial-vector irrep ``1e``. Tensor products use
    e3nn's default component and element normalization. The registered
    recoupling coefficients include the normalization ratio between the two
    coupling trees.
    """

    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_edge: o3.Irreps,
        irreps_out: o3.Irreps,
        magnetic_irreps: o3.Irreps = o3.Irreps("1x1e"),
        l1l2: Union[str, None] = None,
    ) -> None:
        super().__init__()

        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_edge = o3.Irreps(irreps_edge)
        requested_irreps_out = o3.Irreps(irreps_out)
        self.magnetic_irreps = o3.Irreps(magnetic_irreps)

        paths: list[_CouplingPath] = []
        reference_intermediate = []
        expanded_output = []
        reference_pos_instructions = []
        reference_mag_instructions = []
        weight_offset = 0

        for _, (_, out_irrep) in enumerate(requested_irreps_out):
            for node_index, (multiplicity, node_irrep) in enumerate(self.irreps_node):
                for edge_index, (_, edge_irrep) in enumerate(self.irreps_edge):
                    if not satisfy(node_irrep.l, edge_irrep.l, l1l2):
                        continue
                    for pos_irrep in node_irrep * edge_irrep:
                        if out_irrep not in pos_irrep * self.magnetic_irreps[0].ir:
                            continue

                        path_index = len(paths)
                        paths.append(
                            _CouplingPath(
                                node_index=node_index,
                                edge_index=edge_index,
                                pos_irrep=pos_irrep,
                                out_irrep=out_irrep,
                                multiplicity=multiplicity,
                                weight_offset=weight_offset,
                            )
                        )
                        reference_intermediate.append((multiplicity, pos_irrep))
                        expanded_output.append((multiplicity, out_irrep))
                        reference_pos_instructions.append(
                            (node_index, edge_index, path_index, "uvu", True, 1.0)
                        )
                        reference_mag_instructions.append(
                            (path_index, 0, path_index, "uvu", True, 1.0)
                        )
                        weight_offset += multiplicity

        if not paths:
            raise ValueError("No magnetic Wigner-6j coupling paths were generated")

        self.irreps_out = o3.Irreps(expanded_output)

        self.reference_pos_tp = o3.TensorProduct(
            self.irreps_node,
            self.irreps_edge,
            o3.Irreps(reference_intermediate),
            reference_pos_instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.reference_mag_tp = o3.TensorProduct(
            o3.Irreps(reference_intermediate),
            self.magnetic_irreps,
            self.irreps_out,
            reference_mag_instructions,
            internal_weights=False,
            shared_weights=False,
        )

        recoupled_intermediate = []
        recoupled_mag_instructions = []
        recoupled_pos_instructions = []
        source_weight_indices = []
        recoupling_path_indices = []
        component_recoupling_coefficients = []

        mag_irrep = self.magnetic_irreps[0].ir
        for path_index, path in enumerate(paths):
            node_irrep = self.irreps_node[path.node_index].ir
            edge_irrep = self.irreps_edge[path.edge_index].ir
            for node_mag_irrep in node_irrep * mag_irrep:
                if path.out_irrep not in node_mag_irrep * edge_irrep:
                    continue

                coefficient = wigner_6j(
                    edge_irrep.l,
                    node_irrep.l,
                    path.pos_irrep.l,
                    mag_irrep.l,
                    path.out_irrep.l,
                    node_mag_irrep.l,
                )
                if abs(coefficient) < 1.0e-14:
                    continue

                intermediate_index = len(recoupled_intermediate)
                recoupled_intermediate.append((path.multiplicity, node_mag_irrep))
                recoupled_mag_instructions.append(
                    (
                        path.node_index,
                        0,
                        intermediate_index,
                        "uvu",
                        True,
                        1.0,
                    )
                )
                recoupled_pos_instructions.append(
                    (
                        intermediate_index,
                        path.edge_index,
                        path_index,
                        "uvu",
                        True,
                        1.0,
                    )
                )
                source_weight_indices.extend(
                    range(
                        path.weight_offset,
                        path.weight_offset + path.multiplicity,
                    )
                )
                recoupling_path_indices.append(path_index)
                component_recoupling_coefficients.append(coefficient)

        self.recoupled_mag_tp = o3.TensorProduct(
            self.irreps_node,
            self.magnetic_irreps,
            o3.Irreps(recoupled_intermediate),
            recoupled_mag_instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.recoupled_pos_tp = O3ScatterTensorProduct(
            o3.Irreps(recoupled_intermediate),
            self.irreps_edge,
            self.irreps_out,
            instructions=recoupled_pos_instructions,
        )

        recoupling_coefficients = []
        for intermediate_index, (path_index, coefficient) in enumerate(
            zip(recoupling_path_indices, component_recoupling_coefficients)
        ):
            path = paths[path_index]
            node_mag_irrep = recoupled_intermediate[intermediate_index][1]
            reference_scale = (
                self.reference_pos_tp.instructions[path_index].path_weight
                * self.reference_mag_tp.instructions[path_index].path_weight
            )
            recoupled_scale = (
                self.recoupled_mag_tp.instructions[intermediate_index].path_weight
                * self.recoupled_pos_tp.tp.instructions[intermediate_index].path_weight
            )
            reference_component_scale = math.sqrt(
                path.pos_irrep.dim * path.out_irrep.dim
            )
            recoupled_component_scale = math.sqrt(
                node_mag_irrep.dim * path.out_irrep.dim
            )
            reference_element_scale = reference_scale / reference_component_scale
            recoupled_element_scale = recoupled_scale / recoupled_component_scale
            normalized_coefficient = (
                coefficient * reference_element_scale / recoupled_element_scale
            )
            recoupling_coefficients.extend(
                [normalized_coefficient] * paths[path_index].multiplicity
            )

        self.radial_weight_numel = self.reference_pos_tp.weight_numel
        self.magnetic_weight_numel = self.reference_mag_tp.weight_numel
        if self.radial_weight_numel != weight_offset:
            raise RuntimeError("Unexpected e3nn radial weight layout")
        if self.magnetic_weight_numel != weight_offset:
            raise RuntimeError("Unexpected e3nn magnetic weight layout")

        self.register_buffer(
            "source_weight_indices",
            torch.tensor(source_weight_indices, dtype=torch.int64),
        )
        self.register_buffer(
            "recoupling_coefficients",
            torch.tensor(recoupling_coefficients, dtype=torch.float64),
        )

    def _recoupled(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        magnetic_moments: torch.Tensor,
        radial_weights: torch.Tensor,
        magnetic_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        indices = self.source_weight_indices
        coefficients = self.recoupling_coefficients.to(dtype=radial_weights.dtype)
        magnetic_intermediate = self.recoupled_mag_tp(
            node_feats,
            magnetic_moments,
            magnetic_weights.index_select(-1, indices),
        )
        return self.recoupled_pos_tp(
            magnetic_intermediate,
            edge_attrs,
            radial_weights.index_select(-1, indices) * coefficients,
            edge_index,
        )

    def _reference_edges(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        magnetic_moments: torch.Tensor,
        radial_weights: torch.Tensor,
        magnetic_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        source = edge_index[0]
        pos_intermediate = self.reference_pos_tp(
            node_feats[source], edge_attrs, radial_weights
        )
        return self.reference_mag_tp(
            pos_intermediate,
            magnetic_moments[source],
            magnetic_weights[source],
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        magnetic_moments: torch.Tensor,
        radial_weights: torch.Tensor,
        magnetic_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        return self._recoupled(
            node_feats,
            edge_attrs,
            magnetic_moments,
            radial_weights,
            magnetic_weights,
            edge_index,
        )

    def forward_reference(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        magnetic_moments: torch.Tensor,
        radial_weights: torch.Tensor,
        magnetic_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the position-first tree for numerical validation."""

        messages = self._reference_edges(
            node_feats,
            edge_attrs,
            magnetic_moments,
            radial_weights,
            magnetic_weights,
            edge_index,
        )
        return scatter_sum(
            messages,
            edge_index[1],
            dim=0,
            dim_size=node_feats.size(0),
        )
