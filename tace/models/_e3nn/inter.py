################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Union

import torch
from e3nn import o3

from tace.utils.torch_scatter import scatter_sum

from ..layout import LayoutTransform
from ..linear import e3nnLinear
from ..mlp import MLP, ScaledSigmoid, ScaledSiLU
from .base import Interaction
from .fused import O3ScatterTensorProduct, uuSO2ScatterTensorProduct, uvSO2TensorProduct
from .layer_norm import get_normalization_layer
from .nonlinear import get_nonlinear_layer
from .residual import get_resnet_layer


class CgtpInteraction(Interaction):
    """
    An interaction module based on Clebsch-Gordan tensor products (CGTP).

    This module performs edge-level convolution using Clebsch-Gordan tensor
    products. It supports operator fusion via OpenEquivariance or CuEquivariance,
    which can significantly reduce memory consumption and improve efficiency.

    This interaction block does not directly add nonlinearity to the edge.
    """

    def _setup(self) -> None:

        self.linear_up = e3nnLinear(
            self.irreps_in,
            self.irreps_in,
            bias=self.use_bias,
        )

        self.rejector = O3ScatterTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            l1l2=self.l1l2,
        )

        (
            self.nonlinearity,
            self.linear_nonlinearity,
            linear_down_irreps_out,
        ) = get_nonlinear_layer(
            self.nonlinear_type,
            o3.Irreps([(self.node_wise_hidden, ir) for _, ir in self.irreps_out]),
            self.irreps_out,
            gate_m0=self.gate_m0,
            scalar_act=self.scalar_act,
            tensor_act=self.tensor_act,
        )

        self.linear_down = e3nnLinear(
            self.rejector.irreps_out.simplify(),
            linear_down_irreps_out,
            bias=self.use_bias,
        )

        self.edge_info = MLP(
            [self.edge_feats_channel] + self.radial_mlp + [self.rejector.weight_numel],
            bias=self.radial_bias,
            layer_norm=self.radial_layer_norm,
            act="silu",
        )

        if self.scatter_norm == "density" or self.scatter_norm == "no_cutoff_density":
            self.edge_density = MLP(
                [self.edge_feats_channel, 64, 1],
                bias=self.radial_bias,
                layer_norm=self.radial_layer_norm,
                act="silu",
            )  # From MACE
            self.alpha = torch.nn.Parameter(torch.tensor(self.avg_num_neighbors))
            self.beta = torch.nn.Parameter(torch.tensor(0.0))

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type == "BB":
            self.resnetBB = get_resnet_layer(
                self.irreps_in,
                self.irreps_sc,
                bias=self.use_bias,
                num_elements=self.num_elements,
                resnet_type=self.resnet_linear_type,
            )

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type == "BAB":
            self.resnetBA = get_resnet_layer(
                self.irreps_in,
                self.irreps_out,
                bias=self.use_bias,
                num_elements=self.num_elements,
                resnet_type=self.resnet_linear_type,
            )
            if (
                self.layer > 0 or self.use_first_dropout
            ) and self.stochastic_depth_p > 0.0:
                from .dropout import GraphDropPath

                self.stochastic_depth = GraphDropPath(self.stochastic_depth_p)

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type in [
            "AB",
            "BAB",
        ]:
            self.resnetAB = get_resnet_layer(
                self.irreps_out,
                self.irreps_sc,
                bias=self.use_bias,
                num_elements=self.num_elements,
                resnet_type=self.resnet_linear_type,
            )

        if (
            self.use_first_pre_norm or self.layer > 0
        ) and self.pre_norm_type is not None:
            if self.resnet_type in ["BB", "BAB"]:
                self.norm1 = get_normalization_layer(
                    self.pre_norm_type,
                    ls=self.irreps_in.lmax,
                    num_channels=self.num_channel,
                )
                self.reshape1 = LayoutTransform(self.irreps_in)
            if self.resnet_type in ["AB", "BAB"]:
                self.norm2 = get_normalization_layer(
                    self.pre_norm_type,
                    ls=self.irreps_out.lmax,
                    num_channels=self.num_channel,
                )
                self.reshape2 = LayoutTransform(self.irreps_out)

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs_total: torch.Tensor,
        node_attrs_slice: torch.Tensor,
        radial_basis,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Union[torch.Tensor, None],
        graph,
        wigner: Union[torch.Tensor, None],
        wigner_inv: Union[torch.Tensor, None],
        batch,
    ):

        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        density = None
        resBB = None
        resBA = None
        resAB = None

        if hasattr(self, "resnetBB"):
            if self.resnet_linear_type == "aware":
                resBB = self.resnetBB(node_feats, node_attrs_slice)
            else:
                resBB = self.resnetBB(node_feats)

        if hasattr(self, "resnetBA"):
            if self.resnet_linear_type == "aware":
                resBA = self.resnetBA(node_feats, node_attrs_slice)
            else:
                resBA = self.resnetBA(node_feats)

        if hasattr(self, "norm1"):
            node_feats = self.reshape1.inverse(self.norm1(self.reshape1(node_feats)))

        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(node_feats, lmp_data, lmp_natoms, self.layer)

        conv_weights = self.edge_info(edge_feats)
        if cutoff is not None:
            conv_weights = conv_weights * cutoff

        m_i = self.linear_down(
            self.truncate_ghosts(
                self.rejector(node_feats, edge_attrs, conv_weights, edge_index), nlocal
            )
        )

        if hasattr(self, "edge_density"):
            density = torch.tanh(self.edge_density(edge_feats) ** 2)
            if cutoff is not None and self.apply_density_cutoff:
                density = density * cutoff
            # density = density * cutoff
            density = scatter_sum(
                density, edge_index[1], dim=0, dim_size=node_attrs_total.size(0)
            )
            density = self.truncate_ghosts(density, nlocal)
            density = density * self.beta + self.alpha
            density = density.masked_fill(density == 0, 1e-9)

        if self.scatter_norm is None:
            pass
        elif self.scatter_norm == "avg_num_neighbors":
            m_i = m_i / self.avg_num_neighbors
        else:
            m_i = m_i / density

        m_i = self.linear_nonlinearity(self.nonlinearity(m_i))

        if resBA is not None:
            if hasattr(self, "stochastic_depth"):
                m_i = self.stochastic_depth(m_i, batch)
            m_i = m_i + resBA

        if hasattr(self, "resnetAB"):
            if self.resnet_linear_type == "aware":
                resAB = self.resnetAB(m_i, node_attrs_slice)
            else:
                resAB = self.resnetAB(m_i)

        if hasattr(self, "norm2"):
            m_i = self.reshape2.inverse(self.norm2(self.reshape2(m_i)))

        if resBB is not None:
            sc = resBB
        elif resAB is not None:
            sc = resAB
        else:
            sc = None

        return m_i, self.truncate_ghosts(sc, nlocal)


class uuSO2Interaction(Interaction):
    """
    An interaction module based on uuSO2Linear.

    It achieves the same accuracy and extrapolation capability as CGTP.
    Set `export TACE_USE_TRITON=1` to reduce memory.

    This interaction block does not directly add nonlinearity to the message.
    """

    def _setup(self) -> None:

        assert self.parity == False, (
            "uuSO2InteractionArchitecture1 not support O(3) group"
        )
        assert self.irreps_in.lmax > 0, (
            "uuSO2InteractionArchitecture1's irreps_in.lmax must > 0, "
            "use uuSO2InteractionArchitecture1 from the second layer or use other node_embedding with l > 0"
        )
        assert self.edge_nonlinear == None

        self.linear_up = e3nnLinear(
            self.irreps_in,
            self.irreps_in,
            bias=self.use_bias,
        )

        self.rejector = uuSO2ScatterTensorProduct(
            mmax=self.mmax,
            lmax=self.lmax,
            num_channel=self.num_channel,
            weight_type=self.so2_linear_type,
            l1l3=self.so2_l1l3,
            reshape_in=LayoutTransform(self.irreps_in),
            reshape_out=LayoutTransform(self.irreps_out),
        )

        (
            self.nonlinearity,
            self.linear_nonlinearity,
            linear_down_irreps_out,
        ) = get_nonlinear_layer(
            self.nonlinear_type,
            o3.Irreps([(self.node_wise_hidden, ir) for _, ir in self.irreps_out]),
            self.irreps_out,
            gate_m0=self.gate_m0,
            scalar_act=self.scalar_act,
            tensor_act=self.tensor_act,
        )

        self.linear_down = e3nnLinear(
            self.irreps_in,
            linear_down_irreps_out,
            bias=self.use_bias,
        )

        self.edge_info = MLP(
            [self.edge_feats_channel] + self.radial_mlp + [self.rejector.weight_numel],
            bias=self.radial_bias,
            layer_norm=self.radial_layer_norm,
            act="silu",
        )

        self.apply_density_cutoff = True
        if self.scatter_norm == "density":
            self.edge_density = MLP(
                [self.edge_feats_channel, 64, 1],
                bias=self.radial_bias,
                layer_norm=self.radial_layer_norm,
                act="silu",
            )  # From MACE
            self.alpha = torch.nn.Parameter(torch.tensor(self.avg_num_neighbors))
            self.beta = torch.nn.Parameter(torch.tensor(0.0))

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type == "BB":
            self.resnetBB = get_resnet_layer(
                self.irreps_in,
                self.irreps_sc,
                bias=self.use_bias,
                num_elements=self.num_elements,
                resnet_type=self.resnet_linear_type,
            )

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type == "BAB":
            self.resnetBA = get_resnet_layer(
                self.irreps_in,
                self.irreps_out,
                bias=self.use_bias,
                num_elements=self.num_elements,
                resnet_type=self.resnet_linear_type,
            )
            if (
                self.layer > 0 or self.use_first_dropout
            ) and self.stochastic_depth_p > 0.0:
                from .dropout import GraphDropPath

                self.stochastic_depth = GraphDropPath(self.stochastic_depth_p)

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type in [
            "AB",
            "BAB",
        ]:
            self.resnetAB = get_resnet_layer(
                self.irreps_out,
                self.irreps_sc,
                bias=self.use_bias,
                num_elements=self.num_elements,
                resnet_type=self.resnet_linear_type,
            )

        if (
            self.use_first_pre_norm or self.layer > 0
        ) and self.pre_norm_type is not None:
            if self.resnet_type in ["BB", "BAB"]:
                self.norm1 = get_normalization_layer(
                    self.pre_norm_type,
                    ls=self.irreps_in.lmax,
                    num_channels=self.num_channel,
                )
                self.reshape1 = LayoutTransform(self.irreps_in)
            if self.resnet_type in ["AB", "BAB"]:
                self.norm2 = get_normalization_layer(
                    self.pre_norm_type,
                    ls=self.irreps_out.lmax,
                    num_channels=self.num_channel,
                )
                self.reshape2 = LayoutTransform(self.irreps_out)

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs_total: torch.Tensor,
        node_attrs_slice: torch.Tensor,
        radial_basis,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Union[torch.Tensor, None],
        graph,
        wigner: Union[torch.Tensor, None],
        wigner_inv: Union[torch.Tensor, None],
        batch,
    ):

        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        density = None
        resBB = None
        resBA = None
        resAB = None

        if hasattr(self, "resnetBB"):
            if self.resnet_linear_type == "aware":
                resBB = self.resnetBB(node_feats, node_attrs_slice)
            else:
                resBB = self.resnetBB(node_feats)

        if hasattr(self, "resnetBA"):
            if self.resnet_linear_type == "aware":
                resBA = self.resnetBA(node_feats, node_attrs_slice)
            else:
                resBA = self.resnetBA(node_feats)

        if hasattr(self, "norm1"):
            node_feats = self.reshape1.inverse(self.norm1(self.reshape1(node_feats)))

        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(node_feats, lmp_data, lmp_natoms, self.layer)

        conv_weights = self.edge_info(edge_feats)

        if cutoff is not None:
            conv_weights = conv_weights * cutoff

        m_i = self.linear_down(
            self.truncate_ghosts(
                self.rejector(
                    node_feats,
                    conv_weights,
                    edge_index,
                    wigner,
                    wigner_inv,
                ),
                nlocal,
            )
        )

        if hasattr(self, "edge_density"):
            density = torch.tanh(self.edge_density(edge_feats) ** 2)
            if cutoff is not None and self.apply_density_cutoff:
                density = density * cutoff
            density = scatter_sum(
                density, edge_index[1], dim=0, dim_size=node_attrs_total.size(0)
            )
            density = self.truncate_ghosts(density, nlocal)
            density = density * self.beta + self.alpha
            density = density.masked_fill(density == 0, 1e-9)

        if self.scatter_norm is None:
            pass
        elif self.scatter_norm == "avg_num_neighbors":
            m_i = m_i / self.avg_num_neighbors
        else:
            m_i = m_i / density

        m_i = self.linear_nonlinearity(self.nonlinearity(m_i))

        if resBA is not None:
            if hasattr(self, "stochastic_depth"):
                m_i = self.stochastic_depth(m_i, batch)
            m_i = m_i + resBA

        if hasattr(self, "resnetAB"):
            if self.resnet_linear_type == "aware":
                resAB = self.resnetAB(m_i, node_attrs_slice)
            else:
                resAB = self.resnetAB(m_i)

        if hasattr(self, "norm2"):
            m_i = self.reshape2.inverse(self.norm2(self.reshape2(m_i)))

        if resBB is not None:
            sc = resBB
        elif resAB is not None:
            sc = resAB
        else:
            sc = None

        return m_i, self.truncate_ghosts(sc, nlocal)


# A little BUG
class uvSO2Interaction(Interaction):
    """
    An interaction module based on uvSO2Linear,
    Edge Cluster Expansion and Radial Rotary Attention.

    It achieves highest accuracy at the expanse of computational efficiency.

    This interaction block add nonlinearity to the message.
    """

    def _setup(self) -> None:

        assert self.parity == False, "uvSO2Interaction not support O(3) group"
        assert self.irreps_in.lmax > 0, (
            "uvSO2Interaction's irreps_in.lmax must > 0, "
            "use uvSO2Interaction from the second layer or use other node_embedding with l > 0"
        )
        assert (
            self.edge_nonlinear == "so2_sigmoid_gate"
            or self.edge_nonlinear == "so2_silu_gate"
        )
        edge_act = self.edge_nonlinear.split("_")[1]
        scalar_act = self.scalar_act or edge_act
        tensor_act = self.tensor_act or edge_act
        self.scatter_norm = None if self.use_graph_softmax else self.scatter_norm

        self.linear_up = e3nnLinear(
            self.irreps_in,
            self.irreps_in,
            bias=self.use_bias,
        )
        self.rejector = uvSO2TensorProduct(
            mmax=self.mmax,
            lmax=self.lmax,
            num_channel=self.num_channel,
            num_radial_basis=self.num_radial_basis,
            num_head=self.num_head,
            use_temperature=self.use_temperature,
            edge_ace_hidden=self.edge_ace_hidden,
            edge_wise_hidden=self.edge_wise_hidden,
            so2_linear_type=self.so2_linear_type,
            gate_m0=self.gate_m0,
            use_so2_edge_ace=self.use_so2_edge_ace,
            use_graph_softmax=self.use_graph_softmax,
            reshape_in=LayoutTransform(self.irreps_in),
            reshape_out=LayoutTransform(
                o3.Irreps([(self.edge_wise_hidden, ir) for _, ir in self.irreps_out])
            ),
            scalar_act=ScaledSigmoid() if scalar_act == "sigmoid" else ScaledSiLU(),
            tensor_act=ScaledSigmoid() if tensor_act == "sigmoid" else ScaledSiLU(),
            use_radial_phase=self.use_radial_phase,
        )

        (
            self.nonlinearity,
            self.linear_nonlinearity,
            linear_down_irreps_out,
        ) = get_nonlinear_layer(
            self.nonlinear_type,
            o3.Irreps([(self.node_wise_hidden, ir) for _, ir in self.irreps_out]),
            self.irreps_out,
            gate_m0=self.gate_m0,
            scalar_act=self.scalar_act,
            tensor_act=self.tensor_act,
        )

        self.linear_down = e3nnLinear(
            o3.Irreps([(self.edge_wise_hidden, ir) for _, ir in self.irreps_out]),
            linear_down_irreps_out,
            bias=self.use_bias,
        )

        self.edge_info = MLP(
            [self.edge_feats_channel] + self.radial_mlp + [self.rejector.weight_numel],
            bias=self.radial_bias,
            layer_norm=self.radial_layer_norm,
            act="silu",
        )

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type == "BB":
            self.resnetBB = get_resnet_layer(
                self.irreps_in,
                self.irreps_sc,
                bias=self.use_bias,
                num_elements=self.num_elements,
                resnet_type=self.resnet_linear_type,
            )

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type == "BAB":
            self.resnetBA = get_resnet_layer(
                self.irreps_in,
                self.irreps_out,
                bias=self.use_bias,
                num_elements=self.num_elements,
                resnet_type=self.resnet_linear_type,
            )
            if (
                self.layer > 0 or self.use_first_dropout
            ) and self.stochastic_depth_p > 0.0:
                from .dropout import GraphDropPath

                self.stochastic_depth = GraphDropPath(self.stochastic_depth_p)

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type in [
            "AB",
            "BAB",
        ]:
            self.resnetAB = get_resnet_layer(
                self.irreps_out,
                self.irreps_sc,
                bias=self.use_bias,
                num_elements=self.num_elements,
                resnet_type=self.resnet_linear_type,
            )

        if (
            self.use_first_pre_norm or self.layer > 0
        ) and self.pre_norm_type is not None:
            if self.resnet_type in ["BB", "BAB"]:
                self.norm1 = get_normalization_layer(
                    self.pre_norm_type,
                    ls=self.irreps_in.lmax,
                    num_channels=self.num_channel,
                )
                self.reshape1 = LayoutTransform(self.irreps_in)
            if self.resnet_type in ["AB", "BAB"]:
                self.norm2 = get_normalization_layer(
                    self.pre_norm_type,
                    ls=self.irreps_out.lmax,
                    num_channels=self.num_channel,
                )
                self.reshape2 = LayoutTransform(self.irreps_out)

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs_total: torch.Tensor,
        node_attrs_slice: torch.Tensor,
        radial_basis,
        edge_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Union[torch.Tensor, None],
        graph,
        wigner: Union[torch.Tensor, None],
        wigner_inv: Union[torch.Tensor, None],
        batch,
    ):

        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        resBB = None
        resBA = None
        resAB = None

        if hasattr(self, "resnetBB"):
            if self.resnet_linear_type == "aware":
                resBB = self.resnetBB(node_feats, node_attrs_slice)
            else:
                resBB = self.resnetBB(node_feats)

        if hasattr(self, "resnetBA"):
            if self.resnet_linear_type == "aware":
                resBA = self.resnetBA(node_feats, node_attrs_slice)
            else:
                resBA = self.resnetBA(node_feats)

        if hasattr(self, "norm1"):
            node_feats = self.reshape1.inverse(self.norm1(self.reshape1(node_feats)))

        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(node_feats, lmp_data, lmp_natoms, self.layer)
        m_i = self.truncate_ghosts(
            self.rejector(
                node_feats,
                self.edge_info(edge_feats),
                edge_index,
                cutoff,
                wigner,
                wigner_inv,
                radial_basis,
            ),
            nlocal,
        )
        m_i = self.linear_down(m_i)
        m_i = self.linear_nonlinearity(self.nonlinearity(m_i))

        if resBA is not None:
            if hasattr(self, "stochastic_depth"):
                m_i = self.stochastic_depth(m_i, batch)
            m_i = m_i + resBA

        if hasattr(self, "resnetAB"):
            if self.resnet_linear_type == "aware":
                resAB = self.resnetAB(m_i, node_attrs_slice)
            else:
                resAB = self.resnetAB(m_i)

        if hasattr(self, "norm2"):
            m_i = self.reshape2.inverse(self.norm2(self.reshape2(m_i)))

        if resBB is not None:
            sc = resBB
        elif resAB is not None:
            sc = resAB
        else:
            sc = None

        return m_i, self.truncate_ghosts(sc, nlocal)


INTERACTION: Dict[str, Interaction] = {
    "normal": CgtpInteraction,
    "spectral": CgtpInteraction,
    "cgtp": CgtpInteraction,
    "uu_so2": uuSO2Interaction,
    "so2": uvSO2Interaction,
    "uv_so2": uvSO2Interaction,
    "attn": uvSO2Interaction,
}
