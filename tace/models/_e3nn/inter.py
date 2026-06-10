################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Dict, Union


import torch
from e3nn import o3


from tace.utils.torch_scatter import scatter_sum
from ..mlp import ACTIVATION, FFN
from ..layout import LayoutTransform
from .base import Interaction
from ..linear import e3nnLinear, e3nnElementLinear
from .fused import O3ScatterTensorProduct, uvSO2TensorProduct, uuSO2ScatterTensorProduct, AttentionSO2TensorProduct
from .nonlinear import O3Gate, O3Norm
from .layer_norm import get_normalization_layer


class CgtpInteraction(Interaction):
    """
    An interaction module based on Clebsch-Gordan tensor products (CGTP).

    This module performs edge-level convolution using Clebsch-Gordan tensor
    products. It supports operator fusion via OpenEquivariance or CuEquivariance,
    which can significantly reduce memory consumption and improve efficiency.

    This interaction block does not directly add nonlinearity to the message.
    """

    def _setup(self) -> None:

        assert self.edge_wise_hidden == self.num_channel

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

        irreps_node_wise_hidden = o3.Irreps([(self.node_wise_hidden, ir) for _, ir in self.irreps_out])
        if self.nonlinear_type == 'gate':
            irreps_gated = irreps_node_wise_hidden
            irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in irreps_node_wise_hidden)
            self.nonlinearity = O3Gate(
                irreps_gates=irreps_gates,
                act_gates=[ACTIVATION[self.nonlinear_act]()] * len(irreps_gates),
                irreps_gated=irreps_gated,
            )
            linear_down_irreps_out = self.nonlinearity.irreps_in.simplify()
            self.linear_nonlinearity = e3nnLinear(
                irreps_node_wise_hidden, 
                self.irreps_out,  
                bias=self.use_bias,
            )
        elif self.nonlinear_type == 'norm':
            self.nonlinearity = O3Norm(
                irreps=irreps_node_wise_hidden,
                activation=[ACTIVATION[self.nonlinear_act]()],
            )
            linear_down_irreps_out = self.nonlinearity.irreps_in.simplify()
            self.linear_nonlinearity = e3nnLinear(
                irreps_node_wise_hidden, 
                self.irreps_out,  
                bias=self.use_bias,
            )
        else:
            self.nonlinearity = torch.nn.Identity()
            self.linear_nonlinearity = torch.nn.Identity()
            linear_down_irreps_out = irreps_node_wise_hidden

        self.linear_down = e3nnLinear(
            self.rejector.irreps_out.simplify(),
            linear_down_irreps_out,
            bias=self.use_bias,
        )

        self.edge_info = FFN[self.edge_info_type](
            [self.edge_feats_channel] + self.radial_mlp + [self.rejector.weight_numel],
            bias=self.radial_bias,
            layer_norm=self.radial_layer_norm,
            act=self.radial_act,
        )

        if self.scatter_norm == 'density' or self.scatter_norm == 'no_cutoff_density': 
            self.edge_density = FFN[self.edge_info_type](
                [self.edge_feats_channel, 64, 1],
                bias=self.radial_bias,
                layer_norm=self.radial_layer_norm,
                act=self.radial_act,
            ) # From MACE
            self.alpha = torch.nn.Parameter(torch.tensor(self.avg_num_neighbors))
            self.beta = torch.nn.Parameter(torch.tensor(0.0))

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type == 'BB':
            if self.resnet_linear_type == 'agnostic':
                self.resnetBB = e3nnLinear(
                    irreps_in=self.irreps_in,
                    irreps_out=self.irreps_sc,
                    bias=self.use_bias,
                )
            else:
                self.resnetBB = e3nnElementLinear(
                    irreps_in = self.irreps_in,
                    irreps_out = self.irreps_sc,
                    bias=self.use_bias,
                    num_elements=self.num_elements,
                )

        if (self.use_first_pre_norm or self.layer > 0) and self.pre_norm_type is not None:
            if self.resnet_type in ['BB']:
                self.norm1 = get_normalization_layer(
                    self.pre_norm_type,
                    ls=self.irreps_in.lmax,
                    num_channels=self.num_channel,
                )
                self.reshape1 = LayoutTransform(self.irreps_in)

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
    ):

        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        density = None
        resBB = None

        if hasattr(self, 'resnetBB'):
            if self.resnet_linear_type == 'aware':
                resBB = self.resnetBB(node_feats, node_attrs_slice)
            else:
                resBB = self.resnetBB(node_feats) 

        if hasattr(self, 'norm1'):
            node_feats = self.reshape1.inverse(self.norm1(self.reshape1(node_feats)))

        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(node_feats, lmp_data, lmp_natoms, self.layer)

        conv_weights = self.edge_info(edge_feats)
        if cutoff is not None:
            conv_weights = conv_weights * cutoff

        m_i = self.linear_down(
            self.truncate_ghosts(
                self.rejector(node_feats, edge_attrs, conv_weights, edge_index), 
                nlocal
            )
        )

        if hasattr(self, "edge_density"):
            density = torch.tanh(self.edge_density(edge_feats) ** 2)
            if cutoff is not None and self.apply_density_cutoff:
                density = density * cutoff
            # density = density * cutoff
            density = scatter_sum(density, edge_index[1], dim=0, dim_size=node_attrs_total.size(0))
            density  = self.truncate_ghosts(density , nlocal)
            density = density * self.beta + self.alpha
            density = density.masked_fill(density == 0, 1e-9)


        if self.scatter_norm is None:
            pass
        elif self.scatter_norm == 'avg_num_neighbors':
            m_i = m_i / self.avg_num_neighbors
        else:
            m_i = m_i / density

        m_i = self.linear_nonlinearity(self.nonlinearity(m_i))

        return m_i, self.truncate_ghosts(resBB, nlocal)
    

class uuSO2InteractionArchitecture1(Interaction):
    """
    An interaction module based on uuSO2Linear.

    It achieves the same accuracy and extrapolation capability as CGTP. 
    Moreover, if an operator fusion library becomes available in the future, 
    it could significantly improve memory efficiency and outperform CGTP even 
    when CGTP is accelerated by oeq and cueq.

    This interaction block does not directly add nonlinearity to the message.
    """

    def _setup(self) -> None:

        assert self.parity == False, "uuSO2InteractionArchitecture1 not support O(3) group"
        assert self.irreps_in.lmax > 0, (
            "uuSO2InteractionArchitecture1's irreps_in.lmax must > 0, "
            "use uuSO2InteractionArchitecture1 from the second layer or use other node_embedding with l > 0"
        )
        assert self.edge_nonlinear == None
        assert self.edge_wise_hidden == self.num_channel

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

        irreps_node_wise_hidden = o3.Irreps([(self.node_wise_hidden, ir) for _, ir in self.irreps_out])
        if self.nonlinear_type == 'gate':
            irreps_gated = irreps_node_wise_hidden
            irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in irreps_node_wise_hidden)
            self.nonlinearity = O3Gate(
                irreps_gates=irreps_gates,
                act_gates=[ACTIVATION[self.nonlinear_act]()] * len(irreps_gates),
                irreps_gated=irreps_gated,
            )
            linear_down_irreps_out = self.nonlinearity.irreps_in.simplify()
            self.linear_nonlinearity = e3nnLinear(
                irreps_node_wise_hidden, 
                self.irreps_out,  
                bias=self.use_bias,
            )
        elif self.nonlinear_type == 'norm':
            self.nonlinearity = O3Norm(
                irreps=irreps_node_wise_hidden,
                activation=[ACTIVATION[self.nonlinear_act]()],
            )
            linear_down_irreps_out = self.nonlinearity.irreps_in.simplify()
            self.linear_nonlinearity = e3nnLinear(
                irreps_node_wise_hidden, 
                self.irreps_out,  
                bias=self.use_bias,
            )
        else:
            self.nonlinearity = torch.nn.Identity()
            self.linear_nonlinearity = torch.nn.Identity()
            linear_down_irreps_out = irreps_node_wise_hidden

        self.linear_down = e3nnLinear(
            self.irreps_in,
            linear_down_irreps_out,
            bias=self.use_bias,
        )

        self.edge_info = FFN[self.edge_info_type](
            [self.edge_feats_channel] + self.radial_mlp + [self.rejector.weight_numel],
            bias=self.radial_bias,
            layer_norm=self.radial_layer_norm,
            act=self.radial_act,
        )

        self.apply_density_cutoff = True
        if self.scatter_norm == 'density': 
            self.edge_density = FFN[self.edge_info_type](
                [self.edge_feats_channel, 64, 1],
                bias=self.radial_bias,
                layer_norm=self.radial_layer_norm,
                act=self.radial_act,
            ) # From MACE
            self.alpha = torch.nn.Parameter(torch.tensor(self.avg_num_neighbors))
            self.beta = torch.nn.Parameter(torch.tensor(0.0))


        if (self.use_first_resnet or self.layer > 0) and self.resnet_type == 'BB':
            if self.resnet_linear_type == 'agnostic':
                self.resnetBB = e3nnLinear(
                    irreps_in=self.irreps_in,
                    irreps_out=self.irreps_sc,
                    bias=self.use_bias,
                )
            else:
                self.resnetBB = e3nnElementLinear(
                    irreps_in = self.irreps_in,
                    irreps_out = self.irreps_sc,
                    bias=self.use_bias,
                    num_elements=self.num_elements,
                )

        if (self.use_first_pre_norm or self.layer > 0) and self.pre_norm_type is not None:
            if self.resnet_type in ['BB']:
                self.norm1 = get_normalization_layer(
                    self.pre_norm_type,
                    ls=self.irreps_in.lmax,
                    num_channels=self.num_channel,
                )
                self.reshape1 = LayoutTransform(self.irreps_in)

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
    ):

        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        density = None
        resBB = None

        if hasattr(self, 'resnetBB'):
            if self.resnet_linear_type == 'aware':
                resBB = self.resnetBB(node_feats, node_attrs_slice)
            else:
                resBB = self.resnetBB(node_feats) 

        if hasattr(self, 'norm1'):
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
                nlocal
            )
        )

        if hasattr(self, "edge_density"):
            density = torch.tanh(self.edge_density(edge_feats) ** 2)
            if cutoff is not None and self.apply_density_cutoff:
                density = density * cutoff
            density = scatter_sum(density, edge_index[1], dim=0, dim_size=node_attrs_total.size(0))
            density  = self.truncate_ghosts(density , nlocal)
            density = density * self.beta + self.alpha
            density = density.masked_fill(density == 0, 1e-9)

        if self.scatter_norm is None:
            pass
        elif self.scatter_norm == 'avg_num_neighbors':
            m_i = m_i / self.avg_num_neighbors
        else:
            m_i = m_i / density

        m_i = self.linear_nonlinearity(self.nonlinearity(m_i))

        return m_i, self.truncate_ghosts(resBB, nlocal)
    

class uvSO2InteractionArchitecture2(Interaction):
    """
    An interaction module based on uvSO2Linear and Edge Complex Product Basis.

    This block is primarily designed for uMLIP. 
    In general, there is little need to use this module unless you are specifically training uMLIP. 
    It sacrifices some extrapolation capability in exchange for improved fitting accuracy.

    This interaction block add nonlinearity to the message.
    """
    def _setup(self) -> None:

        assert self.parity == False, "uvSO2InteractionArchitecture2 not support O(3) group"
        assert self.irreps_in.lmax > 0, (
            "uvSO2InteractionArchitecture2's irreps_in.lmax must > 0, "
            "use uvSO2InteractionArchitecture2 from the second layer or use other node_embedding with l > 0"
        )
        assert self.edge_nonlinear == 'so2_sigmoid_gate'

        self.linear_up = e3nnLinear(
            self.irreps_in,
            self.irreps_in,
            bias=self.use_bias,
        )    

        self.rejector = uvSO2TensorProduct(
            mmax=self.mmax,
            lmax=self.lmax,
            num_channel=self.num_channel,
            edge_wise_hidden=self.edge_wise_hidden,
            so2_linear_type=self.so2_linear_type,
            num_elements=self.num_elements,
            use_so2_edge_ace=self.use_so2_edge_ace,
            agnostic=self.so2_agnostic,
            reshape_in=LayoutTransform(self.irreps_in),
            reshape_out=LayoutTransform(self.irreps_out),
        )

        irreps_node_wise_hidden = o3.Irreps([(self.node_wise_hidden, ir) for _, ir in self.irreps_out])

        if self.nonlinear_type == 'gate':  
            irreps_gated = irreps_node_wise_hidden
            irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in irreps_node_wise_hidden)
            self.nonlinearity = O3Gate(
                irreps_gates=irreps_gates,
                act_gates=[ACTIVATION[self.nonlinear_act]()] * len(irreps_gates),
                irreps_gated=irreps_gated,
            )
            linear_down_irreps_out = self.nonlinearity.irreps_in.simplify()
            self.linear_nonlinearity = e3nnLinear(
                irreps_node_wise_hidden, 
                self.irreps_out,  
                bias=self.use_bias,
            )
        elif self.nonlinear_type == 'norm':
            self.nonlinearity = O3Norm(
                irreps=irreps_node_wise_hidden,
                activation=[ACTIVATION[self.nonlinear_act]()],
            )
            linear_down_irreps_out = self.nonlinearity.irreps_in.simplify()
            self.linear_nonlinearity = e3nnLinear(
                irreps_node_wise_hidden, 
                self.irreps_out,  
                bias=self.use_bias,
            )
        else:
            self.nonlinearity = torch.nn.Identity()
            self.linear_nonlinearity = torch.nn.Identity()
            linear_down_irreps_out = irreps_node_wise_hidden

        self.linear_down = e3nnLinear(
            self.irreps_out,
            linear_down_irreps_out,
            bias=self.use_bias,
        )

        self.edge_info = FFN[self.edge_info_type](
            [self.edge_feats_channel] + self.radial_mlp + [self.rejector.weight_numel],
            bias=self.radial_bias,
            layer_norm=self.radial_layer_norm,
            act=self.radial_act,
        )

        self.apply_density_cutoff = True
        if self.scatter_norm == 'density': 
            self.edge_density = FFN[self.edge_info_type](
                [self.edge_feats_channel, 64, 1],
                bias=self.radial_bias,
                layer_norm=self.radial_layer_norm,
                act=self.radial_act,
            ) # From MACE
            self.alpha = torch.nn.Parameter(torch.tensor(self.avg_num_neighbors))
            self.beta = torch.nn.Parameter(torch.tensor(0.0))


        if (self.use_first_resnet or self.layer > 0) and self.resnet_type == 'BB':
            if self.resnet_linear_type == 'agnostic':
                self.resnetBB = e3nnLinear(
                    irreps_in=self.irreps_in,
                    irreps_out=self.irreps_sc,
                    bias=self.use_bias,
                )
            else:
                self.resnetBB = e3nnElementLinear(
                    irreps_in = self.irreps_in,
                    irreps_out = self.irreps_sc,
                    bias=self.use_bias,
                    num_elements=self.num_elements,
                )

        if (self.use_first_pre_norm or self.layer > 0) and self.pre_norm_type is not None:
            if self.resnet_type in ['BB']:
                self.norm1 = get_normalization_layer(
                    self.pre_norm_type,
                    ls=self.irreps_in.lmax,
                    num_channels=self.num_channel,
                )
                self.reshape1 = LayoutTransform(self.irreps_in)

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
    ):
    
        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        density = None
        resBB = None

        if hasattr(self, 'resnetBB'):
            if self.resnet_linear_type == 'aware':
                resBB = self.resnetBB(node_feats, node_attrs_slice)
            else:
                resBB = self.resnetBB(node_feats) 

        if hasattr(self, 'norm1'):
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
                    node_attrs_slice, 
                    conv_weights, 
                    edge_index, 
                    cutoff,
                    wigner,
                    wigner_inv,
                ), 
                nlocal
            ) # check   node_attrs_slice TODO
        )

        if hasattr(self, "edge_density"):
            density = torch.tanh(self.edge_density(edge_feats) ** 2)
            if cutoff is not None and self.apply_density_cutoff:
                density = density * cutoff
            density = scatter_sum(density, edge_index[1], dim=0, dim_size=node_attrs_total.size(0))
            density  = self.truncate_ghosts(density , nlocal)
            density = density * self.beta + self.alpha
            density = density.masked_fill(density == 0, 1e-9)

        if self.scatter_norm is None:
            m_i = m_i
        elif self.scatter_norm == 'avg_num_neighbors':
            m_i = m_i / self.avg_num_neighbors
        else:
            m_i = m_i / density

        m_i = self.linear_nonlinearity(self.nonlinearity(m_i))

        return m_i, self.truncate_ghosts(resBB, nlocal)
    

class AttentionInteractionArchitecture3(Interaction):
    def _setup(self) -> None:

        assert self.parity == False, "uvSO2Interaction not support O(3) group"
        assert self.irreps_in.lmax > 0, (
            "uvSO2Interaction's irreps_in.lmax must > 0, "
            "use uvSO2Interaction from the second layer or use other node_embedding with l > 0"
        )
        assert self.edge_nonlinear == 'so2_sigmoid_gate'
        self.scatter_norm = None

        self.linear_up = e3nnLinear(
            self.irreps_in,
            self.irreps_in,
            bias=self.use_bias,
        )    

        self.rejector = AttentionSO2TensorProduct(
            mmax=self.mmax,
            lmax=self.lmax,
            num_channel=self.num_channel,
            num_radial_basis=self.num_radial_basis,
            num_head=self.num_head,
            use_temperature=self.use_temperature,
            edge_wise_hidden=self.edge_wise_hidden,
            so2_linear_type=self.so2_linear_type,
            reshape_in=LayoutTransform(self.irreps_in),
            reshape_out=LayoutTransform(o3.Irreps([(self.edge_wise_hidden, ir) for _, ir in self.irreps_out])), 
        )

        irreps_node_wise_hidden = o3.Irreps([(self.node_wise_hidden, ir) for _, ir in self.irreps_out])

        if self.nonlinear_type == 'gate':  
            irreps_gated = irreps_node_wise_hidden
            irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in irreps_node_wise_hidden)
            self.nonlinearity = O3Gate(
                irreps_gates=irreps_gates,
                act_gates=[ACTIVATION[self.nonlinear_act]()] * len(irreps_gates),
                irreps_gated=irreps_gated,
            )
            linear_down_irreps_out = self.nonlinearity.irreps_in.simplify()
            self.linear_nonlinearity = e3nnLinear(
                irreps_node_wise_hidden, 
                self.irreps_out,  
                bias=self.use_bias,
            )
        elif self.nonlinear_type == 'norm':
            self.nonlinearity = O3Norm(
                irreps=irreps_node_wise_hidden,
                activation=[ACTIVATION[self.nonlinear_act]()],
            )
            linear_down_irreps_out = self.nonlinearity.irreps_in.simplify()
            self.linear_nonlinearity = e3nnLinear(
                irreps_node_wise_hidden, 
                self.irreps_out,  
                bias=self.use_bias,
            )
        else:
            self.nonlinearity = torch.nn.Identity()
            self.linear_nonlinearity = torch.nn.Identity()
            linear_down_irreps_out = irreps_node_wise_hidden

        self.linear_down = e3nnLinear(
            o3.Irreps([(self.edge_wise_hidden, ir) for _, ir in self.irreps_out]),
            # self.irreps_out,
            linear_down_irreps_out,
            bias=self.use_bias,
        )

        self.edge_info = FFN[self.edge_info_type](
            [self.edge_feats_channel] + self.radial_mlp + [self.rejector.weight_numel],
            bias=self.radial_bias,
            layer_norm=self.radial_layer_norm,
            act=self.radial_act,
        )

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type == 'BB':
            if self.resnet_linear_type == 'agnostic':
                self.resnetBB = e3nnLinear(
                    irreps_in=self.irreps_in,
                    irreps_out=self.irreps_sc,
                    bias=self.use_bias,
                )
            else:
                self.resnetBB = e3nnElementLinear(
                    irreps_in = self.irreps_in,
                    irreps_out = self.irreps_sc,
                    bias=self.use_bias,
                    num_elements=self.num_elements,
                )

        if (self.use_first_resnet or self.layer > 0) and self.resnet_type == 'BAB':
            if self.resnet_linear_type == 'agnostic':
                self.resnetBA = e3nnLinear(
                    irreps_in = self.irreps_in,
                    irreps_out = self.irreps_out,
                    bias=self.use_bias,
                )
                self.resnetAB = e3nnLinear(
                    irreps_in = self.irreps_out,
                    irreps_out = self.irreps_sc,
                    bias=self.use_bias,
                ) 
            else:
                self.resnetBA = e3nnElementLinear(
                    irreps_in = self.irreps_in,
                    irreps_out = self.irreps_out,
                    bias=self.use_bias,
                    num_elements=self.num_elements,
                )
                self.resnetAB = e3nnElementLinear(
                    irreps_in = self.irreps_out,
                    irreps_out = self.irreps_sc,
                    bias=self.use_bias,
                    num_elements=self.num_elements,
                ) 

        if (self.use_first_pre_norm or self.layer > 0) and self.pre_norm_type is not None:
            self.norm1 = get_normalization_layer(
                self.pre_norm_type,
                ls=self.irreps_in.lmax,
                num_channels=self.num_channel,
            )
            self.reshape1 = LayoutTransform(self.irreps_in)
            if self.resnet_type == "BAB":
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
    ):
    
        lmp_data = graph.lmp_data
        lmp_natoms = graph.lmp_natoms
        nlocal = lmp_natoms[0] if lmp_data is not None else None

        resBB = None
        resBA = None
        resAB = None

        if hasattr(self, 'resnetBB'):
            if self.resnet_linear_type == 'aware':
                resBB = self.resnetBB(node_feats, node_attrs_slice)
            else:
                resBB = self.resnetBB(node_feats) 

        if hasattr(self, 'resnetBA'):
            if self.resnet_linear_type == 'aware':
                resBA = self.resnetBA(node_feats, node_attrs_slice)
            else:
                resBA = self.resnetBA(node_feats)

        if hasattr(self, 'norm1'):
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
            nlocal
        )
    
        if resBA is not None:
            m_i = m_i + resBA

        if hasattr(self, 'resnetAB'):
            if self.resnet_linear_type == 'aware':
                resAB = self.resnetAB(m_i, node_attrs_slice)
            else:
                resAB = self.resnetAB(m_i)

        if hasattr(self, 'norm2'):
            m_i = self.reshape2.inverse(self.norm2(self.reshape2(m_i)))

        m_i = self.linear_down(m_i)
        m_i = self.linear_nonlinearity(self.nonlinearity(m_i))

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

    "uu_so2": uuSO2InteractionArchitecture1,

    "so2": uvSO2InteractionArchitecture2,
    "uv_so2": uvSO2InteractionArchitecture2,

    "attn": AttentionInteractionArchitecture3,

    # "w6j": Wigner6jInteraction,
    # "wigner6j": Wigner6jInteraction,

}