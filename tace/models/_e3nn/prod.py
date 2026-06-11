###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Dict


import torch
from e3nn import o3


from ..layout import LayoutTransform, LayoutTransform2
from ..linear import e3nnLinear, e3nnElementLinear, e3nnMoEElementLinear
from ..so2 import SO3Grid
from .base import Product
from .fused import uuuTensorProduct
from .dropout import GraphDropPath


# class CgtpACE(Product):
#     """
#     The most expressive ACE implementation based on Clebsch-Gordan tensor products.

#     This class computes all possible many-body tensor product paths and couples
#     all channels, forming a highly expressive product basis.

#     Note:
#         It is recommended to use no more than 64 channels, as increasing the
#         number of channels beyond this does not necessarily lead to better
#         performance and may introduce unnecessary computational overhead.
#     """

#     def _setup(self):

#         self.linear_up = e3nnLinear(
#             self.irreps_in,
#             self.irreps_hidden,
#             bias=self.use_bias,
#         ) if self.num_channel != self.num_hidden_channel else torch.nn.Identity()

#         for_coefs = {
#             "irreps_out": self.irreps_coefs_out,
#             "bias": self.use_bias,
#             "num_elements": self.num_elements,
#         }
#         coefs_cls = e3nnElementLinear
#         if self.num_expert > 1:
#             coefs_cls = e3nnMoEElementLinear
#             for_coefs["num_experts"] = self.num_expert

#         self.aces = torch.nn.ModuleList()
#         self.coefs = torch.nn.ModuleList()
#         self.coefs.append(
#             coefs_cls(
#                 o3.Irreps([(self.num_hidden_channel, ir) for _, ir in self.irreps_hidden]).simplify(),
#                 **for_coefs,
#             )
#         )

#         product_in1 = self.irreps_hidden

#         for nu in range(2, self.correlation+1):
#             this_ace = uuuTensorProduct(
#                 irreps_in1=product_in1,
#                 irreps_in2=self.irreps_hidden,
#                 irreps_out=self.irreps_tp_out_list[nu-2],
#                 l1l2=self.l1l2,
#             )
#             self.aces.append(this_ace)
#             self.coefs.append(coefs_cls(
#                 o3.Irreps([(self.num_hidden_channel, ir) for _, ir in this_ace.irreps_out]).simplify(), 
#                 **for_coefs,
#                 )
#             )
#             product_in1 = this_ace.irreps_out

#         if self.nonlinear_type == 'cwnorm':
#             from .nonlinear import ChannelWiseO3NormGate
#             from ..mlp import ScaledSigmoid
#             self.nonlinearty = ChannelWiseO3NormGate(
#                 for_coefs["irreps_out"],
#                 ScaledSigmoid(),
#             )

#         self.linear = e3nnLinear(
#             o3.Irreps([(self.num_hidden_channel, ir) for _, ir in self.irreps_coefs_out]),
#             self.irreps_out,
#             bias=self.use_bias,
#         )    

#         if (self.layer > 0 or self.use_first_dropout) and self.stochastic_depth_p > 0.0:
#             self.stochastic_depth = GraphDropPath(self.stochastic_depth_p) 

#     def forward(
#             self, 
#             node_feats: torch.Tensor, 
#             node_attrs: torch.Tensor,
#             sc: torch.Tensor,
#             batch: torch.Tensor,
#         ) -> torch.Tensor:

#         node_feats = self.linear_up(node_feats)

#         corr_feats = {
#             1: node_feats,
#         }

#         outs = self.coefs[0](corr_feats[1], node_attrs)

#         for nu in range(2, self.correlation+1):
#             corr_feats[nu] = self.aces[nu-2](corr_feats[nu-1], node_feats)
#             outs = outs + self.coefs[nu-1](corr_feats[nu], node_attrs)

#         if hasattr(self, "nonlinearty"):
#             outs = self.nonlinearty(outs)

#         outs = self.linear(outs)

#         if hasattr(self, "stochastic_depth"):
#             outs = self.stochastic_depth(outs, batch)
        
#         if sc is not None:
#             outs = outs + sc

#         return outs

    
class CgtpACE(Product):
    """
    The most expressive ACE implementation based on Clebsch-Gordan tensor products.

    This class computes all possible many-body tensor product paths and couples
    all channels, forming a highly expressive product basis.

    Note:
        It is recommended to use no more than 64 channels, as increasing the
        number of channels beyond this does not necessarily lead to better
        performance and may introduce unnecessary computational overhead.
    """

    def _setup(self):

        self.linear_up = e3nnLinear(
            self.irreps_in,
            self.irreps_hidden,
            bias=self.use_bias,
        ) if self.num_channel != self.num_hidden_channel else torch.nn.Identity()

        for_coefs = {
            "irreps_out": self.irreps_coefs_out,
            "bias": self.use_bias,
            "num_elements": self.num_elements,
        }
        coefs_cls = e3nnElementLinear
        if self.num_expert > 1:
            coefs_cls = e3nnMoEElementLinear
            for_coefs["num_experts"] = self.num_expert
            if self.use_softmax:
                self.router = e3nnLinear(
                    self.irreps_hidden,
                    o3.Irreps([(self.num_expert, o3.Irrep("0e"))]),
                    # bias=False,
                    bias=True,
                )
    
                with torch.no_grad():
                    if isinstance(self.router.weight, torch.nn.ParameterList):
                        for p in self.router.weight:
                            p.mul_(1e-2)
                    else:
                        self.router.weight.mul_(1e-2)
                self._routed_irreps = self.irreps_coefs_out

        self.aces = torch.nn.ModuleList()
        self.coefs = torch.nn.ModuleList()
        self.coefs.append(
            coefs_cls(
                o3.Irreps([(self.num_hidden_channel, ir) for _, ir in self.irreps_hidden]).simplify(),
                **for_coefs,
            )
        )

        product_in1 = self.irreps_hidden

        for nu in range(2, self.correlation+1):
            this_ace = uuuTensorProduct(
                irreps_in1=product_in1,
                irreps_in2=self.irreps_hidden,
                irreps_out=self.irreps_tp_out_list[nu-2],
                l1l2=self.l1l2,
            )
            self.aces.append(this_ace)
            self.coefs.append(coefs_cls(
                o3.Irreps([(self.num_hidden_channel, ir) for _, ir in this_ace.irreps_out]).simplify(), 
                **for_coefs,
                )
            )
            product_in1 = this_ace.irreps_out

        if self.nonlinear_type == 'cwnorm':
            from .nonlinear import ChannelWiseO3NormGate
            from ..mlp import ScaledSigmoid
            self.nonlinearty = ChannelWiseO3NormGate(
                for_coefs["irreps_out"],
                ScaledSigmoid(),
            )

        self.linear = e3nnLinear(
            o3.Irreps([(self.num_hidden_channel, ir) for _, ir in self.irreps_coefs_out]),
            self.irreps_out,
            bias=self.use_bias,
        )    

        if (self.layer > 0 or self.use_first_dropout) and self.stochastic_depth_p > 0.0:
            self.stochastic_depth = GraphDropPath(self.stochastic_depth_p) 

    def _softmax_router_weights(self, node_feats: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(self.router(node_feats), dim=-1)
        # print(probabilities[0])
        rms = probabilities.square().mean(dim=-1, keepdim=True).sqrt()
        return probabilities / rms

    def _route_experts(
        self,
        x: torch.Tensor,
        router_weights: torch.Tensor,
    ) -> torch.Tensor:
        routed_fields = []
        for tensor_slice, (mul, ir) in zip(
            self._routed_irreps.slices(),
            self._routed_irreps,
        ):
            expert_mul = mul // self.num_expert
            field = x[:, tensor_slice].reshape(
                x.shape[0],
                self.num_expert,
                expert_mul,
                ir.dim,
            )
            routed_fields.append(
                (field * router_weights[:, :, None, None]).reshape(x.shape[0], -1)
            )
        return torch.cat(routed_fields, dim=-1)

    def forward(
            self, 
            node_feats: torch.Tensor, 
            node_attrs: torch.Tensor,
            sc: torch.Tensor,
            batch: torch.Tensor,
        ) -> torch.Tensor:

        node_feats = self.linear_up(node_feats)

        router_weights = (
            self._softmax_router_weights(node_feats)
            if hasattr(self, "router")
            else None
        )

        corr_feats = {
            1: node_feats,
        }

        outs = self.coefs[0](corr_feats[1], node_attrs)

        for nu in range(2, self.correlation+1):
            corr_feats[nu] = self.aces[nu-2](corr_feats[nu-1], node_feats)
            outs = outs + self.coefs[nu-1](corr_feats[nu], node_attrs)

        if hasattr(self, "nonlinearty"):
            outs = self.nonlinearty(outs)

        if router_weights is not None:
            outs = self._route_experts(outs, router_weights)

        outs = self.linear(outs)

        if hasattr(self, "stochastic_depth"):
            outs = self.stochastic_depth(outs, batch)
        
        if sc is not None:
            outs = outs + sc

        return outs
    
# TODO, refactor
class GtpACE(Product):
    """
    An ACE implementation based on Gaunt tensor products.

    This module uses Gaunt tensor products to perform many-body expansions.
    However, this approach introduces equivariance errors (though typically small),
    lacks antisymmetric interactions, and averages over multiple many-body
    expansion paths.

    As a result, increasing the correlation order does not always lead to improved
    accuracy. 

    In practice, the grid-processing operation can be fused with the linear layer. 
    However, considering modules such as LoRA, we do not perform such fusion for the sake of 
    simplicity and flexibility.
    """

    def _setup(self):

        assert self.parity == False, "GtpACE not support O(3) group now"
        assert self.num_expert == 1

        self.linear_up = e3nnLinear(
            self.irreps_in,
            self.irreps_hidden,
            bias=self.use_bias,
        ) if self.num_channel != self.num_hidden_channel else torch.nn.Identity()

        self.reshape1 = LayoutTransform(self.irreps_hidden)
 
        self.grid = SO3Grid(
            lmax=self.irreps_in.lmax,
            mmax=self.irreps_in.lmax,
            resolution_list=self.resolution,
            use_m_primary=False,
        )

        for_coefs = {
            "irreps_in": self.irreps_hidden,
            "irreps_out": self.irreps_coefs_out,
            "bias": self.use_bias,
            "num_elements": self.num_elements,
        }
        coefs_cls = e3nnElementLinear
        self.coefs = torch.nn.ModuleList()
        for _ in range(1, self.correlation+1):
            self.coefs.append(coefs_cls(**for_coefs))

        self.linear = e3nnLinear(
            self.irreps_coefs_out,
            self.irreps_out,
            bias=self.use_bias
        )

        if (self.layer > 0 or self.use_first_dropout) and self.stochastic_depth_p > 0.0:
            self.stochastic_depth = GraphDropPath(self.stochastic_depth_p) 

    def forward(
            self, 
            node_feats: torch.Tensor, 
            node_attrs: torch.Tensor,
            sc: torch.Tensor,
            batch: torch.Tensor,
        ) -> torch.Tensor:

        node_feats = self.linear_up(node_feats)
        
        outs = self.coefs[0](node_feats, node_attrs)
        node_feats = self.reshape1(node_feats)
        base_grid = self.grid.to_grid(node_feats)

        corr_feats_list = []
        grid_prev = base_grid
        for nu in range(2, self.correlation + 1):
            grid_prev = grid_prev * base_grid
            corr_feats_list.append(grid_prev)

        for nu in range(2, self.correlation + 1):
            this_corr_feats = self.reshape1.inverse(self.grid.from_grid(corr_feats_list[nu-2]))
            outs = outs + self.coefs[nu-1](this_corr_feats, node_attrs)
           
        outs = self.linear(outs)

        if hasattr(self, "stochastic_depth"):
            outs = self.stochastic_depth(outs, batch)

        if sc is not None:
            outs = outs + sc

        return outs   

# TODO, refactor
class MACE(Product):
    """
    An ACE implementation from MACE.
    https://github.com/ACEsuit/mace
    """
    def _setup(self):

        assert self.num_expert == 1

        self.linear_up = e3nnLinear(
            self.irreps_in,
            self.irreps_hidden,
            bias=self.use_bias,
        ) if self.num_channel != self.num_hidden_channel else torch.nn.Identity()

        self.reshape = LayoutTransform2(self.irreps_hidden if self.num_channel != self.num_hidden_channel else self.irreps_in)

        from tace.utils.env import get_tace_use_cue
        from .symmetric_contraction import SymmetricContractionWrapper
        self.use_cueq = get_tace_use_cue == '1'
        self.symmetric_contractions = SymmetricContractionWrapper(
            irreps_in=self.irreps_hidden,
            irreps_out=self.irreps_coefs_out,
            correlation=self.correlation,
            num_elements=self.num_elements,
            use_reduced_cg=True,
            use_cueq=self.use_cueq,
        )

        self.linear = e3nnLinear(
            self.irreps_coefs_out,
            self.irreps_out,
            bias=self.use_bias
        )    

        if (self.layer > 0 or self.use_first_dropout) and self.stochastic_depth_p > 0.0:
            self.stochastic_depth = GraphDropPath(self.stochastic_depth_p) 


    def forward(
        self, 
        node_feats: torch.Tensor, 
        node_attrs: torch.Tensor,
        sc: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:

        node_feats = self.linear_up(node_feats)

        node_feats = self.reshape(node_feats)

        if self.use_cueq:
            node_feats = torch.transpose(node_feats, 1, 2)
            index_attrs = node_attrs.argmax(dim=-1).int()
            outs = self.symmetric_contractions(
                node_feats.flatten(1),
                index_attrs,
            )
        else:
            outs = self.symmetric_contractions(node_feats, node_attrs)

        outs = self.linear(outs)
        
        if hasattr(self, "stochastic_depth"):
            outs = self.stochastic_depth(outs, batch)
        
        if sc is not None:
            outs = outs + sc

        return outs


PRODUCT: Dict[str, torch.nn.Module] = {
    "spatial": CgtpACE,
    "coupled": CgtpACE,
    "cgtp": CgtpACE,
    "glu": CgtpACE,

    "spectral": GtpACE,
    "grid": GtpACE,
    "gtp": GtpACE,

    "mace": MACE,

    # "so2": So2ACE,

    # "vstp": VstpACE,
}
