###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict


import torch


from ..layout import LayoutTransform, LayoutTransform2
from ..so2 import SO3Grid
from ..linear import e3nnLinear, e3nnElementLinear
from .base import Product
from .fused import uuuTensorProduct
from .dropout import GraphDropPath

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
            
        self.aces = torch.nn.ModuleList()
        self.coefs = torch.nn.ModuleList()
        self.coefs.append(coefs_cls(self.irreps_hidden, **for_coefs))

        product_in1 = self.irreps_hidden

        for nu in range(2, self.correlation+1):
            this_ace = uuuTensorProduct(
                irreps_in1=product_in1,
                irreps_in2=self.irreps_hidden,
                irreps_out=self.irreps_tp_out_list[nu-2],
                l1l2=self.l1l2,
            )
            self.aces.append(this_ace)
            self.coefs.append(coefs_cls(this_ace.irreps_out.simplify(), **for_coefs))
            product_in1 = this_ace.irreps_out

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

        corr_feats = {
            1: node_feats,
        }

        outs = self.coefs[0](corr_feats[1], node_attrs)

        for nu in range(2, self.correlation+1):
            corr_feats[nu] = self.aces[nu-2](corr_feats[nu-1], node_feats)
            outs = outs + self.coefs[nu-1](corr_feats[nu], node_attrs)

        # if hasattr(self, "nonlinearity"):
        #     outs = self.nonlinearity(outs, node_attrs)

        outs = self.linear(outs)
        
        if hasattr(self, "stochastic_depth"):
            outs = self.stochastic_depth(outs, batch)
        
        if sc is not None:
            outs = outs + sc

        return outs


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


class MACE(Product):
    def _setup(self):

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


class OamACE(Product):
    def _setup(self):

        self.linear_up = e3nnLinear(
            self.irreps_in,
            self.irreps_hidden,
            bias=self.use_bias,
        ) if self.num_channel != self.num_hidden_channel else torch.nn.Identity()

        self.ace = uuuTensorProduct(
            irreps_in1=self.irreps_hidden,
            irreps_in2=self.irreps_hidden[:1] + self.irreps_hidden,
            irreps_out=self.irreps_coefs_out,
            l1l2=self.l1l2,
            trainable=True,
        ) 
        self.coef = torch.nn.Parameter(torch.randn(self.num_elements, self.ace.weight_numel))

        self.linear = e3nnLinear(
            self.ace.irreps_out.simplify(),
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
        ones = node_feats.new_ones(node_feats.size(0), self.num_hidden_channel)

        outs = self.ace(
            node_feats, 
            torch.cat(
                [
                    ones,
                    node_feats,
                ],
                dim=-1,
            ), 
            torch.einsum('bz, zi -> bi', node_attrs, self.coef),
        )

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

    "oam": OamACE,

}
