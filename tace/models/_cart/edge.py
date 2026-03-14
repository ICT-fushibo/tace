################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


from typing import Optional


import torch


from .base import EdgeEmbedding
from .linear import Linear


class IdentityEdgeEmbedding(EdgeEmbedding):
    
    def _setup(self) -> None:

        self.out_dim = self.num_radial_basis

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        return edge_feats


class LinearElementEdgeEmbedding(EdgeEmbedding):

    def _setup(self) -> None:

        self.out_dim = self.num_channel

        self.elem_emb1 = Linear(
            [0],
            self.num_elements,
            self.num_channel,
            bias=self.bias,
        )
        self.elem_emb2 = Linear(
            [0],
            self.num_elements,
            self.num_channel,
            bias=self.bias,
        )
        self.radial_proj = Linear(
            [0],
            self.num_radial_basis,
            self.num_channel,
            bias=self.bias,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        node_attrs = node_attrs
        source, target = edge_index
        
        x_j = self.elem_emb2({0: node_attrs})[source][0]
        x_i = self.elem_emb1({0: node_attrs})[target][0]
        w_ij = self.radial_proj({0: edge_feats})[0]
    
        return (x_i + x_j) * w_ij
    

EDGE_EMBEDDING = {
    "identity": IdentityEdgeEmbedding,
    "linear": LinearElementEdgeEmbedding,
}