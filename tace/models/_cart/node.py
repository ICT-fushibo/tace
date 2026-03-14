################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional, Dict


import torch
from torch_scatter import scatter_sum


from .base import NodeEmbedding
from .linear import Linear


class LinearNodeEmbedding(NodeEmbedding):
    
    def _setup(self) -> None:
        self.elem_emb1 = Linear(
            [0],
            self.num_elements,
            self.num_channel,
            bias=self.bias,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        
        return self.elem_emb1({0:node_attrs})

class NonlinearNodeEmbedding(NodeEmbedding):
    
    def _setup(self) -> None:
        self.elem_emb1 = Linear(
            [0],
            self.num_elements,
            self.num_channel,
            bias=self.bias,
        )
        self.act1 = torch.nn.SiLU()

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        
        return {0: self.act1(self.elem_emb1({0:node_attrs})[0])}
    
class SurroundingNodeEmbedding(NodeEmbedding):

    def _setup(self) -> None:
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
        self.mix = Linear(  
            [0],
            self.num_channel * 2,
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

        source, target = edge_index

        n_i = self.elem_emb1({0: node_attrs})
        e_ij = self.elem_emb2({0: node_attrs})[source]
        w_ij = self.radial_proj({0: edge_feats})

        n_i = n_i[0]
        e_ij = e_ij[0]
        W_ij = w_ij[0]
        if cutoff is not None:
            w_ij = w_ij * cutoff
            
        m_ij = e_ij * w_ij
        m_i = scatter_sum(
            m_ij,
            target,
            dim=0,
            dim_size=n_i.size(0)
        )

        return self.mix(
            {0: torch.cat([n_i, m_i], dim=-1)}
        )
    

NODE_EMBEDDING = {
    "linear": LinearNodeEmbedding,
    "nonlinear": NonlinearNodeEmbedding,
    # "group": GroupNodeEmbedding,
    "surrounding": SurroundingNodeEmbedding,
}