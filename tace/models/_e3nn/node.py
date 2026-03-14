################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Optional


import torch
from torch_scatter import scatter_sum


from .base import NodeEmbedding
from .linear import Linear


class LinearNodeEmbedding(NodeEmbedding):
    
    def _setup(self) -> None:

        self.elem_emb1 = Linear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor
    ) -> torch.Tensor:
        
        return self.elem_emb1(node_attrs)


class NonlinearNodeEmbedding(NodeEmbedding):
    
    def _setup(self) -> None:

        self.elem_emb1 = Linear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias,
        )
        self.act1 = torch.nn.SiLU()

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor
    ) -> torch.Tensor:
        
        return self.act1(self.elem_emb1(node_attrs))
    

class GroupNodeEmbedding(NodeEmbedding):
    
    def _setup(self) -> None:
        
        self.num_groups = 32

        self.elem_emb1 = Linear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias
        )

        self.elem_emb2 = Linear(
            f"{self.num_elements}x0e",
            f"{self.num_groups}x0e",
            bias=self.bias
        )

        self.group_emb1 = Linear(
            f"{self.num_groups}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias
        )

        self.act1 = torch.nn.SiLU()
        self.act2 = torch.nn.Softmax(dim=-1)


    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor
    ) -> torch.Tensor:
         

        elem_emb = self.elem_emb1(node_attrs)     
        elem_emb = self.act1(elem_emb)        

        logits = self.elem_emb2(node_attrs)   
        scores = self.act2(logits)      
        group_emb = self.group_emb1(scores)

        return elem_emb + group_emb
    

class SurroundingNodeEmbedding(NodeEmbedding):

    def _setup(self) -> None:

        self.elem_emb1 = Linear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias,
        )
        self.elem_emb2 = Linear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias,
        )
        self.radial_proj = Linear(
            f"{self.num_radial_basis}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias,
        )
        self.mix = Linear(  
            f"{self.num_channel * 2}x0e",
            f"{self.num_channel}x0e",
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
        n_i = self.elem_emb1(node_attrs)
        e_ij = self.elem_emb2(node_attrs)[source]
        w_ij = self.radial_proj(edge_feats)
        if cutoff is not None:
            w_ij = w_ij * cutoff
        m_ij = e_ij * w_ij
        m_i = scatter_sum(
            m_ij,
            target,
            dim=0,
            dim_size=n_i.shape[0]
        )

        return self.mix(
            torch.cat([n_i, m_i], dim=-1)
        )
    


NODE_EMBEDDING = {
    "linear": LinearNodeEmbedding,
    "nonlinear": NonlinearNodeEmbedding,
    "group": GroupNodeEmbedding,
    "surrounding": SurroundingNodeEmbedding,
}