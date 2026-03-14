################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import abc
from typing import Optional


import torch
from torch_scatter import scatter_sum


from .base import NodeEmbedding
from .linear import Linear


class LinearNodeEmbedding(NodeEmbedding):
    
    def _setup(self) -> None:
        self.elem_emb1 = Linear(
            "0e",
            "0e",
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
    ) -> torch.Tensor:
        
        node_attrs = node_attrs.unsqueeze(1)

        return self.elem_emb1(node_attrs)

class NonlinearNodeEmbedding(NodeEmbedding):
    
    def _setup(self) -> None:
        self.elem_emb1 = Linear(
            "0e",
            "0e",
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
    ) -> torch.Tensor:
        
        node_attrs = node_attrs.unsqueeze(1)

        return self.act1(self.elem_emb1(node_attrs))
    
class GroupNodeEmbedding(NodeEmbedding):
    
    def _setup(self) -> None:
        
        self.num_groups = 32

        self.elem_emb1 = Linear(
            "0e",
            "0e",
            self.num_elements,
            self.num_channel,
            bias=self.bias,
        )

        self.elem_emb2 = Linear(
            "0e",
            "0e",
            self.num_elements,
            self.num_groups,
            bias=self.bias,
        )

        self.group_emb1 = Linear(
            "0e",
            "0e",
            self.num_groups,
            self.num_channel,
            bias=self.bias,
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
       
        node_attrs = node_attrs.unsqueeze(1)      

        elem_emb = self.elem_emb1(node_attrs)     
        elem_emb = self.act1(elem_emb)        

        logits = self.elem_emb2(node_attrs)   
        scores = self.act2(logits)     
        group_emb = self.group_emb1(scores)

        return elem_emb + group_emb
    

class SurroundingNodeEmbedding(NodeEmbedding):

    def _setup(self) -> None:
        self.elem_emb1 = Linear(
            "0e",
            "0e",
            self.num_elements,
            self.num_channel,
            bias=self.bias,
        )
        self.elem_emb2 = Linear(
            "0e",
            "0e",
            self.num_elements,
            self.num_channel,
            bias=self.bias,
        )
        self.radial_proj = Linear(
            "0e",
            "0e",
            self.num_radial_basis,
            self.num_channel,
            bias=self.bias,
        )
        self.mix = Linear(  
            "0e",
            "0e",
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
        node_attrs = node_attrs.unsqueeze(1)
        edge_feats = node_attrs.unsqueeze(1)

        n_i = self.elem_emb1(node_attrs)
        e_ij = self.elem_emb2(node_attrs)[source]
        w_ij = self.radial_proj(edge_feats)
        if cutoff is not None:
            w_ij = w_ij * cutoff.unsqueeze(1)
        m_ij = e_ij * w_ij
        m_i = scatter_sum(
            m_ij,
            target,
            dim=0,
            dim_size=n_i.shape[0]
        )

        return self.mix(
            torch.cat([n_i.squeeze(1), m_i.squeeze(1)], dim=-1)
        )
    

NODE_EMBEDDING = {
    "linear": LinearNodeEmbedding,
    "nonlinear": NonlinearNodeEmbedding,
    "group": GroupNodeEmbedding,
    "surrounding": SurroundingNodeEmbedding,
}