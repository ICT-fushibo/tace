################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


from typing import Optional


import torch


from .base import EdgeEmbedding, EdgeUpdate
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

class LinearEdgeEmbedding(EdgeEmbedding):

    def _setup(self) -> None:

        self.out_dim = self.num_channel

        self.radial_proj = Linear(
            "0e",
            "0e",
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
        
        return self.radial_proj(edge_feats.unsqueeze(1)).squeeze(1)


class NonlinearEdgeEmbedding(EdgeEmbedding):

    def _setup(self) -> None:

        self.out_dim = self.num_channel

        self.radial_proj = Linear(
            "0e",
            "0e",
            self.num_radial_basis,
            self.num_channel,
            bias=self.bias,
        )
        self.act1 = torch.nn.SiLU()

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        return self.act1(self.radial_proj(edge_feats.unsqueeze(1)).squeeze(1))


class ElementEdgeEmbedding(EdgeEmbedding):

    def _setup(self) -> None:

        self.out_dim = self.num_channel * 3

        self.radial_proj = Linear(
            "0e",
            "0e",
            self.num_radial_basis,
            self.num_channel,
            bias=self.bias,
        )
        self.source_embedding = Linear(
            "0e",
            "0e",
            self.num_elements,
            self.num_channel,
            bias=self.bias,
        )
        self.target_embedding = Linear(
            "0e",
            "0e",
            self.num_elements,
            self.num_channel,
            bias=self.bias,
        )
        self.act1 = torch.nn.SiLU()
        torch.nn.init.uniform_(self.source_embedding.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.target_embedding.weight, a=-0.001, b=0.001)

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        assert cutoff is not None, "Please set radial_basis.apply_cutoff = False"
        
        node_attrs = node_attrs.unsqueeze(1) 
        x_j = self.source_embedding(node_attrs)
        x_i = self.target_embedding(node_attrs)
        edge_feats = self.radial_proj(edge_feats.unsqueeze(1))
  
        return self.act1(torch.cat([edge_feats, x_i[edge_index[1]], x_j[edge_index[0]]], dim=-1).squeeze(1))
    

class GroupEdgeEmbedding(EdgeEmbedding):
    
    def _setup(self) -> None:
        
        self.out_dim = self.num_channel * 3

        self.num_groups = 32

        self.radial_proj = Linear(
            "0e",
            "0e",
            self.num_radial_basis,
            self.num_channel,
            bias=self.bias,
        )

        self.source_elem_emb1 = Linear(
            "0e",
            "0e",
            self.num_elements,
            self.num_channel,
            bias=self.bias
        )

        self.source_elem_emb2 = Linear(
            "0e",
            "0e",
            self.num_elements,
            self.num_groups,
            bias=self.bias
        )

        self.source_group_emb1 = Linear(
            "0e",
            "0e",
            self.num_groups,
            self.num_channel,
            bias=self.bias
        )

        self.target_elem_emb1 = Linear(
            "0e",
            "0e",
            self.num_elements,
            self.num_channel,
            bias=self.bias
        )

        self.target_elem_emb2 = Linear(
            "0e",
            "0e",
            self.num_elements,
            self.num_groups,
            bias=self.bias
        )

        self.target_group_emb1 = Linear(
            "0e",
            "0e",
            self.num_groups,
            self.num_channel,
            bias=self.bias
        )

        torch.nn.init.uniform_(self.source_elem_emb1.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.source_elem_emb2.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.source_group_emb1.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.target_elem_emb1.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.target_elem_emb2.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.target_group_emb1.weight, a=-0.001, b=0.001)

        self.act1 = torch.nn.SiLU()
        self.act2 = torch.nn.Softmax(dim=-1)


    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        assert cutoff is not None, "Please set radial_basis.apply_cutoff = False"
        node_attrs = node_attrs.unsqueeze(1)
         
        source_elem_emb = self.source_elem_emb1(node_attrs)     
        source_elem_emb = self.act1(source_elem_emb)        
        source_logits = self.source_elem_emb2(node_attrs)   
        source_scores = self.act2(source_logits)      
        source_group_emb = self.source_group_emb1(source_scores)
        source_embedding = source_elem_emb + source_group_emb

        target_elem_emb = self.target_elem_emb1(node_attrs)     
        target_elem_emb = self.act1(target_elem_emb)        
        target_logits = self.target_elem_emb2(node_attrs)   
        target_scores = self.act2(target_logits)      
        target_group_emb = self.target_group_emb1(target_scores)
        target_embedding = target_elem_emb + target_group_emb

        edge_feats = self.act1(self.radial_proj(edge_feats.unsqueeze(1)))
        return torch.cat([edge_feats, source_embedding[edge_index[0]], target_embedding[edge_index[1]]], dim=-1).squeeze(1)


class IdentityEdgeUpdate(EdgeUpdate):
    
    def _setup(self) -> None:

        self.out_dim = self.num_radial_basis

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        
        return edge_feats
    

class ElementEdgeUpdate(EdgeUpdate):
    
    def _setup(self) -> None:

        self.out_dim = self.edge_embedding_channel + self.num_channel * 2

        self.source_embedding = Linear(
            '0e',
            '0e',
            self.num_elements,
            self.num_channel,
            bias=self.use_bias,
        )
        self.target_embedding = Linear(
            '0e',
            '0e',
            self.num_elements,
            self.num_channel,
            bias=self.use_bias,
        )
        torch.nn.init.uniform_(self.source_embedding.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.target_embedding.weight, a=-0.001, b=0.001)

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        edge_feats_list = [edge_feats]
        edge_feats_list.append(
            self.source_embedding(node_attrs.unsqueeze(1)[edge_index[0]]).squeeze(1)
        )
        edge_feats_list.append(
            self.target_embedding(node_attrs.unsqueeze(1)[edge_index[1]]).squeeze(1)
        )
        return torch.cat(edge_feats_list, dim=-1)


EDGE_EMBEDDING = {
    "identity": IdentityEdgeEmbedding,
    "linear": LinearEdgeEmbedding,
    "nonlinear": NonlinearEdgeEmbedding,
    "group": GroupEdgeEmbedding,
    "element": ElementEdgeEmbedding,
}

EDGE_UPDATE = {
    "identity": IdentityEdgeUpdate,
    "element": ElementEdgeUpdate,
}