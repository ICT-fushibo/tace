################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


from typing import Optional


import torch
from e3nn.nn import Activation
from e3nn import o3


from .base import EdgeEmbedding, EdgeUpdate
from .linear import Linear
from ...utils.torch_scatter import scatter_sum


class IdentityEdgeEmbedding(EdgeEmbedding):
    
    def _setup(self) -> None:

        self.out_dim = self.num_radial_basis

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None,
    ) -> torch.Tensor:
        
        return edge_feats


class LinearEdgeEmbedding(EdgeEmbedding):

    def _setup(self) -> None:

        self.out_dim = self.num_channel

        self.radial_proj = Linear(
            f"{self.num_radial_basis}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None,
    ) -> torch.Tensor:
        
        return self.radial_proj(edge_feats)


class NonlinearEdgeEmbedding(EdgeEmbedding):

    def _setup(self) -> None:

        self.out_dim = self.num_channel

        self.radial_proj = Linear(
            f"{self.num_radial_basis}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias,
        )

        self.act1 = Activation(self.radial_proj.irreps_out, [torch.nn.SiLU()])

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None,
    ) -> torch.Tensor:
        
        return self.act1(self.radial_proj(edge_feats)) # / 1.6791767923989418


class NonlinearElementEdgeEmbedding(EdgeEmbedding):

    def _setup(self) -> None:

        self.out_dim = self.num_channel * 3

        self.radial_proj = Linear(
            f"{self.num_radial_basis}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias,
        )
        self.source_embedding = Linear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias,
        )
        self.target_embedding = Linear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias,
        )
        self.act1 = Activation(self.radial_proj.irreps_out, [torch.nn.SiLU()])
        torch.nn.init.uniform_(self.source_embedding.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.target_embedding.weight, a=-0.001, b=0.001)

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None,
    ) -> torch.Tensor:
        
        assert cutoff is not None, "Please set radial_basis.apply_cutoff = False"
        
        source, target = edge_index
        x_j = self.source_embedding(node_attrs)[source]
        x_i = self.target_embedding(node_attrs)[target]
        edge_feats = self.radial_proj(edge_feats)
        edge_feats = self.act1(edge_feats)
  
        return torch.cat([edge_feats, x_i, x_j], dim=-1)


class GroupEdgeEmbedding(EdgeEmbedding):
    
    def _setup(self) -> None:
        
        self.out_dim = self.num_channel * 3

        self.num_groups = 32

        self.radial_proj =Linear(
            f"{self.num_radial_basis}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias
        )

        self.source_elem_emb1 = Linear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias
        )

        self.source_elem_emb2 = Linear(
            f"{self.num_elements}x0e",
            f"{self.num_groups}x0e",
            bias=self.bias
        )


        self.source_group_emb1 = Linear(
            f"{self.num_groups}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias
        )

        self.target_elem_emb1 = Linear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias
        )

        self.target_elem_emb2 = Linear(
            f"{self.num_elements}x0e",
            f"{self.num_groups}x0e",
            bias=self.bias
        )

        self.target_group_emb1 = Linear(
            f"{self.num_groups}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias
        )

        torch.nn.init.uniform_(self.source_elem_emb1.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.source_elem_emb2.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.source_group_emb1.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.target_elem_emb1.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.target_elem_emb2.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.target_group_emb1.weight, a=-0.001, b=0.001)

        self.act1 = Activation(self.radial_proj.irreps_out, [torch.nn.SiLU()])
        self.act2 = torch.nn.Softmax(dim=-1)


    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None,
    ) -> torch.Tensor:
        
        assert cutoff is not None, "Please set radial_basis.apply_cutoff = False"
         
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

        edge_feats = self.act1(self.radial_proj(edge_feats))
        return torch.cat([edge_feats, source_embedding[edge_index[0]], target_embedding[edge_index[1]]], dim=-1)
    

class IdentityEdgeUpdate(EdgeUpdate):
    
    def _setup(self) -> None:

        self.out_dim = self.edge_embedding_channel

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None,
    ) -> torch.Tensor:
        
        return edge_feats
    

class ElementEdgeUpdate(EdgeUpdate):
    
    def _setup(self) -> None:

        self.out_dim = self.edge_embedding_channel + self.num_channel * 2

        self.source_embedding = Linear(
            f'{self.num_elements}x0e',
            f'{self.num_channel}x0e',
            bias=self.use_bias,
        )
        self.target_embedding = Linear(
            f'{self.num_elements}x0e',
            f'{self.num_channel}x0e',
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
        cutoff: torch.Tensor | None,
    ) -> torch.Tensor:
        
        edge_feats_list = [edge_feats]
        edge_feats_list.append(
            self.source_embedding(node_attrs[edge_index[0]])
        )

        edge_feats_list.append(
            self.target_embedding(node_attrs[edge_index[1]])
        )
        return torch.cat(edge_feats_list, dim=-1)


class Element2EdgeUpdate(ElementEdgeUpdate):
    
    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None,
    ) -> torch.Tensor:
        
        edge_feats_list = [edge_feats]

        edge_feats_list.append(
            self.target_embedding(node_attrs[edge_index[1]])
        )

        edge_feats_list.append(
            self.source_embedding(node_attrs[edge_index[0]])
        )

        return torch.cat(edge_feats_list, dim=-1)


class TensorDotEdgeUpdate(EdgeUpdate):
    
    def _setup(self) -> None:

        c = 8
        irreps_hidden = o3.Irreps([(c, ir) for _, ir in self.irreps_node])

        if self.layer == 0:
            self.out_dim = self.edge_embedding_channel       
        else:
            self.out_dim = self.edge_embedding_channel + c * len(self.irreps_node)

        if self.layer > 0:
            self.linear_q = Linear(
                self.irreps_node,
                irreps_hidden
            )
            self.linear_k = Linear(
                self.irreps_node,
                irreps_hidden
            )
            self.dot = o3.TensorProduct(
                irreps_in1=irreps_hidden, 
                irreps_in2=irreps_hidden, 
                irreps_out=[(c, (0, 1)) for _, _ in self.irreps_node], 
                instructions=[
                    (i, i, 0, 'uuu', False)
                    for i, (mul, ir) in enumerate(self.irreps_node)
                ],
                shared_weights=False,
                internal_weights=False,
            )

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None,
    ) -> torch.Tensor:
        
        edge_feats_list = [edge_feats]
        if self.layer > 0:
            Q = self.linear_q(node_feats)
            K = self.linear_k(node_feats)
            edge_feats_list.append(
                self.dot(Q[edge_index[1]], K[edge_index[0]])
            )
        return torch.cat(edge_feats_list, dim=-1)
        


EDGE_EMBEDDING = {
    "identity": IdentityEdgeEmbedding,
    "linear": LinearEdgeEmbedding,
    "nonlinear": NonlinearEdgeEmbedding,
    "group": GroupEdgeEmbedding,
    "element": NonlinearElementEdgeEmbedding,
}

EDGE_UPDATE = {
    "identity": IdentityEdgeUpdate,
    "element": ElementEdgeUpdate,
    "element2": Element2EdgeUpdate,
    "tensor_dot": TensorDotEdgeUpdate,
}



