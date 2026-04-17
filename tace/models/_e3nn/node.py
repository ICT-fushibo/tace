################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import torch
from e3nn.nn import Activation
from e3nn import o3


from .base import NodeEmbedding
from .linear import Linear


class LinearNodeEmbedding(NodeEmbedding):
    """
    A simple node embedding module based on a linear transformation.

    This class projects discrete node attributes (e.g., element types)
    into a continuous feature space using a single linear layer,
    without introducing nonlinearity or structural information.
    """

    def _setup(self) -> None:

        self.irreps_out = o3.Irreps(f"{self.num_channel}x0e")

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
        edge_vector: torch.Tensor,
        cutoff: torch.Tensor
    ) -> torch.Tensor:
        
        return self.elem_emb1(node_attrs)


class NonLinearNodeEmbedding(NodeEmbedding):
    """
    A node embedding module with nonlinear transformation.

    This class applies a nonlinear activation function after a linear
    projection. 
    """

    def _setup(self) -> None:

        self.irreps_out = o3.Irreps(f"{self.num_channel}x0e")

        self.elem_emb1 = Linear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias,
        )
        self.act1 = Activation(self.elem_emb1.irreps_out, [torch.nn.SiLU()])

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vector: torch.Tensor,
        cutoff: torch.Tensor
    ) -> torch.Tensor:
        
        return self.act1(self.elem_emb1(node_attrs))
    

class GroupNodeEmbedding(NodeEmbedding):
    """
    A group-based node embedding module.

    This class augments basic element embeddings with a learned grouping
    mechanism, where nodes are softly assigned to 32 latent groups and
    group-level representations are combined with element-wise features
    to enhance expressiveness.
    """

    def _setup(self) -> None:
        
        self.irreps_out = o3.Irreps(f"{self.num_channel}x0e")

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

        self.act1 = Activation(self.elem_emb1.irreps_out, [torch.nn.SiLU()])
        self.act2 = torch.nn.Softmax(dim=-1)


    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vector: torch.Tensor,
        cutoff: torch.Tensor
    ) -> torch.Tensor:
         
        elem_emb = self.elem_emb1(node_attrs)     
        elem_emb = self.act1(elem_emb)        

        logits = self.elem_emb2(node_attrs)   
        scores = self.act2(logits)      
        group_emb = self.group_emb1(scores)

        return elem_emb + group_emb
    
NODE_EMBEDDING = {
    "linear": LinearNodeEmbedding,
    "nonlinear": NonLinearNodeEmbedding,
    "group": GroupNodeEmbedding,
}