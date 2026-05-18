################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import torch
from e3nn.nn import Activation
from e3nn import o3


from tace.utils.torch_scatter import scatter_sum
from ..layout import LayoutTransform
from ..mlp import MLP
from .base import NodeEmbedding
from ..linear import e3nnLinear
from .fused import O3ScatterTensorProduct


class LinearNodeEmbedding(NodeEmbedding):
    """
    A simple node embedding module based on a linear transformation.

    This class projects discrete node attributes (e.g., element types)
    into a continuous feature space using a single linear layer,
    without introducing nonlinearity or structural information.
    """

    def _setup(self) -> None:

        self.irreps_out = o3.Irreps(f"{self.num_channel}x0e")

        self.elem_emb1 = e3nnLinear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attrs: torch.Tensor,
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

        self.elem_emb1 = e3nnLinear(
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
        edge_attrs: torch.Tensor,
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

        self.elem_emb1 = e3nnLinear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias
        )

        self.elem_emb2 = e3nnLinear(
            f"{self.num_elements}x0e",
            f"{self.num_groups}x0e",
            bias=self.bias
        )

        self.group_emb1 = e3nnLinear(
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
        edge_attrs: torch.Tensor,
        cutoff: torch.Tensor
    ) -> torch.Tensor:
         
        elem_emb = self.elem_emb1(node_attrs)     
        elem_emb = self.act1(elem_emb)        

        logits = self.elem_emb2(node_attrs)   
        scores = self.act2(logits)      
        group_emb = self.group_emb1(scores)

        return elem_emb + group_emb
    

class TensorNodeEmbedding(NodeEmbedding):

    def _setup(self) -> None:
        
        self.node_embedding = e3nnLinear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias
        )
        self.source_embedding = e3nnLinear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias
        )
        self.target_embedding = e3nnLinear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias
        )
        torch.nn.init.uniform_(self.source_embedding.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.target_embedding.weight, a=-0.001, b=0.001)

        self.rejector = O3ScatterTensorProduct(
            [(self.num_channel, (0, 1))],
            [(1, (l, (-1)**l)) for l in range(self.lmax+1)],
            [(1, (l, (-1)**l)) for l in range(self.Lmax+1)],
        )

        self.irreps_out = self.rejector.irreps_out

        self.edge_info = MLP(
            channels=[
                self.num_radial_basis + self.num_channel * 2, 
                self.num_channel,
                self.num_channel,
                self.rejector.weight_numel,                
            ],
            bias=True,
            layer_norm=True,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attrs: torch.Tensor,
        cutoff: torch.Tensor
    ) -> torch.Tensor:
        
        assert cutoff is not None
         
        base_node_feats = self.node_embedding(node_attrs) 
        source_feats = self.source_embedding(node_attrs[edge_index[0]]) 
        target_feats = self.target_embedding(node_attrs[edge_index[1]]) 
        node_feats = self.rejector(
            torch.ones_like(base_node_feats),
            edge_attrs,
            self.edge_info(torch.cat([edge_feats, source_feats, target_feats], dim=-1)) * cutoff,
            edge_index,
        ) / self.avg_num_neighbors

        node_feats[:, :self.num_channel] = node_feats.narrow(1, 0, self.num_channel) + base_node_feats

        return node_feats
    

class SO2TensorNodeEmbedding(NodeEmbedding):

    def _setup(self) -> None:

        self.node_embedding = e3nnLinear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias
        )
        self.source_embedding = e3nnLinear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias
        )
        self.target_embedding = e3nnLinear(
            f"{self.num_elements}x0e",
            f"{self.num_channel}x0e",
            bias=self.bias
        )
        torch.nn.init.uniform_(self.source_embedding.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.target_embedding.weight, a=-0.001, b=0.001)
        self.edge_info = MLP(
            channels=[
                self.num_radial_basis + self.num_channel * 2, 
                self.num_channel,
                self.num_channel,
                (self.Lmax + 1) * self.num_channel,                
            ],
            bias=True,
            layer_norm=True,
        )
        self.irreps_out = o3.Irreps([(self.num_channel, (l, (-1)**l)) for l in range(self.Lmax + 1)])
        self.reshape = LayoutTransform(self.irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attrs: torch.Tensor,
        cutoff: torch.Tensor
    ) -> torch.Tensor:
        
        assert cutoff is not None

        base_node_feats = self.node_embedding(node_attrs) 
        source_feats = self.source_embedding(node_attrs[edge_index[0]]) 
        target_feats = self.target_embedding(node_attrs[edge_index[1]]) 

        edge_feats = self.edge_info(torch.cat([edge_feats, source_feats, target_feats], dim=-1)) * cutoff
        edge_feats = edge_feats.view(edge_feats.size(0), (self.Lmax + 1), self.num_channel)

        edge_feats = torch.bmm(
            self.so2_angular_basis.wigner_inv.narrow(2, 0, (self.lmax + 1)), 
            edge_feats
        )  # (edge, so3_m, C)

        node_feats = scatter_sum(
            edge_feats, 
            edge_index[1], 
            dim=0, 
            dim_size=base_node_feats.size(0)
        ) / self.avg_num_neighbors

        node_feats[:, 0:1, :] = node_feats.narrow(1, 0, 1) + base_node_feats.unsqueeze(1)

        # node_feats = torch.zeros(
        #     (
        #         node_attrs.size(0),
        #         ((self.lmax + 1) ** 2),
        #         self.num_channel
        #     ),
        #     device=node_attrs.device,
        #     dtype=node_attrs.dtype
        # )
        # base_node_feats = self.node_embedding(node_attrs) 
        # node_feats[:, 0:1, :] = node_feats.narrow(1, 0, 1) + base_node_feats.unsqueeze(1)

        return self.reshape.inverse(node_feats)


# class RepresentationNodeEmbedding
#     def _setup(self) -> None:

#         self.irreps_out = o3.Irreps(f"{self.num_channel}x0e")

#         self.representation = Representation

#     def forward(
#         self,
#         node_attrs: torch.Tensor,
#         edge_feats: torch.Tensor,
#         edge_index: torch.Tensor,
#         edge_attrs: torch.Tensor,
#         cutoff: torch.Tensor
#     ) -> torch.Tensor:
        
#         return self.representation(node_attrs)
    
NODE_EMBEDDING = {
    "linear": LinearNodeEmbedding,
    "nonlinear": NonLinearNodeEmbedding,
    "group": GroupNodeEmbedding,
    "tensor": TensorNodeEmbedding,
    "so2_tensor": SO2TensorNodeEmbedding,
}