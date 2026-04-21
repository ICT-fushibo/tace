################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch
from e3nn.nn import Activation
from e3nn import o3


from .base import EdgeEmbedding, EdgeUpdate
from .linear import Linear


class IdentityEdgeEmbedding(EdgeEmbedding):
    """
    An identity edge embedding module.

    This class directly returns the input edge features (radial) without any transformation.
    """

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
    """
    A linear edge embedding module.

    This class projects the input edge features (radial)
    into a higher-dimensional feature space using a linear transformation.

    This is motivated by the fact that when edge update are used,
    a low-dimensional radial representation may become a bottleneck and limit
    the expressiveness of edge features.
    """
    
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


class NonLinearEdgeEmbedding(EdgeEmbedding):
    """
    A nonlinear edge embedding module.

    This class applies a nonlinear activation function after a linear projection
    of edge features, allowing for more expressive representations compared to
    purely linear transformations.

    This is motivated by the fact that when edge update are used,
    a low-dimensional radial representation may become a bottleneck and limit
    the expressiveness of edge features.
    """

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
        
        return self.act1(self.radial_proj(edge_feats))


class ElementEdgeEmbedding(EdgeEmbedding):
    """
    An edge embedding module that incorporates both radial and element information.

    This class combines transformed edge features with embeddings of the source
    and target nodes, allowing the edge representation to depend not only on
    geometric information but also on the types of connected elements.

    Note:
        When using this module, it is recommended not to additionally use
        edge update modules that rely purely on element information, as this
        may lead to an overemphasis on element features in the edge representation.
    """

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
        
        x_j = self.source_embedding(node_attrs)[edge_index[0]]
        x_i = self.target_embedding(node_attrs)[edge_index[1]]
        edge_feats = self.radial_proj(edge_feats)

        return self.act1(torch.cat([edge_feats, x_i, x_j], dim=-1))


class IdentityEdgeUpdate(EdgeUpdate):
    """
    An identity edge update module.

    This class directly returns the input edge features (edge embedding) without modification.
    """

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
    """
    An edge update module that incorporates edge element information.

    This class augments edge features by concatenating embeddings of the
    source and target node elements, allowing edge representations to
    explicitly depend on the types of connected nodes.
    """

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
    """
    A variant of element-based edge update with reversed ordering.

    This class is similar to ElementEdgeUpdate but swaps the order of
    source and target element embeddings when concatenating.
    """

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
    """
    An edge update module based on tensor dot interactions.

    This module is analogous to self-attention. However,
    instead of applying a softmax to obtain attention weights, the resulting
    interactions are directly treated as edge features and used to generate
    convolution weights, enabling the model to capture higher-order geometric 
    and feature correlations between connected nodes.

    """

    def _setup(self) -> None:

        c = self.tensor_dot_channel
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


class TensorDotElement2EdgeUpdate(TensorDotEdgeUpdate, ElementEdgeUpdate):

    def _setup(self) -> None:

        TensorDotEdgeUpdate._setup(self)
        ElementEdgeUpdate._setup(self)


        if self.layer == 0:
            self.out_dim = self.edge_embedding_channel + 2 * self.num_channel     
        else:
            self.out_dim = self.edge_embedding_channel + self.tensor_dot_channel * len(self.irreps_node) + 2 * self.num_channel  


    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None,
    ) -> torch.Tensor:
        
        assert cutoff is not None
        
        edge_feats_list = [edge_feats]
        if self.layer > 0:
            Q = self.linear_q(node_feats)
            K = self.linear_k(node_feats)
            edge_feats_list.append(
                self.dot(Q[edge_index[1]], K[edge_index[0]])
            )
        edge_feats_list.append(
            self.target_embedding(node_attrs[edge_index[1]])
        )

        edge_feats_list.append(
            self.source_embedding(node_attrs[edge_index[0]])
        )

        return torch.cat(edge_feats_list, dim=-1)

EDGE_EMBEDDING = {
    "identity": IdentityEdgeEmbedding,
    "linear": LinearEdgeEmbedding,
    "nonlinear": NonLinearEdgeEmbedding,
    "element": ElementEdgeEmbedding,
}

EDGE_UPDATE = {
    "identity": IdentityEdgeUpdate,
    "element": ElementEdgeUpdate,
    "element2": Element2EdgeUpdate,
    "tensor_dot": TensorDotEdgeUpdate,
    "tensor_dot_element2": TensorDotElement2EdgeUpdate,
}



