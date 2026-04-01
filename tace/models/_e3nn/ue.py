################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict


import torch
from e3nn import o3


from ...dataset.quantity import PROPERTY
from ..utils import expand_dims_to
from ..mlp import ACTIVATION, MLP
from ..layout import LayoutTransform
from ..ictd import ICTD

def add_l0_to_left(t: torch.Tensor, l0: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [t[:, 0:1, :] + l0, t[:, 1:, :]],
        dim=1
    )

def add_l1_to_left(t: torch.Tensor, l1: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [t[:, 0:1, :], t[:, 1:4, :] + l1, t[:, 4:, :]],
        dim=1
    )

# def add_l2_to_left(t: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
#     return torch.cat(
#         [t[:, 0:4, :], t[:, 4:9, :] + l2, t[:, 9:, :]],
#         dim=1
#     )

# def add_l3_to_left(t: torch.Tensor, l3: torch.Tensor) -> torch.Tensor:
#     return torch.cat(
#         [t[:, 0:9, :], t[:, 9:16, :] + l3, t[:, 16:, :]],
#         dim=1
#     )

ADD_FN = {
    0: add_l0_to_left,
    1: add_l1_to_left,
    # 2: add_l2_to_left,
    # 3: add_l3_to_left,
}


class UniversalInvariantEmbedding(torch.nn.Module):
    def __init__(
        self,
        out_dim: int,
        invariant_embedding: Dict[str, bool | str | int],
        bias: bool,
        act: str,
    ):
        super().__init__()

        self.uie = torch.nn.ModuleDict()

        total_dim = 0
        for k, v in invariant_embedding.items():
            p_type = PROPERTY[k]["type"]
            if p_type == "int":
                self.uie[k] = torch.nn.Embedding(v["num_embeddings"], out_dim)
            elif p_type == "float":
                self.uie[k] = MLP(
                    [1, out_dim, out_dim],
                    bias=bias,
                    layer_norm=False,
                    act=act,
                )
            total_dim += out_dim

        self.project = MLP(
            [total_dim, out_dim],
            bias=bias,
            layer_norm=False,
            act=act,
        )
        self.act = ACTIVATION[act]()

    def forward(
        self,
        batch: torch.Tensor,
        attrs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        embeddings = []

        for p, module in self.uie.items():

            p_type = PROPERTY[p]['type']
            p_scope = PROPERTY[p]['scope']

            attr = attrs[p]
            if p_scope == "per-system":
                attr = attr[batch]
            if p_type == 'float':
                attr = attr.unsqueeze(-1)

            embeddings.append(module(attr))

        return self.act(self.project(torch.cat(embeddings, dim=-1))).unsqueeze(1)


class EquivariantEmbedding(torch.nn.Module):
    def __init__(
        self,
        rank: int,
        scope: str,
        num_elements: int,
        num_channel: int,
        element_trainable: bool = True,
        channel_trainable: bool = True,
        normalizer: float = 1.0,
    ):
        super().__init__()

        if element_trainable:
            self.element_weights = torch.nn.Parameter(
                torch.ones(num_elements, dtype=torch.get_default_dtype())
            )
        else:
            self.register_buffer(
                "element_weights",
                torch.ones(num_elements, dtype=torch.get_default_dtype()),
                persistent=False,
            )
        if channel_trainable:
            self.channel_weights = torch.nn.Parameter(
                torch.ones(num_channel, dtype=torch.get_default_dtype())
            )
        else:
            self.register_buffer(
                "channel_weights",
                torch.ones(num_channel, dtype=torch.get_default_dtype()),
                persistent=False,
            )
        self.p_rank = rank
        self.p_scope = scope
        self.p_add_fn = ADD_FN[self.p_rank]
        self.p_normalizer = normalizer

        # self.register_buffer(
        #     "C",
        #     ICTD(rank, rank)[3][0],
        #     persistent=False,
        # )

    def forward(
        self,
        node_feats: torch.Tensor,
        node_attrs: torch.Tensor,
        batch: torch.Tensor,
        attr: torch.Tensor,
    ):
        
        label = attr * self.p_normalizer
        if self.p_scope == "per-system":
            label = label[batch].unsqueeze(-1)
        else:
            label = label.unsqueeze(-1) # [B, M, C]

        element_weights = torch.einsum('bz, z -> b', node_attrs, self.element_weights)
        W = element_weights.unsqueeze(-1) * self.channel_weights.unsqueeze(0) # [B, C]
        W = W.unsqueeze(1) # [B, M, C]
        embedding = label * expand_dims_to(W, label.ndim, dim=1)
        return self.p_add_fn(node_feats, embedding)
  
    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return (
            f"{cls}(\n"
            f"  rank={self.p_rank},\n"
            f"  normalizer={self.p_normalizer},\n"
            f")"
        )


class UniversalEquivariantEmbedding(torch.nn.Module):
    def __init__(
        self,
        equivariant_embedding: Dict[str, bool | str | int],
        num_elements: int,
        num_channel: int,
        lmax: int,
    ):
        super().__init__()

        self.uee = torch.nn.ModuleDict()
        for k, v in equivariant_embedding.items():
            if PROPERTY[k]['rank'] > 0:
                self.uee[k] = EquivariantEmbedding(
                    PROPERTY[k]["rank"],
                    PROPERTY[k]["scope"],
                    num_elements,
                    num_channel,
                    element_trainable=True,
                    channel_trainable=True,
                    normalizer=v['normalizer'],
                )

        self.reshape = LayoutTransform(
            o3.Irreps([(num_channel, (l, 1)) for l in range(lmax+1)])
        )

    def forward(
            self, 
            node_feats: torch.Tensor, 
            node_attrs: torch.Tensor,
            batch: torch.Tensor,
            attrs: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
        node_feats = self.reshape(node_feats)
        for p, module in self.uee.items():
            node_feats = module(node_feats, node_attrs, batch, attrs[p])
        node_feats = self.reshape.inverse(node_feats)
        return node_feats