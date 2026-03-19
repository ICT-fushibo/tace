################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
# TODO, add basis change  for l > 1

from typing import Dict


import torch
from cartnn.o3 import ICTD, expand_dims_to

from ...dataset.quantity import PROPERTY
from ..mlp import ACTIVATION, MLP


def add_l0_to_left(T: Dict[int, torch.Tensor], rank0: torch.Tensor) -> Dict[int, torch.Tensor]:
    if 0 in T:
        T[0] = T[0] + rank0
    else:
        T[0] = rank0
    return T


def add_l1_to_left(T: Dict[int, torch.Tensor], rank1: torch.Tensor) -> Dict[int, torch.Tensor]:
    if 1 in T:
        T[1] = T[1] + rank1
    else:
        T[1] = rank1
    return T


# def add_l2_to_left(T: Dict[int, torch.Tensor], rank2: torch.Tensor) -> Dict[int, torch.Tensor]:
#     if 2 in T:
#         T[2] = T[2] + rank2
#     else:
#         T[2] = rank2
#     return T


# def add_l3_to_left(T: Dict[int, torch.Tensor], rank3: torch.Tensor) -> Dict[int, torch.Tensor]:
#     if 3 in T:
#         T[3] = T[3] + rank3
#     else:
#         T[3] = rank3
#     return T


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

    def forward(
        self,
        node_feats: Dict[int, torch.Tensor],
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

    def forward(
            self, 
            node_feats: Dict[int, torch.Tensor], 
            node_attrs: torch.Tensor,
            batch: torch.Tensor,
            attrs: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
        for p, module in self.uee.items():
            node_feats = module(node_feats, node_attrs, batch, attrs[p])
        return node_feats