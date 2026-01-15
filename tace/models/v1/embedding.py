################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import List, Dict, Optional


import torch
from torch import nn, Tensor
from cartnn import ICTD

from .act import ACT
from .gate import NormGate
from .linear import Linear
from .utils import expand_dims_to
from ...dataset.quantity import PROPERTY, UNIVERSAL_EMBEDDING_ALLOWED_PROPERTY


def add_rank0_to_left(T: Dict[int, torch.Tensor], rank0: torch.Tensor) -> Dict[int, torch.Tensor]:
    if 0 in T:
        T[0] = T[0] + rank0
    else:
        T[0] = rank0
    return T


def add_rank1_to_left(T: Dict[int, torch.Tensor], rank1: torch.Tensor) -> Dict[int, torch.Tensor]:
    if 1 in T:
        T[1] = T[1] + rank1
    else:
        T[1] = rank1
    return T


def add_rank2_to_left(T: Dict[int, torch.Tensor], rank2: torch.Tensor) -> Dict[int, torch.Tensor]:
    if 2 in T:
        T[2] = T[2] + rank2
    else:
        T[2] = rank2
    return T


def add_rank3_to_left(T: Dict[int, torch.Tensor], rank3: torch.Tensor) -> Dict[int, torch.Tensor]:
    if 3 in T:
        T[3] = T[3] + rank3
    else:
        T[3] = rank3
    return T


ADD_FN = {
    0: add_rank0_to_left,
    1: add_rank1_to_left,
    2: add_rank2_to_left,
    3: add_rank3_to_left,
}


class UniversalInvariantEmbedding(torch.nn.Module):
    def __init__(
        self,
        out_dim: int,
        invariant_embedding: Dict[str, bool | str | int],
    ):
        super().__init__()

        self.embeddings = nn.ModuleDict()

        total_dim = 0
        for k, v in invariant_embedding.items():
            if v.get('enable', False) and PROPERTY[k]['rank'] == 0:
                p_type = PROPERTY[k]["type"]
                if p_type == "int":
                    self.embeddings[k] = nn.Embedding(v["num_embeddings"], out_dim)
                elif p_type == "float":
                    act = v.get("act", "silu")
                    self.embeddings[k] = nn.Sequential(
                        nn.Linear(1, out_dim, bias=False),
                        ACT[act](),
                        nn.Linear(out_dim, out_dim, bias=False),
                    )
                total_dim += out_dim

        self.project = nn.Sequential(
            nn.Linear(total_dim, out_dim, bias=False),
            nn.SiLU(),
        )

    def forward(
        self,
        batch: Tensor,
        attrs: Dict[str, Tensor],
    ) -> Tensor:
        embeddings = []

        for p, module in self.embeddings.items():
            attr = attrs[p]

            type_ = PROPERTY[p]['type']
            scope = PROPERTY[p]['scope']

            if scope == "per-system":
                attr = attr[batch]

            if type_ == 'float':
                attr = attr.unsqueeze(-1)
    
            embeddings.append(module(attr))

        return self.project(torch.cat(embeddings, dim=-1))


class EquivariantEmbedding(torch.nn.Module):
    def __init__(
        self,
        p: str,
        rank: int,
        scope: str,
        atomic_numbers: List,
        num_channel: int,
        element_trainable: bool = True,
        channel_trainable: bool = True,
        normalizer: float = 1.0,
        gate: Optional[str] = None,
        linear: bool = False,
        extra: Dict[str, bool] = {},
    ):
        super().__init__()
        num_elements = len(atomic_numbers)
        if element_trainable:
            self.element_weights = nn.Parameter(
                torch.ones(num_elements, dtype=torch.get_default_dtype())
            )
        else:
            self.register_buffer(
                "element_weights",
                torch.ones(num_elements, dtype=torch.get_default_dtype()),
            )
        if channel_trainable:
            self.channel_weights = nn.Parameter(
                torch.ones(num_channel, dtype=torch.get_default_dtype())
            )
        else:
            self.register_buffer(
                "channel_weights",
                torch.ones(num_channel, dtype=torch.get_default_dtype()),
            )
        self.p = p
        self.rank = rank
        self.add_fn = ADD_FN[self.rank]
        self.scope = scope
        self.normalizer = normalizer
        self.gate = gate
        self.nonlinearity = NormGate[gate](self.rank, num_channel)

        if linear:
            self.linear = Linear(
                in_dim=num_channel,
                out_dim=num_channel,
                bias=False,
                l=self.rank,
            )

    def forward(
        self,
        node_feats: Dict[int, Tensor],
        node_attrs: Tensor,
        batch: Tensor,
        attr: Dict[str, Tensor],
    ):
        
        element_idx = torch.argmax(node_attrs, dim=-1)

        label = attr * self.normalizer

        # === General ===
        if self.scope == "per-system":
            label = label[batch].unsqueeze(1)
        else:
            label = label.unsqueeze(1)

        element_weights = expand_dims_to(
            self.element_weights[element_idx], n_dim=self.rank + 2
        )
        shape = (1, -1) + (1,) * self.rank
        channel_weights = self.channel_weights.view(*shape)
        
        embedding = self.nonlinearity(label * element_weights * channel_weights)

        if hasattr(self, "linear"):
            embedding = self.linear(embedding)

        return self.add_fn(node_feats, embedding)
  
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rank={self.rank}, normalizer={self.normalizer}), gate={self.gate}"

class UniversalEquivariantEmbedding(torch.nn.Module):
    def __init__(
        self,
        equivariant_embedding: Dict[str, bool | str | int],
        atomic_numbers: List,
        num_channel: int,
    ):
        super().__init__()

        self.equivariant_embedding = equivariant_embedding
        self.embeddings = nn.ModuleDict()
        for k, v in equivariant_embedding.items():
            if v.get('enable', False) and PROPERTY[k]['rank'] > 0:
                self.embeddings[k] = EquivariantEmbedding(
                    k,
                    PROPERTY[k]["rank"],
                    PROPERTY[k]["scope"],
                    atomic_numbers,
                    num_channel,
                    element_trainable=True,
                    channel_trainable=True,
                    normalizer=float(v.get('normalizer', 1.0)),
                    gate=v.get("gate", None),
                    linear=v.get("linear", False),
                    extra=v.get("extra", {}),
                )

    def forward(
            self, 
            node_feats: Dict[int, torch.Tensor], 
            node_attrs: torch.Tensor,
            batch: torch.Tensor,
            attrs: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
        for p, module in self.embeddings.items():
            node_feats = module(node_feats, node_attrs, batch, attrs[p])
        return node_feats
