# ################################################################################
# # Authors: Zemin Xu
# # License: MIT, see LICENSE.md
# ################################################################################

# import math
# from typing import List


# import torch
# from e3nn import o3


# from ..layout import LayoutTransform

# import torch
# import torch.nn as nn
# from e3nn import o3


# class AttentionResidual(nn.Module):
#     def __init__(
#         self,
#         num_layers: int,
#         Lmax: int,
#         target_weight: List[int],
#         num_channel: int,
#         num_elements: int,
#     ) -> None:
#         super().__init__()

#         self.num_layers = num_layers
#         self.Lmax = Lmax
#         self.num_channel = num_channel

#         # irreps
#         self.irreps = (o3.Irreps.spherical_harmonics(Lmax) * num_channel).regroup()
#         self.reshape = LayoutTransform(self.irreps)

#         # grid size
#         self.num_latitude = 2 * (Lmax + 1)
#         self.num_longitude = 2 * (Lmax) + 1

#         # S2 transforms
#         to_s2 = o3.ToS2Grid(
#             Lmax,
#             (self.num_latitude, self.num_longitude),
#             normalization="component",
#         )
#         from_s2 = o3.FromS2Grid(
#             (self.num_latitude, self.num_longitude),
#             Lmax,
#             normalization="component",
#         )

#         self.register_buffer(
#             "to_grid",
#             torch.einsum("mbi, am -> bai", to_s2.shb, to_s2.sha).detach(),
#             persistent=False,
#         )
#         self.register_buffer(
#             "from_grid",
#             torch.einsum("am, mbi -> bai", from_s2.sha, from_s2.shb).detach(),
#             persistent=False,
#         )

#         # ===== Attention over layers =====
#         C = num_channel

#         self.W_q = torch.nn.Parameter(torch.randn(num_elements, C, C))
#         self.W_k = torch.nn.Parameter(torch.randn(num_elements, C, C))
#         self.W_v = torch.nn.Parameter(torch.randn(num_elements, C, C))
#         self.alpha = 1 / math.sqrt(num_channel)

#         self.layer_emb = nn.Parameter(torch.randn(num_layers, C))


#     def forward(
#         self,
#         node_feats_list,
#         node_attrs,
#     ):
#         B = node_feats_list[0].shape[0]
#         L = len(node_feats_list)

#         grids = []
#         pooled = []

#         for feats in node_feats_list:
#             pad_shape = list(feats.shape)
#             pad_shape[1] = self.irreps.dim - pad_shape[1]
#             if pad_shape[1] > 0:
#                 padding = torch.zeros(
#                     pad_shape,
#                     dtype=feats.dtype,
#                     device=feats.device,
#                 )
#                 feats = torch.cat([feats, padding], dim=1)
#             feats = self.reshape(feats)
#             grid = self._to_grid(feats)  # [B, beta, alpha, C]
#             grids.append(grid)
#             pooled_feat = grid.mean(dim=(1, 2))  # [B, C]
#             pooled.append(pooled_feat)

#         # [B, L, C]
#         pooled = torch.stack(pooled, dim=1)
#         pooled = pooled + self.layer_emb[:L].unsqueeze(0)
#         Q = torch.einsum('bz, zij, bli -> blj', node_attrs, self.W_q, pooled) * self.alpha # [B, L, C]
#         K = torch.einsum('bz, zij, bli -> blj', node_attrs, self.W_q, pooled) * self.alpha
#         V = torch.einsum('bz, zij, bli -> blj', node_attrs, self.W_q, pooled) * self.alpha
#         attn_scores = torch.matmul(Q, K.transpose(-1, -2))  # [B, L, L]

#         mask = torch.triu(
#             torch.ones(L, L, device=attn_scores.device), diagonal=1
#         ).bool()
#         attn_scores = attn_scores.masked_fill(mask, float("-inf"))
#         # attn_scores[:, 0, 0] = 0.0
#         attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, L, L]
#         aggregated = torch.matmul(attn_weights, V)  # [B, L, C]
#         outs = []

#         for l in range(L):
#             grid = grids[l]  # [B, beta, alpha, C]
#             agg = aggregated[:, l]  # [B, C]
#             agg = agg[:, None, None, :]  # broadcast

#             out_grid = grid + agg  # residual + attention

#             out = self.reshape.inverse(self._from_grid(out_grid))

#             if l == L -1:

#                 out = out[:, :self.num_channel]
#             outs.append(out)

#         return outs


#     def _to_grid(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B, i, C]
#         return torch.einsum("bai, Bic -> Bbac", self.to_grid, x)

#     def _from_grid(self, x: torch.Tensor) -> torch.Tensor:
#         return torch.einsum("bai, Bbac -> Bic", self.from_grid, x)

