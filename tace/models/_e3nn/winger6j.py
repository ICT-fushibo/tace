# import math

# import torch
# import torch.nn as nn

# from e3nn import o3
# from e3nn.o3 import wigner_3j

# from sympy.physics.wigner import wigner_6j
# from sympy import S


# class NodeSphericalHarmonics(nn.Module):

#     def __init__(self, lmax: int) -> None:
#         super().__init__()

#         self.lmax = lmax

#         self.sh = o3.SphericalHarmonics(
#             irreps_out=o3.Irreps.spherical_harmonics(
#                 lmax,
#                 p=-1,
#             ),
#             normalize=False,
#             normalization="integral",
#         )

#     def forward(
#         self,
#         pos: torch.Tensor,
#     ) -> torch.Tensor:
#         return self.sh(pos) # [node, (lmax+1)^2]

# def irrep_slice(l: int):

#     start = l * l
#     end = (l + 1) * (l + 1)

#     return slice(start, end)


# def allowed_ls(l1, l2):
#     return list(range(abs(l1 - l2), l1 + l2 + 1))


# class Wigner6jConv(nn.Module):
#     def __init__(
#         self,
#         Lmax,
#         lmax,
#         num_channel: int,
#     ):
#         super().__init__()

    
#         self.num_channel = num_channel

#         for l1 in range(Lmax + 1):
#             for l2 in range(lmax + 1):
#                 for l in allowed_ls(l1, l2):
#                     if l > lmax:
#                         continue
#                     w3j = wigner_3j(l1, l2, l).permute(2, 0, 1).contiguous()
#                     self.register_buffer(f"w3j_{l1}_{l2}_{l}", w3j, persistent=False)

#         for l1 in range(l1max + 1):
#             for l2 in range(l2max + 1):
#                 for l3 in range(l3max + 1):
#                     for l23 in allowed_ls(l2, l3):
#                         if l23 > l23max:
#                             continue
#                         for l12 in allowed_ls(l1, l2):
#                             if l12 > lmax:
#                                 continue
#                             for L in allowed_ls(l1, l23):
#                                 if L > lmax:
#                                     continue
#                                 if L not in allowed_ls(l12, l3):
#                                     continue
#                                 coeff = (
#                                     math.sqrt((2 * l12 + 1) * (2 * l23 + 1))
#                                     * float(
#                                         wigner_6j(
#                                             S(l1), S(l2), S(l12),
#                                             S(l3), S(L), S(l23),
#                                         )
#                                     )
#                                 )
#                                 self.register_buffer(
#                                     (
#                                         f"w6j_"
#                                         f"{l1}_{l2}_{l12}_"
#                                         f"{l3}_{L}_{l23}"
#                                     ),
#                                     torch.tensor(coeff),
#                                 )

#     def get_cg(self, l1, l2, L):
#         return getattr(self, f"w3j_{l1}_{l2}_{L}")

#     def get_w6j(
#         self,
#         lf, li, l12,
#         lj, L, l23,
#     ):
#         return getattr(
#             self,
#             (
#                 f"w6j_"
#                 f"{lf}_"
#                 f"{li}_"
#                 f"{l12}_"
#                 f"{lj}_"
#                 f"{L}_"
#                 f"{l23}"
#             )
#         )

#     def couple(
#         self,
#         x,
#         y,
#         l1,
#         l2,
#         L,
#     ):
#         C = self.get_cg(l1, l2, L)
#         return torch.einsum("Mab,Bac,Bbc->BMc", C, x, y)

#     def forward(
#         self,
#         node_feats,
#         node_sh,
#         edge_index,
#     ):

#         source = edge_index[0]
#         target = edge_index[1]
#         N = node_feats.size(0)

#         out = torch.zeros_like(node_feats)


#         feats_by_l = {}
#         sh_by_l = {}

#         for l in range(self.lmax + 1):
#             sl = irrep_slice(l)
#             feats_by_l[l] = node_feats[:, sl]
#             sh_by_l[l] = node_sh[:, sl].unsqueeze(-1)
    
#         # ====================================================
#         # LEFT TREE:
#         # f_j ⊗ (Yi⊗Yj)_l23
#         # ====================================================

#         for li in range(self.lmax + 1):
#             Yi = sh_by_l[li][target]
#             for lj in range(self.lmax + 1):
#                 Yj = sh_by_l[lj][source]

#                 #
#                 # (Yi⊗Yj)_l23
#                 #

#                 for l23 in allowed_ls(li, lj):

#                     if l23 > self.lmax:
#                         continue

#                     edge_sh = self.couple(
#                         Yi,
#                         Yj,
#                         li,
#                         lj,
#                         l23,
#                     )

#                     for lf in range(self.lmax + 1):
#                         fj = feats_by_l[lf][source]
#                         # LEFT:
#                         # fj ⊗ edge_sh

#                         for L in allowed_ls(lf, l23):

#                             if L > self.lmax:
#                                 continue

#                             left = self.couple(
#                                 fj,
#                                 edge_sh,
#                                 lf,
#                                 l23,
#                                 L,
#                             )

#                             agg = torch.zeros(
#                                 N,
#                                 2 * L + 1,
#                                 self.channels,
#                                 device=node_feats.device,
#                                 dtype=node_feats.dtype,
#                             )
#                             agg.index_add_(
#                                 0,
#                                 target,
#                                 left,
#                             )

#                             # recouple: fj⊗(Yi⊗Yj) -> ((fj⊗Yi)⊗Yj)
#                             recoupled = torch.zeros_like(
#                                 agg
#                             )

#                             for l12 in allowed_ls(
#                                 lf,
#                                 li,
#                             ):

#                                 if l12 > self.lmax:
#                                     continue

#                                 if L not in allowed_ls(
#                                     l12,
#                                     lj,
#                                 ):
#                                     continue

#                                 coeff = self.get_w6j(
#                                     lf, li, l12,
#                                     lj, L, l23,
#                                 )
#                                 # (fj⊗Yi)_l12
#                                 tmp12 = self.couple(
#                                     fj,
#                                     Yi,
#                                     lf,
#                                     li,
#                                     l12,
#                                 )

#                                 # ((fj⊗Yi)_l12⊗Yj)_L
#                                 right = self.couple(
#                                     tmp12,
#                                     Yj,
#                                     l12,
#                                     lj,
#                                     L,
#                                 )

#                                 tmpagg = torch.zeros_like(
#                                     agg
#                                 )

#                                 tmpagg.index_add_(
#                                     0,
#                                     source,
#                                     right,
#                                 )
#                                 recoupled += coeff * tmpagg
                            
#                             # LEFT == RIGHT
#                             sl = irrep_slice(L)

#                             out[:, sl] += recoupled

#         return out


# if __name__ == "__main__":

#     torch.manual_seed(0)

#     N = 8
#     lmax = 2
#     C = 4

#     pos = torch.randn(N, 3)

#     feat = torch.randn(N, (lmax + 1) ** 2, C)

#     edge_index = torch.tensor([
#         [0, 1, 2, 3, 4],
#         [1, 2, 3, 4, 5],
#     ])

#     sh_layer = NodeSphericalHarmonics(
#         lmax=lmax,
#     )

#     node_sh = sh_layer(pos)

#     conv = Wigner6jConv(
#         lmax=lmax,
#         channels=C,
#     )

#     out = conv(
#         feat,
#         node_sh,
#         edge_index,
#     )

#     print(out.shape)
