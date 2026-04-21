# ################################################################################
# # Authors: Zemin Xu
# # License: MIT, see LICENSE.md
# ################################################################################

# import math
# import torch
# import torch.nn.functional as F
# from e3nn import o3


# from ..layout import LayoutTransform
# from .base import Residual
# from .linear import Linear, ElementLinear


# # class AttentionResidual_BB(Residual):
# #     def _setup(self):

# #         self.alpha = 1 / math.sqrt(self.num_channel) # not use rms_norm for key
# #         self.query = torch.nn.Parameter(torch.zeros(1, self.num_channel)) # if need, can add element
# #         self.linear = torch.nn.ModuleList()
# #         self.reshape = LayoutTransform(self.irreps_out)

# #         irreps_in = [o3.Irreps(f"{self.num_channel}x0e")]
# #         for _ in range(self.window):
# #             irreps_in.append(self.irreps_in)
# #         for idx in range(self.window):
# #             if self.linear_type == 'aware':
# #                 self.linear.append(
# #                     ElementLinear(
# #                         irreps_in[idx],
# #                         self.irreps_out,
# #                         bias=self.use_bias,
# #                         num_elements=self.num_elements,
# #                     )
# #                 )
# #             else:
# #                 self.linear.append(
# #                     Linear(
# #                         irreps_in[idx],
# #                         self.irreps_out,
# #                         bias=self.use_bias,
# #                     )
# #                 )

# #     def forward(self, prev_feats: list[torch.Tensor], node_attrs: torch.Tensor):

# #         prev_feats = prev_feats[-self.window:]
# #         key = torch.stack([feats[:, :self.num_channel] for feats in prev_feats], dim=0)
# #         logits = torch.einsum('c, lbc -> lb', self.query.squeeze(0), key) * self.alpha
# #         attn = F.softmax(logits, dim=0) 
# #         new_feats = []
# #         for idx in range(self.window):
# #             if self.linear_type == 'aware':
# #                 new_feats.append(
# #                     self.reshape(self.linear[idx](prev_feats[idx], node_attrs))
# #                 )
# #             else:
# #                 new_feats.append(
# #                     self.reshape(self.linear[idx](prev_feats[idx]))
# #                 )     
# #         value = torch.stack(new_feats, dim=0)

# #         return self.reshape.inverse(torch.einsum('lb, lbmc -> bmc', attn, value))

    
# # class AttentionResidual_AB(Residual):
# #     def _setup(self):

# #         self.alpha = 1 / math.sqrt(self.num_channel) # not use rms_norm for key
# #         self.query = torch.nn.Parameter(torch.zeros(1, self.num_channel)) # if need, can add element
# #         self.linear = torch.nn.ModuleList()
# #         self.reshape = LayoutTransform(self.irreps_out)

# #         for idx in range(self.window):
# #             if self.linear_type == 'aware':
# #                 self.linear.append(
# #                     ElementLinear(
# #                         self.irreps_in,
# #                         self.irreps_out,
# #                         bias=self.use_bias,
# #                         num_elements=self.num_elements,
# #                     )
# #                 )
# #             else:
# #                 self.linear.append(
# #                     Linear(
# #                         self.irreps_out,
# #                         self.irreps_out,
# #                         bias=self.use_bias,
# #                     )
# #                 )

# #     def forward(self, prev_feats: list[torch.Tensor], node_attrs: torch.Tensor):

# #         prev_feats = prev_feats[-self.window:]
# #         key = torch.stack([feats[:, :self.num_channel] for feats in prev_feats], dim=0)
# #         logits = torch.einsum('c, lbc -> lb', self.query.squeeze(0), key) * self.alpha
# #         attn = F.softmax(logits, dim=0) 
# #         new_feats = []
# #         for idx in range(self.window):
# #             if self.linear_type == 'aware':
# #                 new_feats.append(
# #                     self.reshape(self.linear[idx](prev_feats[idx], node_attrs))
# #                 )
# #             else:
# #                 new_feats.append(
# #                     self.reshape(self.linear[idx](prev_feats[idx]))
# #                 )     
# #         value = torch.stack(new_feats, dim=0)

# #         return self.reshape.inverse(torch.einsum('lb, lbmc -> bmc', attn, value))



# class AttentionResidual(Residual):
#     def _setup(self):

#         self.alpha = 1 / math.sqrt(self.num_channel) # not use rms_norm for key
#         self.query = torch.nn.Parameter(torch.zeros(1, self.num_channel)) # if need, can add element
#         self.reshape = LayoutTransform(self.irreps_out)

#     def forward(self, prev_feats: list[torch.Tensor]):

#         prev_feats = prev_feats[-self.window:]
#         key = torch.stack([feats[:, :self.num_channel] for feats in prev_feats], dim=0)
#         logits = torch.einsum('c, lbc -> lb', self.query.squeeze(0), key) * self.alpha
#         attn = F.softmax(logits, dim=0)     
#         value = torch.stack(prev_feats[-self.window:], dim=0)

#         return self.reshape.inverse(torch.einsum('lb, lbmc -> bmc', attn, value))