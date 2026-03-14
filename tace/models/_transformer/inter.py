################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

# class TransformerInteraction(Interaction):

#     def _setup(self):

#         assert self.Lmax == self.lmax

#         self.linear_up = Linear(
#             self.irreps_in,
#             self.irreps_in,
#             self.num_channel,
#             self.num_channel * 3,
#             bias=self.use_bias,
#         )    

#         def get_scalar_linear(channel_in, channel_out, bias=self.use_bias):
#             return Linear(
#                 "0e",
#                 "0e",
#                 channel_in,
#                 channel_out,
#                 bias=bias,
#             )
        
#         if self.layer > 0 and self.resnet in ['BB']:
#             self.resnetBB = ElementLinear(
#                 self.irreps_in,
#                 self.irreps_sc,
#                 self.num_channel,
#                 self.num_channel,
#                 bias=self.use_bias,
#                 num_elements=self.num_elements,
#             )    
         
#         self.edge_info_k = MLP(
#             [self.radial_in_channel] + self.radial_mlp + [self.num_channel],
#             bias=self.radial_bias,
#             layer_norm=False,
#             act=self.radial_act,
#         )

#         self.edge_info_v = MLP(
#             [self.radial_in_channel] + self.radial_mlp + [self.num_channel],
#             bias=self.radial_bias,
#             layer_norm=False,
#             act=self.radial_act,
#         )

#         self.act = torch.nn.SiLU()
#         self.tensor_attn_proj = get_scalar_linear(self.num_channel, self.num_heads)

#         if self.layer == 0:
#             self.product_gate = get_scalar_linear(self.num_channel, self.num_channel)
#             self.linear_down = get_scalar_linear(self.num_channel, self.num_channel)
#         else:
#             self.product_gate = get_scalar_linear(self.num_channel, self.num_channel * 2)
#             self.linear_down = get_scalar_linear(self.num_channel, self.num_channel * 3)


#     def forward(
#         self,
#         node_feats: torch.Tensor,
#         node_attrs_total: torch.Tensor,
#         node_attrs_slice: torch.Tensor,
#         edge_feats: torch.Tensor,
#         edge_attrs: torch.Tensor,
#         edge_index: torch.Tensor,
#         cutoff: Optional[torch.Tensor],
#         graph,
#     ):
        
#         source, target = edge_index
#         edge_attrs = edge_attrs[:, 1:] 

#         if self.layer == 0:
#             sc = None
#             node_feats = self.linear_up(node_feats).view(
#                 node_feats.size(0),
#                 3,
#                 self.num_heads,
#                 self.head_channel,   
#             )
#             Q, K, V = torch.unbind(node_feats, dim=1)
#             dK = self.edge_info_k(edge_feats).view(-1, self.num_heads, self.head_channel)
#             dV = self.edge_info_v(edge_feats).view(-1, self.num_heads, self.head_channel)
#             A = self.act((Q[target] * K[source] * dK).sum(dim=-1)) # [B, H]   
#             if cutoff is not None:
#                 A =  A * cutoff
#             # M_ij
#             SM = V[source] * dV # [B, H, hc]
#             SM = (SM * A.unsqueeze(-1)).reshape(-1, 1, self.num_channel) # [B, 1, C]
#             G = self.act(self.product_gate(SM)) # [B, 1, C]
#             TM = G * edge_attrs.unsqueeze(2)

#             # M_i
#             SM = scatter_sum(SM, target, dim=0, dim_size=node_feats.size(0)) / self.avg_num_neighbors
#             TM = scatter_sum(TM, target, dim=0, dim_size=node_feats.size(0)) / self.avg_num_neighbors
#             SM = self.linear_down(SM)

#             M = torch.cat([SM, TM], dim=1)
#         else:
#             OT = node_feats[:, 1:, :]
#             sc = self.resnetBB(node_feats, node_attrs_slice)
#             node_feats = self.linear_up(node_feats)
#             S = node_feats[:, :1, :].view(node_feats.size(0), 3, self.num_heads, self.head_channel)
#             Q, K, V = torch.unbind(S, dim=1)
#             T = node_feats[:, 1:, :]
#             T1, T2, T3 = torch.split(T, self.num_channel, -1)
#             dK = self.edge_info_k(edge_feats).view(-1, self.num_heads, self.head_channel)
#             dV = self.edge_info_v(edge_feats).view(-1, self.num_heads, self.head_channel)
#             SA = (Q[target] * K[source] * dK).sum(dim=-1) # [B, H]   
#             TA = (OT[source] * edge_attrs.unsqueeze(-1)).sum(dim=1, keepdim=True) # [B, 1, C]
#             TA = self.tensor_attn_proj(TA).squeeze(1)     # [B, H]
#             A = self.act(SA + TA)
#             if cutoff is not None:
#                 A =  A * cutoff

#             T1_T2 = (T1 * T2).sum(dim=1, keepdim=True) # [B, C]

#             # M_ij
#             SM = V[source] * dV # [B, H, hc]
#             SM = (SM * A.unsqueeze(-1)).reshape(-1, 1, self.num_channel) # [B, 1, H * C]
#             G = self.act(self.product_gate(SM))
#             G1, G2 = torch.split(G, self.num_channel, -1)
#             TM = G1 * OT[source] * + G2 * edge_attrs.unsqueeze(2)

#             # M_i
#             SM = scatter_sum(SM, target, dim=0, dim_size=node_feats.size(0)) / self.avg_num_neighbors
#             TM = scatter_sum(TM, target, dim=0, dim_size=node_feats.size(0)) / self.avg_num_neighbors
#             SM = self.linear_down(SM)
#             S1, S2, S3 = torch.split(SM, self.num_channel, -1)

#             dS = T1_T2 * S2 + S3
#             dT = T3 * S1 + TM

#             M = torch.cat([dS, dT], dim=1)

            
#         return M, sc, None