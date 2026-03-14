################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

# class EdgeMoE(torch.nn.Module):
#     def __init__(
#         self,
#         channels: List[int],
#         bias: bool = True,
#         layer_norm: bool = False,
#         act: Optional[str | torch.nn.Module] = "silu",
#         num_experts: int = 4,
#         top_k: int = 2,
#         aux_loss_weight: float = 1e-2,  
#         z_loss_weight: float = 1e-2, 

#     ):
#         super().__init__()

#         assert channels == 3, "You should use only one hidden layer in EdgeMoe"

#         self.out_dim = channels[-1]
#         self.num_experts = num_experts
#         self.top_k = top_k
#         self.aux_loss_weight = aux_loss_weight
#         self.z_loss_weight = z_loss_weight
  
#         self.experts = torch.nn.ModuleList(
#             MLP(
#                 channels,
#                 layer_norm=layer_norm,
#                 act=act,
#                 bias=bias,
#             )
#             for _ in range(num_experts)
#         )

#         self.router = MLP(
#             [channels[0], num_experts],
#             layer_norm=False,
#             act=None,
#             bias=bias,
#         )

#     def forward(self, x: torch.Tensor, return_aux_loss: bool = False):
#         N = x.size(0)
#         E = self.num_experts
#         k = self.top_k


#         gate_logits = self.router(x)                # (N, E)

#         gate_probs = torch.softmax(gate_logits, dim=-1)

#         topk_probs, topk_idx = torch.topk(gate_probs, k, dim=-1)
#         topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

#         aux_loss = None

#         if self.training or return_aux_loss:
#         # if True:
#             p = gate_probs.mean(dim=0)                      # (E,)
#             flat_expert = topk_idx.reshape(-1)
#             counts = torch.bincount(flat_expert, minlength=E).float()
#             f = counts / (N * k)

#             # print("p:", p.detach().cpu())
#             # print("f:", f.detach().cpu())

#             load_balance_loss = E * torch.sum(f * p)

#             log_z = torch.logsumexp(gate_logits, dim=-1)
#             z_loss = torch.mean(log_z ** 2)

#             aux_loss = (
#                 self.aux_loss_weight * load_balance_loss
#                 + self.z_loss_weight * z_loss
#             )

#         flat_expert = topk_idx.reshape(-1)           # (N*k,)
#         flat_weight = topk_probs.reshape(-1)         # (N*k,)
#         flat_x = x.repeat_interleave(k, dim=0)       # (N*k, d)

#         sort_idx = torch.argsort(flat_expert)
#         flat_expert = flat_expert[sort_idx]
#         flat_weight = flat_weight[sort_idx]
#         flat_x = flat_x[sort_idx]

#         flat_out = torch.zeros(
#             flat_x.size(0),
#             self.out_dim,
#             device=x.device,
#             dtype=x.dtype,
#         )

#         expert_counts = torch.bincount(flat_expert, minlength=E)

#         start = 0
#         for expert_id in range(E):
#             count = expert_counts[expert_id].item()
#             if count == 0:
#                 continue

#             end = start + count
#             expert_x = flat_x[start:end]

#             out = self.experts[expert_id](expert_x)
#             flat_out[start:end] = out

#             start = end

#         flat_out = flat_out * flat_weight.unsqueeze(-1)

#         out = torch.zeros(N, self.out_dim, device=x.device, dtype=x.dtype)
#         out.index_add_(
#             0,
#             sort_idx // k,
#             flat_out
#         )

#         return out, aux_loss
