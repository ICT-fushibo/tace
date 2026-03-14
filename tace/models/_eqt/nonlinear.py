################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import torch
import equitorch as eqt
from equitorch.irreps import check_irreps
from e3nn import o3

from ..activation import ACTIVATION
from ..mlp import MLP
from .linear import Linear


GateNonlinear = eqt.nn.Gate


class NormNonlinear(torch.nn.Module):
    def __init__(
        self,
        irreps: eqt.irreps.Irreps,
        num_channel: int,
        activation: torch.nn.Module,
        bias: bool = True,
    ):
        super().__init__()

        self.irreps = check_irreps(irreps)
        self.num_irreps = len(self.irreps)
        self.num_channel = num_channel
        self.irreps_dim = self.irreps.dim

        # self.norm1 = eqt.nn.LayerRMSNorm(
        #     f"{self.num_irreps}x0e", 
        #     num_channel,
        #     scaled=True,
        # )

        self.norm_fn = eqt.nn.Norm(irreps=self.irreps, scaled=True)
        # self.weight = torch.nn.Parameter(torch.empty(self.num_irreps, self.num_channel))
        # with torch.no_grad():
        #     ls = torch.tensor([ir.l for ir in self.irreps], dtype=torch.get_default_dtype())
        #     init_val = 1 / torch.sqrt(2*ls + 1).unsqueeze(1)
        #     self.weight.data = init_val.repeat(1, self.num_channel)
        self.weight = torch.nn.Parameter(torch.ones(self.num_irreps, self.num_channel))
        self.bias = torch.nn.Parameter(torch.empty(self.num_irreps, self.num_channel))
        torch.nn.init.zeros_(self.bias)

        self.activation = activation

        self.slices = []
        start = 0
        for ir in self.irreps:
            dim = ir.dim
            self.slices.append(slice(start, start + dim))
            start += dim

        assert start == self.irreps_dim

    def forward(self, x: torch.Tensor):
        '''
        x: (B, M, C)
        '''
        norm = self.norm_fn(x)
        norm = norm * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        # norm = self.norm1(norm)
        norm = self.activation(norm)

        out_chunks = []
        for i, sl in enumerate(self.slices):
            scale = norm[..., i:i+1, :]
            out_chunks.append(x[..., sl, :] * scale)

        return torch.cat(out_chunks, dim=-2)

class GLUNonlinear(torch.nn.Module):
    def __init__(
        self,
        irreps_in: eqt.irreps.Irreps,
        irreps_out: eqt.irreps.Irreps,
        num_channel: int,
        num_hidden_channel: int,
        activation: torch.nn.Module,
        bias: bool,
    ):
        super().__init__()

        self.irreps_in = check_irreps(irreps_in)
        self.irreps_out = check_irreps(irreps_out)
        self.num_irreps = len(self.irreps_out)
        self.num_channel = num_channel
        self.irreps_dim = self.irreps_out.dim

        self.linear1 =Linear(
            self.irreps_in,
            self.irreps_out,
            num_channel,
            num_hidden_channel,
            bias=bias,
        )
        self.linear2 =Linear(
            self.irreps_in,
            self.irreps_out,
            num_channel,
            num_hidden_channel,
            bias=bias,
        )
        # self.linear2 = Linear(
        #     f"{len(self.irreps_in)}x0e",
        #     f"{len(self.irreps_out)}x0e",
        #     num_channel,
        #     num_hidden_channel,
        #     bias=bias,
        # )
        # self.norm1 = eqt.nn.LayerRMSNorm(
        #     f"{len(self.irreps_out)}x0e", 
        #     num_hidden_channel,
        #     scaled=True,
        # )
        self.norm_fn = eqt.nn.Norm(irreps=self.irreps_out, scaled=True)
        self.weight = torch.nn.Parameter(torch.ones(self.num_irreps, self.num_channel))
        self.bias = torch.nn.Parameter(torch.empty(self.num_irreps, self.num_channel))
        torch.nn.init.zeros_(self.bias)
        self.activation = activation

        self.slices = []
        start = 0
        for ir in self.irreps_out:
            dim = ir.dim
            self.slices.append(slice(start, start + dim))
            start += dim

        assert start == self.irreps_dim

    def forward(self, x: torch.Tensor):
        '''
        x: (B, M, C)
        '''
        out1 = self.linear1(x)
        out2 = self.linear2(x)
        norm = self.norm_fn(out2)
        norm = norm * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        # norm = self.norm1(norm)
        norm = self.activation(norm)

        out_chunks = []
        for i, sl in enumerate(self.slices):
            scale = norm[..., i:i+1, :]
            out_chunks.append(out1[..., sl, :] * scale)

        return torch.cat(out_chunks, dim=-2)
    

class GridNonlinear(torch.nn.Module):
    def __init__(
        self,
        irreps: eqt.irreps.Irreps,
        num_channel: int,
        activation: torch.nn.Module,
        bias: bool = False,
    ):
        super().__init__()

        self.irreps = check_irreps(irreps)

        # Default truncation may not enough
        lmax = max(ir.l for ir in irreps)
        self.truncation = lmax
        self.num_latitude = 2 * (self.truncation + 1)
        self.num_longitude = 2 * (self.truncation+ 1) + 1

        self.mlp = MLP(
            [num_channel] * 4,
            bias=bias,
            layer_norm=False,
            act=activation,
        )

        to_s2 = o3.ToS2Grid(
            self.truncation, 
            (self.num_latitude, self.num_longitude), 
            normalization="component",
        )
        from_s2 = o3.FromS2Grid(
            (self.num_latitude, self.num_longitude), 
            self.truncation, 
            normalization="component",
        )

        self.register_buffer(
            "to_grid", 
            torch.einsum(
                "mbi, am -> bai", to_s2.shb, to_s2.sha
            ).detach(),
            persistent=False,
        )
        self.register_buffer(
            "from_grid", 
            torch.einsum(
                "am, mbi -> bai", from_s2.sha, from_s2.shb
            ).detach(),
            persistent=False,
        )

    def _to_grid(self, x: torch.Tensor) -> torch.Tensor:           
        return torch.einsum("bai, Bic -> Bbac", self.to_grid, x)

    def _from_grid(self, x: torch.Tensor) -> torch.Tensor:       
        return torch.einsum("bai, Bbac -> Bic", self.from_grid, x)
    
    def forward(self, x: torch.Tensor):
        '''
        x: (B, M, C)
        '''
        grid = self._to_grid(x)
        B, b, a, C = grid.shape
        freq = self._from_grid(self.mlp(grid.reshape(-1, C)).reshape(B, b, a, C))
        return freq


# class MoENonlinear(torch.nn.Module):   
#     def __init__(
#         self, 
#         irreps: eqt.irreps.Irreps, 
#         activation: torch.nn.Module,
#         irrep_wise: bool, 
#         num_channel: int,
#         bias: bool,
#         *,
#         num_experts: int,
#         top_k: int,
#         aux_loss_weight: float,
#         z_loss_weight: float,
#     ):
#         super().__init__()
        
#         self.num_experts = num_experts
#         self.top_k = top_k
#         self.aux_loss_weight = aux_loss_weight
#         self.z_loss_weight = z_loss_weight

#         self.expert_out_dim = (irreps + f"{len(irreps)}x0e").dim
#         self.experts = torch.nn.ModuleList(
#             Linear(
#                 irreps,
#                 irreps + f"{len(irreps)}x0e",
#                 num_channel,
#                 num_channel,
#                 bias=bias,
#             ) for _ in range(self.num_experts)
#         )

#         self.nonlinear = GateNonlinear(
#                 irreps,
#                 activation=activation,
#                 irrep_wise=irrep_wise,
#             )

#         # self.norm_fn = eqt.nn.Norm(irreps=irreps, scaled=True)
#         # self.router = Linear(
#         #     f"{len(irreps) * num_channel}x0e" ,
#         #     f"{num_experts}x0e",
#         #     1,
#         #     1,
#         #     bias=bias,
#         # )
 
#         self.router = Linear(
#             irreps,
#             "0e",
#             num_channel,
#             num_experts,
#             bias=bias,
#         )

#     def forward(self, x: torch.Tensor, return_aux_loss: bool = False):  # x: [B, M, C]
#         B, M, C = x.shape
#         E = self.num_experts
#         k = self.top_k
 
#         # norm = self.norm_fn(x).view(B, -1, 1)       # [B, len(irreps)*C, 1]
#         # gate_logits = self.router(norm).squeeze(-1) # (B, E)
#         gate_logits = self.router(x).squeeze(1) # (B, E)
#         gate_probs = torch.softmax(gate_logits, dim=-1)

#         topk_probs, topk_idx = torch.topk(gate_probs, k, dim=-1)  # (B, k)
#         topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

#         aux_loss = None

#         if self.training or return_aux_loss:
#             # importance
#             p = gate_probs.mean(dim=0)                 # [E]

#             # load
#             flat_expert = topk_idx.reshape(-1)
#             counts = torch.bincount(flat_expert, minlength=E).float()
#             f = counts / (B * k)

#             load_balance_loss = E * torch.sum(f * p)

#             # z-loss (router stability)
#             log_z = torch.logsumexp(gate_logits, dim=-1)
#             z_loss = torch.mean(log_z ** 2)

#             aux_loss = (
#                 self.aux_loss_weight * load_balance_loss
#                 + self.z_loss_weight * z_loss
#             )

#         flat_expert = topk_idx.reshape(-1)          # (B*k,)
#         flat_weight = topk_probs.reshape(-1)        # (B*k,)
#         # Repeat tokens for top-k
#         flat_x = x.repeat_interleave(k, dim=0)      # (B*k, M, C)

#         # Sort by expert id (group tokens per expert)
#         sort_idx = torch.argsort(flat_expert)

#         flat_expert = flat_expert[sort_idx]
#         flat_weight = flat_weight[sort_idx]
#         flat_x = flat_x[sort_idx]

#         # ---- expert output M dimension ----
#         M_out = self.expert_out_dim

#         flat_out = torch.zeros(
#             flat_x.size(0), M_out, C,
#             device=x.device,
#             dtype=x.dtype
#         )

#         expert_counts = torch.bincount(flat_expert, minlength=E)

#         start = 0
#         for expert_id in range(E):
#             count = expert_counts[expert_id].item()
#             if count == 0:
#                 continue

#             end = start + count
#             expert_x = flat_x[start:end]              # (n_i, M, C)

#             out = self.experts[expert_id](expert_x)   # (n_i, M_out, C)

#             flat_out[start:end] = out
#             start = end

#         flat_out = flat_out * flat_weight.view(-1, 1, 1)

#         out = torch.zeros(
#             B, M_out, C,
#             device=x.device,
#             dtype=x.dtype
#         )

#         token_idx = sort_idx // k
#         out.index_add_(0, token_idx, flat_out)

#         out = self.nonlinear(out)  

#         return out, aux_loss