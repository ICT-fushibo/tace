# ################################################################################
# # Authors: Zemin Xu
# # License: MIT, see LICENSE.md
# ################################################################################

# import math


# import torch
# import torch.nn.functional as F


# from ..mlp import ScaledSigmoid
# from .utils import so2_expand_index


# class SO2Linear(torch.nn.Module):
#     def __init__(
#         self,
#         mmax,
#         lmax,
#         num_channel_in,
#         num_channel_out,
#     ):
#         super().__init__()

#         self.mmax = mmax
#         self.lmax = lmax

#         if isinstance(num_channel_in, int):
#             self.in_channels = [num_channel_in * min((lmax+1-m), mmax+1) for m in range(lmax + 1)]
#         else:
#             assert len(num_channel_in) == mmax + 1
#             self.in_channels = num_channel_in

#         if isinstance(num_channel_out, int):
#             self.out_channels = [num_channel_out * min((lmax+1-m), mmax+1) for m in range(lmax + 1)]
#         else:
#             assert len(num_channel_out) == mmax + 1
#             self.out_channels = num_channel_out

#         self.so2_m_linear = torch.nn.ModuleList()

#         for m in range(0, mmax + 1):

#             Cin = self.in_channels[m]
#             Cout = self.out_channels[m]

#             if m == 0:
#                 fc = torch.nn.Linear(Cin, Cout, bias=True)
#                 a = 1.0 / math.sqrt(fc.in_features)
#                 torch.nn.init.uniform_(fc.weight, -a, a)
#                 torch.nn.init.zeros_(fc.bias)
#             else:
#                 fc = torch.nn.Linear(Cin, Cout * 2, bias=False)
#                 a = 1.0 / math.sqrt(fc.in_features)
#                 torch.nn.init.uniform_(fc.weight, -a, a)
#                 fc.weight.data.mul_(1 / math.sqrt(2))

#             self.so2_m_linear.append(fc)


#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         B = x.size(0)
#         x = x.view(B, -1)
#         outputs = []

#         offset = 0

#         for m in range(0, self.mmax + 1):

#             n_l = self.lmax - m + 1
#             Cin = self.in_channels[m]
#             Cout = self.out_channels[m]

#             fc = self.so2_m_linear[m]

#             if m == 0:
#                 size = n_l
#                 x_m = x[:, offset:offset + Cin]
#                 offset += Cin
#                 x_m = fc(x_m)
#                 x_m = x_m.view(B, n_l, -1)
#                 outputs.append(x_m)
#             else:
#                 size = 2 * n_l
#                 x_m = x[:, offset:offset + 2 * Cin]
#                 offset += 2 * Cin
#                 x_m = x_m.reshape(B, 2, Cin)
#                 x_m = fc(x_m)
#                 x_r = x_m[:, :, :Cout]
#                 x_i = x_m[:, :, Cout:]
#                 # SO(2) equivariant combine
#                 x_m_r = x_r[:, 0] - x_i[:, 1]
#                 x_m_i = x_r[:, 1] + x_i[:, 0]
#                 x_m_r = x_m_r.view(B, n_l, -1)
#                 x_m_i = x_m_i.view(B, n_l, -1)
#                 outputs.append(x_m_r)
#                 outputs.append(x_m_i)

#         outputs = torch.cat(outputs, dim=1)

#         return outputs
    
