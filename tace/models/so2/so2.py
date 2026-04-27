''''
Copy from EquiformerV3.

MIT License

Copyright (c) 2026 The Atomic Architects

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import torch
import math


from ..mlp import ScaledSigmoid, ScaledSiLU
from .utils import so2_expand_index


class SO2MLinear(torch.nn.Module):
    """
        Perform an SO(2) linear operation to features corresponding to +- m

        Args:
            m (int):                    Order of the spherical harmonic coefficients
            num_in_channels (int):      Number of input channels
            num_out_channels (int):     Number of output channels
            lmax (int):                 Maximum degrees (l)
            mmax (int):                 Maximum order (m)
    """
    def __init__(
        self,
        m,
        num_in_channels,
        num_out_channels,
        lmax,
        mmax,
    ):
        super().__init__()

        self.m = m
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.lmax = lmax
        self.mmax = mmax

        num_m_components = self.lmax - self.m + 1
        assert num_m_components > 0

        self.in_features = num_m_components * self.num_in_channels
        self.out_features = num_m_components * self.num_out_channels
        
        self.fc = torch.nn.Linear(
            self.in_features,
            (2 * self.out_features),
            bias=False,
        )
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        a = 1.0 / math.sqrt(self.fc.in_features) 
        torch.nn.init.uniform_(self.fc.weight, -a, a)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias, 0)
        self.fc.weight.data.mul_(1 / math.sqrt(2))

    def forward(self, x_m, concat_outputs=True):
        x_m = self.fc(x_m)
        x_r = x_m.narrow(2, 0, self.out_features)
        x_i = x_m.narrow(2, self.out_features, self.out_features)
        x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1) 
        x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1)
        x_out = (x_m_r, x_m_i)
        if concat_outputs:
            x_out = torch.cat(x_out, dim=1)
        return x_out


class SO2Linear(torch.nn.Module):
    """
        Perform SO(2) linear operations to all m (orders) components

        Args:
            num_in_channels (int):      Number of input channels
            num_out_channels (int):     Number of output channels
            lmax (int):                 Maximum degrees (l)
            mmax (int):                 Maximum order (m)
            extra_m0_out_channels (int):    If not None, return `outputs` (torch.Tensor) and `extra_m0_features` (torch.Tensor).
    """
    def __init__(
        self,
        mmax,
        lmax,
        num_in_channels,
        num_out_channels,
        extra_m0_out_channels=None
    ):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.lmax = lmax
        self.mmax = mmax
        self.extra_m0_out_channels = extra_m0_out_channels

        # for m = 0
        num_in_channels_m0 = (self.lmax + 1) * self.num_in_channels
        num_out_channels_m0 = (self.lmax + 1) * self.num_out_channels
        if self.extra_m0_out_channels is not None:
            self.num_channels_m0_list = [self.extra_m0_out_channels, num_out_channels_m0]
            num_out_channels_m0 = num_out_channels_m0 + self.extra_m0_out_channels
        self.fc_m0 = torch.nn.Linear(num_in_channels_m0, num_out_channels_m0)

        # SO(2) linear for non-zero m
        self.so2_m_linear = torch.nn.ModuleList()
        for m in range(1, self.mmax + 1):
            self.so2_m_linear.append(
                SO2MLinear(
                    m,
                    self.num_in_channels,
                    self.num_out_channels,
                    self.lmax,
                    self.mmax,
                )
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        a = 1.0 / math.sqrt(self.fc_m0.in_features) 
        torch.nn.init.uniform_(self.fc_m0.weight, -a, a)
        if self.fc_m0.bias is not None:
            torch.nn.init.constant_(self.fc_m0.bias, 0)

    def forward(self, x):
        """
            1.  `x` shape: [num_edges, num_m_components, num_channels]
            2.  We assume the layout of m components is (0, 0, ...), (1, 1, ...), ...
        """
        num_edges = x.shape[0]
        outputs = []

        # Compute m=0 coefficients separately since they only have real values (no imaginary)
        x_m0 = x.narrow(1, 0, (self.lmax + 1))
        x_m0 = x_m0.reshape(num_edges, -1)
        x_m0 = self.fc_m0(x_m0)

        x_m0_extra = None
        # extract extra m0 features
        if self.extra_m0_out_channels is not None:
            x_m0_extra, x_m0 = torch.split(x_m0, self.num_channels_m0_list, dim=1)

        x_m0 = x_m0.view(num_edges, -1, self.num_out_channels)
        outputs.append(x_m0)

        # Compute the values for the m > 0 coefficients
        offset = self.lmax + 1
        for m in range(1, self.mmax + 1):
            x_m = x.narrow(1, offset, 2 * (self.lmax + 1 - m))
            offset = offset + 2 * (self.lmax + 1 - m)
            x_m = x_m.reshape(num_edges, 2, -1)
            """
            x_m = self.so2_m_linear[m - 1](x_m, concat_outputs=True)
            x_m = x_m.view(num_edges, -1, self.num_out_channels)
            out.append(x_m)
            """
            # Replace the original one with the followings to prevent one `torch.cat()` for each m > 0
            x_m = self.so2_m_linear[m - 1](x_m, concat_outputs=False)
            x_m_pos, x_m_neg = x_m[0], x_m[1]
            x_m_pos = x_m_pos.view(num_edges, -1, self.num_out_channels)
            x_m_neg = x_m_neg.view(num_edges, -1, self.num_out_channels)
            outputs.append(x_m_pos)
            outputs.append(x_m_neg)
            
        outputs = torch.cat(outputs, dim=1)

        if self.extra_m0_out_channels is not None:
            return outputs, x_m0_extra
        else:
            return outputs
  

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
    

class SO2TensorProduct(torch.nn.Module):

    def enumerate_paths(self, m3):
        paths = []
        for m1 in range(0, self.mmax + 1):
            for m2 in range(0, self.mmax + 1):
                if m1 == 0 and m2 == 0 and m3 == 0:
                    paths.append((m1, m2, "sum"))
                else:
                    if m1 + m2 == m3:
                        paths.append((m1, m2, "sum"))
                    elif abs(m1 - m2) == m3:
                        paths.append((m1, m2, "diff"))
                    else:
                        raise
        return paths
    
    def __init__(
        self,
        mmax,
        lmax,
        num_channel,
    ):
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel

        self.m3_instructions = []
        for m3 in range(0, mmax + 1):
            instructions = []
            in_feats = 0
            for (m1, m2, mode) in self.enumerate_paths(m3):
                n_l1 = lmax - m1 + 1
                n_l2 = lmax - m2 + 1
                in_feats += n_l1 * n_l2 * num_channel
                instructions.append([m1, m2, mode])
            self.m3_instructions.append(instructions)

    def split(self, x):
        B = x.shape[0]
        offset = 0
        out = {}

        # m=0
        size = self.lmax + 1
        out[0] = x[:, offset:offset + size]
        offset += size

        # m>0
        for m in range(1, self.mmax + 1):
            size = self.lmax + 1 - m
            xm = x[:, offset:offset + 2 * size]
            xm = xm.view(B, 2, size, self.num_channel)
            out[m] = xm
            offset += 2 * size

        return out

    def couple(self, x1, x2, mode):
        a1, b1 = x1[:, 0], x1[:, 1]
        a2, b2 = x2[:, 0], x2[:, 1]

        a1 = a1.unsqueeze(2)
        b1 = b1.unsqueeze(2)
        a2 = a2.unsqueeze(1)
        b2 = b2.unsqueeze(1)

        if mode == "sum":
            real = a1 * a2 - b1 * b2
            imag = a1 * b2 + b1 * a2
        else:
            real = a1 * a2 + b1 * b2
            imag = b1 * a2 - a1 * b2

        out = torch.stack([real, imag], dim=1)
        return out.view(out.size(0), 2, -1, out.size(-1))

    def forward(self, x, y, ws=None):

        B = x.size(0)
        C = self.num_channel

        x1_dict = self.split(x)
        x2_dict = self.split(y)

        outputs = []
        z0 = x1_dict[0].unsqueeze(2) * x2_dict[0].unsqueeze(1)  # [B, n_l, C]
        z0 = z0.view(B, -1, C)  # [B,n_l,C]

        outputs.append(z0)

        instructions0 = self.m3_instructions[m3]
        for m1, m2, mode in instructions0:
            z = self.couple(x1_dict[m1], x2_dict[m2], mode) # # [B,2,P,C]
            pos_list.append(z[:, 0, :, :])
            neg_list.append(z[:, 1, :, :])  

        for m3 in range(1, self.mmax + 1):
            instructions = self.m3_instructions[m3]
            pos_list = []
            neg_list = []
            for m1, m2, mode in instructions:
                z = self.complex_outer(x1_dict[m1], x2_dict[m2], mode) # # [B,2,P,C]
                pos_list.append(z[:, 0, :, :])
                neg_list.append(z[:, 1, :, :])
            pos = torch.cat(pos_list, dim=1)
            neg = torch.cat(neg_list, dim=1)
            outputs.append(torch.cat([pos, neg], dim=1))

        out = torch.cat(outputs, dim=1)

        return out
    

class SO2GatedLinearUnit(torch.nn.Module):
    def __init__(
        self,
        mmax,
        lmax,
        num_channel,
    ):
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.num_components, expand_index = so2_expand_index(mmax, lmax)
        self.register_buffer('expand_index', expand_index, persistent=False)

        self.activation = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            B = x.size(0)
            g = self.activation(g).view(B, self.num_components, -1)
            g = torch.index_select(g, dim=1, index=self.expand_index)
            return g * x 


class SO2e3nnGatedLinearUnit(torch.nn.Module):
    def __init__(
        self,
        mmax,
        lmax,
        num_channel,
    ):
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.num_components, expand_index = so2_expand_index(mmax, lmax, start=1)
        self.register_buffer('expand_index', expand_index, persistent=False)

        self.scalar_activation = ScaledSiLU()
        self.tensor_activation = ScaledSigmoid()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            # m = 0
            x_scalar = x[:, :self.lmax+1, :]
            x_scalar = self.scalar_activation(x_scalar)

            # m > 0
            B = x.size(0)
            g = self.tensor_activation(g).view(B, self.num_components, -1)
            g = torch.index_select(g, dim=1, index=self.expand_index)
            x_tensor = x[:, self.lmax+1:, :]
            x_tensor = g * x_tensor
            return torch.cat([x_scalar, x_tensor], dim=1) 
    
class SO2NormLinearUnit(torch.nn.Module):
    def __init__(
        self,
        mmax,
        lmax,
        num_channel,
    ):
        super().__init__()

        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.num_components, expand_index = so2_expand_index(mmax, lmax)
        self.register_buffer('expand_index', expand_index, persistent=False)


        offset = lmax + 1 
        slices = [slice(0, offset)]
        for m in range(1, self.mmax + 1):
            length = (self.lmax + 1 -m) * 2
            slices.append(slice(offset, offset+length))
            offset += length
        self.slices = slices
        self.activation = torch.nn.Sigmoid()
        scale = torch.tensor([1] * (lmax+1) + [2] * (self.num_components - (lmax+1)))
        self.weight = torch.nn.Parameter(
            torch.randn(self.num_components, self.num_channel) 
            / scale.unsqueeze(-1)
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(self.num_components, self.num_channel)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B = x.size(0)   
        C = x.size(-1)   

        norms = []
        # m = 0
        irreps = x[:, self.slices[0], :]
        norms.append(irreps.pow(2))

        # m > 0
        for s in self.slices[1:]:
            irreps = x[:, s, :]
            irreps = irreps.view(B, 2, -1, C)
            norms.append(irreps.pow(2).sum(dim=1))
        g = torch.cat(norms, dim=1)
        g = g * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        g = self.activation(g)
        g = torch.index_select(g, dim=1, index=self.expand_index)
        return g * x 

