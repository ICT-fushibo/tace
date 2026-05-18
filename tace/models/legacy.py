# class OamACE(Product):
#     def _setup(self):

#         self.linear_up = Linear(
#             self.irreps_in,
#             self.irreps_hidden1,
#             bias=self.use_bias,
#         ) if self.num_channel != self.num_hidden_channel else torch.nn.Identity()

#         self.ace = uuuTensorProduct(
#             irreps_in1=self.irreps_hidden1,
#             irreps_in2=self.irreps_hidden1[:1] + self.irreps_hidden1,
#             irreps_out=self.irreps_hidden2,
#             l1l2=self.l1l2,
#             l3s=self.l3s,
#             trainable=True,
#         ) 
#         self.coef = torch.nn.Parameter(torch.randn(self.num_elements, self.ace.weight_numel))

#         irreps_gated = self.irreps_hidden2
#         irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in self.irreps_hidden2)
#         self.nonlinearity = GatedLinearUnit(
#             irreps_gates=irreps_gates,
#             act_gates=[ACTIVATION[self.nonlinear_act]()] * len(irreps_gates),
#             irreps_gated=irreps_gated,
#         )

#         self.linear_gate = Linear(
#             self.ace.irreps_out.simplify(),
#             self.nonlinearity.irreps_in,
#             bias=self.use_bias,
#         )

#         self.linear_down = Linear(
#             self.irreps_hidden2,
#             self.irreps_out,
#             bias=self.use_bias
#         )    
        
#     def forward(
#             self, 
#             node_feats: torch.Tensor, 
#             node_attrs: torch.Tensor,
#             sc: torch.Tensor,
#             batch: torch.Tensor,
#         ) -> torch.Tensor:

#         node_feats = self.linear_up(node_feats)
#         ones = node_feats.new_ones(node_feats.size(0), self.num_hidden_channel)

#         corr_feats = self.ace(
#             node_feats, 
#             torch.cat(
#                 [
#                     ones,
#                     node_feats,
#                 ],
#                 dim=-1,
#             ), 
#             torch.einsum('bz, zi -> bi', node_attrs, self.coef),
#         )

#         outs = self.linear_down(self.nonlinearity(self.linear_gate(corr_feats)))

#         if sc is not None:
#             outs = outs + sc

#         return outs
    



# class SO2ScatterTensorProduct(torch.nn.Module):
#     def __init__(
#         self,
#         mmax: int,
#         lmax: int,
#         num_channel: int,
#         num_hidden_channel: int,
#         num_head: Union[int, None],
#         num_channel_per_head: int,
#         is_scalar_tp: bool,
#         is_so2_layout: bool,
#         use_so2_edge_ace: bool,
#         edge_nonlinear: Union[str, None],
#         num_elements: int,
#         so2_angular_basis: SO3Rotation,
#         reshape_in: LayoutTransform,
#         reshape_out: LayoutTransform,
#     ) -> None:
#         super().__init__()

#         self.mmax = mmax
#         self.lmax = lmax
#         self.num_channel = num_channel
#         self.num_hidden_channel = num_hidden_channel or self.num_channel
#         self.is_so2_layout = is_so2_layout
#         self.is_scalar_tp = is_scalar_tp
#         self.edge_nonlinear = edge_nonlinear
#         self.use_so2_edge_ace = use_so2_edge_ace

#         self.so2_angular_basis = so2_angular_basis
#         self.reshape_in = reshape_in
#         self.reshape_out = reshape_out

#         # Transformer
#         self.num_head = num_head or 1
#         self.num_channel_per_head = num_channel_per_head or num_channel
#         assert self.num_hidden_channel % self.num_head == 0
#         if self.num_head > 1:
#             self.use_transformer = True
#         else:
#             self.use_transformer = False

#         Cin = num_channel if not self.use_transformer else num_channel * 2
#         Cout = num_channel
#         self.num_out_channel = Cout

        
#         if self.is_so2_layout and not self.is_scalar_tp:
#             self.num_components, expand_index = so2_expand_index(self.mmax, self.lmax)
#             self.weight_numel = self.num_components * Cin
#             self.register_buffer('expand_index', expand_index, persistent=False)
#         else:
#             self.num_components, expand_index = so3_expand_index(self.mmax, self.lmax)
#             self.weight_numel = self.num_components * Cin
#             self.register_buffer('expand_index', expand_index, persistent=False)

#         self.num_gates = 0
#         for m in range(mmax + 1):
#             if self.use_so2_edge_ace:
#                 self.num_gates += lmax + 1
#             else:
#                 self.num_gates += lmax + 1 -m

#         if self.is_scalar_tp:
#             pass
#         else:
#             assert edge_nonlinear is not None, "We force to use SO2 edge nonlinear in TACE"

#             if self.use_transformer: # TODO
#                 self.linear_alpha = SO2Linear(
#                     0, 
#                     lmax, 
#                     Cin, 
#                     num_head * num_channel_per_head,
#                     num_components_out=[1]
#                 )
#                 self.alpha_norm = torch.nn.LayerNorm(self.num_channel_per_head)
#                 self.alpha_act = SmoothLeakyReLU()
#                 self.alpha_dot = torch.nn.Parameter(torch.randn(self.num_head, self.num_channel_per_head))
#                 std = 1.0 / math.sqrt(self.num_channel_per_head)
#                 torch.nn.init.uniform_(self.alpha_dot, -std, std)
#                 self.attn_softmax = GraphSoftmax()

#             if self.use_so2_edge_ace:
#                 self.ace = SO2EdgeProductBasis(
#                     mmax, 
#                     lmax, 
#                     self.num_hidden_channel,
#                     num_elements=num_elements,
#                 )
#                 self.linear_up = SO2Linear(
#                     mmax,
#                     lmax,
#                     Cin,
#                     self.num_hidden_channel,     
#                     num_components_in=None,
#                     num_components_out=[self.num_gates + lmax+1] + [lmax+1] * (lmax),
#                 )
#                 self.nonlinearity = SO2Gate(
#                     mmax,
#                     lmax,
#                     self.num_hidden_channel,     
#                     channel_wise=True
#                 )
#                 self.linear_down = SO2Linear(
#                     mmax,
#                     lmax,
#                     self.num_hidden_channel,     
#                     Cout,     
#                     num_components_in=[lmax+1] * (lmax+1),
#                     num_components_out=None,
#                 )
#             else:
#                 self.linear_up = SO2Linear(
#                     mmax,
#                     lmax,
#                     Cin,
#                     self.num_hidden_channel,    
#                     num_components_in=None,
#                     num_components_out=[self.num_gates + lmax+1] + [lmax+1-m for m in range(1, mmax+1)],
#                 )
#                 self.nonlinearity = SO2Gate(
#                     mmax,
#                     lmax,
#                     self.num_hidden_channel,    
#                     channel_wise=False
#                 )
#                 self.linear_down = SO2Linear(
#                     mmax,
#                     lmax,
#                     self.num_hidden_channel,    
#                     Cout,     
#                 )     


#     def forward(
#             self, 
#             x: torch.Tensor, # [B, so_m, C]
#             y: torch.Tensor,  # node_attrs here
#             w: torch.Tensor, 
#             edge_index: torch.Tensor,
#             cutoff: torch.Tensor,
#         ) -> torch.Tensor:

#         num_nodes = x.size(0)
#         num_edges = w.size(0)
#         x = self.reshape_in(x) 

#         if self.use_transformer:
#             x = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=-1)
#         else:
#             x = x[edge_index[0]]

#         if self.is_scalar_tp:
#             w = w.view(num_edges, self.num_components, -1)
#             m_ij = torch.einsum(
#                 'bij, bjc -> bic', 
#                     self.so2_angular_basis.wigner_inv.narrow(2, 0, (self.lmax + 1)),
#                     x * w
#             ) # first so3 tp, no nonlinearity is required here
#         else:
#             w = w.view(num_edges, self.num_components, -1)
#             w = torch.index_select(w, dim=1, index=self.expand_index)

#             if self.is_so2_layout:
#                 m_ij = self.so2_angular_basis.rotate(x)
#                 m_ij = m_ij * w
#             else:
#                 m_ij =  x * w
#                 m_ij = self.so2_angular_basis.rotate(m_ij)

#             if self.use_transformer:
#                 alpha = self.linear_alpha(m_ij)
#             m_ij = self.linear_up(m_ij)

#             gate = m_ij.narrow(1, 0, self.num_gates)
#             m_ij = m_ij.narrow(
#                 1,
#                 self.num_gates,
#                 m_ij.size(1) - self.num_gates
#             )
#             if hasattr(self, 'ace'):
#                 m_ij = self.ace(m_ij, y, edge_index)
#             m_ij = self.nonlinearity(m_ij, gate) 
#             m_ij = self.linear_down(m_ij)
            
#             if self.use_transformer:
#                 alpha = alpha.reshape(-1, self.num_head, self.num_channel_per_head)
#                 alpha = self.alpha_norm(alpha)
#                 alpha = self.alpha_act(alpha)
#                 alpha = torch.einsum('bik, ik -> bi', alpha, self.alpha_dot)
#                 alpha = self.attn_softmax(alpha, edge_index[1], num_nodes=num_nodes, exp_rescale=cutoff)
#                 if cutoff is not None:
#                     alpha = alpha * cutoff
#                 # one = scatter_sum(
#                 #     alpha,
#                 #     edge_index[1],
#                 #     dim=0,
#                 #     dim_size=num_nodes
#                 # )
#                 # print(one[0])
#                 alpha = alpha.view(alpha.size(0), 1, self.num_head, 1)
#                 attn = m_ij
#                 attn = attn.view(attn.size(0), attn.size(1), self.num_head, -1)
#                 attn = attn * alpha
#                 attn = attn.view(attn.size(0), attn.size(1), -1)
#                 m_ij = attn

#             m_ij = self.so2_angular_basis.rotate_inv(m_ij)

#         m_i = scatter_sum(
#                 m_ij, 
#                 edge_index[1], 
#                 dim=0, 
#                 dim_size=num_nodes,
#         )
#         return self.reshape_out.inverse(m_i)

# class SO2EdgeProductBasis(torch.nn.Module):
#     def __init__(
#         self,
#         mmax: int,
#         lmax: int,
#         num_channel: int,
#         m1m2: Union[str, None] = '<=',
#         internal_weights: bool = False,
#     ):
#         super().__init__()

#         self.mmax = mmax
#         self.lmax = lmax
#         self.num_components = lmax+1
#         self.num_channel = num_channel

#         self.ace = SO2TensorProduct(
#             mmax, 
#             lmax,
#             num_channel, 
#             m1m2=m1m2, 
#             internal_weights=internal_weights
#         )
#         self.weight_numel = self.ace.weight_numel

#     def forward(self, x, y, w) -> torch.Tensor:
#         return self.ace(x, y, w) # 
   

#     def extra_repr(self) -> str:
#         p = {
#             0: 'e',
#             1: 'o',
#         }
#         irreps = []
#         for m in range(self.mmax + 1):
#             irreps.append(f"{self.num_channel*(self.lmax+1)}x{m}{p[m % 2]}")
#         num_weights = sum(
#             p.numel() for p in self.parameters() if p.requires_grad
#         )
#         return (
#             f"{self.__class__.__name__}"
#             f"({'+'.join(irreps)} x {'+'.join(irreps)} -> "
#             f"{'+'.join(irreps)} | "
#             f"{num_weights} weights)"
#         )


# class SO2ScatterTensorProduct(torch.nn.Module):
#     def __init__(
#         self,
#         mmax: int,
#         lmax: int,
#         num_channel: int,
#         num_hidden_channel: int,
#         is_scalar_tp: bool,
#         is_so2_layout: bool,
#         use_so2_edge_ace: bool,
#         edge_nonlinear: Union[str, None],
#         num_elements: int,
#         so2_angular_basis: SO3Rotation,
#         reshape_in: LayoutTransform,
#         reshape_out: LayoutTransform,
#         scatter: Union[str, None],
#     ) -> None:
#         super().__init__()

#         self.mmax = mmax
#         self.lmax = lmax
#         self.num_channel = num_channel
#         self.num_hidden_channel = num_hidden_channel or self.num_channel
#         self.is_so2_layout = is_so2_layout
#         self.is_scalar_tp = is_scalar_tp
#         self.edge_nonlinear = edge_nonlinear
#         self.use_so2_edge_ace = use_so2_edge_ace
#         self.scatter = scatter

#         self.so2_angular_basis = so2_angular_basis
#         self.reshape_in = reshape_in
#         self.reshape_out = reshape_out
        

#         Cin = num_channel if not self.use_so2_edge_ace else num_channel * 2
#         Cout = num_channel
#         self.num_out_channel = Cout


#         self.num_gates = 0
#         for m in range(mmax + 1):
#             if self.use_so2_edge_ace:
#                 self.num_gates += lmax + 1
#             else:
#                 self.num_gates += lmax + 1 -m

#         assert edge_nonlinear is not None, "We force to use SO2 edge nonlinear in TACE"

#         self.linear_gate = SO2Linear(
#             0,
#             lmax,
#             Cin,
#             Cin // 2,
#             num_components_in=None,
#             num_components_out=[self.num_gates], 
#         )
#         self.linear_up = SO2Linear(
#             mmax,
#             lmax,
#             Cin,
#             Cin,
#             num_components_in=None,
#             num_components_out=[lmax+1] * (lmax+1),
#         )
#         self.ace = SO2EdgeProductBasis(
#             mmax, 
#             lmax, 
#             Cin // 2,
#             m1m2='<=',
#             internal_weights=False,
#         )
#         self.weight_numel = self.ace.weight_numel
#         self.nonlinearity = SO2Gate(
#             mmax,
#             lmax,
#             Cin // 2, 
#             channel_wise=True
#         )
#         self.linear_down = SO2Linear(
#             mmax,
#             lmax,
#             Cin // 2,
#             Cout,     
#             num_components_in=[lmax+1] * (lmax+1),
#             num_components_out=None,
#         )

#     def forward(
#             self, 
#             x: torch.Tensor, # [B, so_m, C]
#             y: torch.Tensor,  # node_attrs here
#             w: torch.Tensor, 
#             edge_index: torch.Tensor,
#             cutoff: torch.Tensor,
#         ) -> torch.Tensor:

#         num_nodes = x.size(0)
#         num_edges = w.size(0)
#         x = self.reshape_in(x) 

#         if self.use_so2_edge_ace:
#             x = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=-1)
#         else:
#             x = x[edge_index[0]]

#         if self.is_scalar_tp:
#             w = w.view(num_edges, self.num_components, -1)
#             m_ij = torch.einsum(
#                 'bij, bjc -> bic', 
#                     self.so2_angular_basis.wigner_inv.narrow(2, 0, (self.lmax + 1)),
#                     x * w
#             ) # first so3 tp, no nonlinearity is required here
#         else:
#             m_ij = self.so2_angular_basis.rotate(x)
#             gate = self.linear_gate(m_ij)
#             m_ij = self.linear_up(m_ij)
#             m_ij_1, m_ij_2 = torch.split(m_ij, self.num_channel, dim=-1)
        
#             m_ij = self.ace(m_ij_1, m_ij_2, w) + (m_ij_1 + m_ij_2) * 0.5

#             m_ij = self.nonlinearity(m_ij, gate) 

#             m_ij = self.linear_down(m_ij)
#             m_ij = self.so2_angular_basis.rotate_inv(m_ij)

#         if cutoff is not None:
#             m_ij = m_ij * cutoff.unsqueeze(-1)


#         if self.scatter is None:
#             return m_ij
        
#         m_i = scatter_sum(
#                 m_ij, 
#                 edge_index[1], 
#                 dim=0, 
#                 dim_size=num_nodes,
#         )
#         return self.reshape_out.inverse(m_i)



# class ChannelWiseFullyConnectedSO2TensorProduct(torch.nn.Module):

#     def __init__(
#         self,
#         mmax: int,
#         lmax: int,
#         num_channels: int,
#         m1m2: Union[str, None] = None,
#     ):
#         super().__init__()

#         self.mmax = mmax
#         self.lmax = lmax
#         self.num_channels = num_channels
#         self.m1m2 = m1m2
#         self.cmul = self.cmul2
#         self.instructions = []

#         weight_numel = 0
#         for m3 in range(mmax + 1):
#             paths = self.enumerate_paths(m3)
#             self.instructions.append(paths)
#             n3 = lmax + 1 - m3 
#             for m1, m2, mode in paths:
#                 n1 = lmax + 1 - m1
#                 n2 = lmax + 1 - m2
#                 weight_numel += (num_channels * n3 * n1 * n2)

#         self.weight_numel = weight_numel
#         self.weight = torch.nn.Parameter(torch.randn(1, self.weight_numel))

#         output_scales = []
#         # m = 0
#         n0 = lmax + 1
#         scale0 = torch.full(
#             (n0,),
#             1.0 / math.sqrt(
#                 sum(
#                     (lmax + 1 - m1) * (lmax + 1 - m2)
#                     for m1, m2, _ in self.instructions[0]
#                 )
#             ),
#         )
#         output_scales.append(scale0)
#         # m > 0
#         for m3 in range(1, mmax + 1):
#             n3 = lmax + 1 - m3
#             num_paths = 0
#             for m1, m2, mode in self.instructions[m3]:
#                 n1 = lmax + 1 - m1
#                 n2 = lmax + 1 - m2
#                 num_paths += n1 * n2
#             scale = 1.0 / math.sqrt(num_paths)
#             output_scales.append(torch.full((2 * n3,), scale))
#         output_scales = torch.cat(output_scales)
#         self.register_buffer( "output_scales", output_scales, persistent=False)

#         # for m3, paths in enumerate(self.instructions):
#         #     for m1, m2, mode in paths:
#         #         print(m1, m2, m3, mode)
#         # print()

#     def enumerate_paths(self, m3: int) -> list[tuple[int, int, str]]:
#         paths = []

#         for m1 in range(self.mmax + 1):
#             for m2 in range(self.mmax + 1):
#                 if satisfy(m1, m2, self.m1m2):
#                     # x1 * x2
#                     if m1 + m2 == m3:
#                         paths.append((m1, m2, "sum"))
#                     # x1 * conj(x2)
#                     elif abs(m1 - m2) == m3:
#                         paths.append((m1, m2, "diff"))

#         return paths

#     def rmul(self, x, y): 
#         # [B, n1, C] * [B, n2, C] =>  [B, n1*n2, C]
#         z = x.unsqueeze(2) * y.unsqueeze(1)
#         B, n1, n2, C = z.shape
#         z = z.reshape(B, n1 * n2, C)
#         return z
    
#     def cmul1(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
#         '''Layout damei, should be 2 in last dim'''
#         # [B, 2, n1, C] * [B, 2, n2, C] => [B, 2, n1*n2, C]
#         x = x.permute(0,2,3,1).contiguous()
#         y = y.permute(0,2,3,1).contiguous()
#         x = torch.view_as_complex(x)
#         y = torch.view_as_complex(y)
#         if mode == "diff":
#             y = y.conj()

#         z = x.unsqueeze(2) * y.unsqueeze(1)
        
#         B = z.size(0)
#         C = self.num_channels

#         z = z.reshape(B, -1, C)
#         z = torch.view_as_real(z)

#         z = z.permute(0,3,1,2)

#         return z
    
#     def cmul2(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
#         # [B, 2, n1, C] * [B, 2, n2, C] => [B, 2, n1*n2, C]
#         a = x[:, 0]
#         b = x[:, 1]
#         c = y[:, 0]
#         d = y[:, 1]
#         a = a.unsqueeze(2)
#         b = b.unsqueeze(2)
#         c = c.unsqueeze(1)
#         d = d.unsqueeze(1)

#         if mode == "sum":
#             real = a * c - b * d
#             imag = a * d + b * c
#         else:
#             real = a * c + b * d
#             imag = b * c - a * d

#         B = real.size(0)
#         C = real.size(-1)

#         real = real.reshape(B, -1, C)
#         imag = imag.reshape(B, -1, C)

#         out = torch.stack([real, imag], dim=1)

#         return out
    
#     def to_list(self, x: torch.Tensor) -> torch.Tensor:
#         B = x.size(0)
#         out = []
#         offset = 0
#         # m = 0
#         n0 = self.lmax + 1
#         out.append(x[:, offset:offset+n0])
#         offset += n0
#         # m > 0
#         for m in range(1, self.mmax + 1):
#             n = self.lmax + 1 - m
#             xm = x[:, offset:offset+2*n]
#             xm = xm.view(B, 2, n, self.num_channels)
#             out.append(xm)
#             offset += 2 * n
#         return out

#     def real_channel_wise_fc(self, z: torch.Tensor, w: torch.Tensor):
#         out = torch.einsum("bpc,bcop->boc", z, w)
#         return out

#     def complex_channel_wise_fc(self, z: torch.Tensor, w: torch.Tensor):
#         out = torch.einsum("btpc, bcop->btoc", z, w)
#         return out
    
#     def forward(
#             self, x: torch.Tensor, 
#             y: torch.Tensor, 
#             ws: torch.Tensor | None = None,
#         ) -> torch.Tensor:

#         xs = self.to_list(x)
#         ys = self.to_list(y)

#         outputs = []
#         w_offset = 0
#         C = self.num_channels

#         # m = 0
#         n0 = self.lmax + 1
#         m0 = 0.0
#         for m1, m2, mode in self.instructions[0]:
#             n1 = self.lmax + 1 - m1
#             n2 = self.lmax + 1 - m2
#             w_numel = (C * n0 * n1 * n2)
#             w = self.weight[:, w_offset:w_offset+w_numel]
#             w_offset += w_numel
#             w = w.view(-1, self.num_channels, n0, n1 * n2,)

#             # 0 x 0
#             if m1 == 0 and m2 == 0:
#                 z = self.rmul(xs[0], ys[0])
#                 out = self.real_channel_wise_fc(z, w)
                
#                 m0 = m0 + out

#             # m > 0 and m1 -m2 = 0
#             elif m1 > 0 and m2 > 0:
#                 z = self.cmul(xs[m1], ys[m2], "diff")
#                 out = self.real_channel_wise_fc(z[:, 0], w) # The imaginary part is 0
#                 m0 = m0 + out

#         outputs.append(m0)

#         # m > 0
#         for m3 in range(1, self.mmax + 1):
#             n3 = self.lmax + 1 - m3
#             real = 0.0
#             imag = 0.0
#             for m1, m2, mode in self.instructions[m3]:
#                 n1 = self.lmax + 1 - m1
#                 n2 = self.lmax + 1 - m2
#                 w_numel = (C * n3 * n1 * n2)
#                 w = self.weight[:, w_offset:w_offset+w_numel]
#                 w_offset += w_numel
#                 w = w.view(-1, C, n3, n1 * n2)
#                 if m1 == 0 or m2 == 0:
#                     continue
                
#                 if m1 < m2 and mode == 'diff':
#                     z = self.cmul(ys[m2], xs[m1], mode)
#                 else:
#                     z = self.cmul(xs[m1], ys[m2], mode)
#                 out = self.complex_channel_wise_fc(z, w)
#                 real = real + out[:, 0]
#                 imag = imag + out[:, 1]

#                 # z = self.cmul(xs[m1], ys[m2], mode)
#                 # out = self.complex_channel_wise_fc(z, w)
#                 # real = real + out[:, 0]
#                 # if m1 < m2 and mode == 'diff':
#                 #     imag = imag - out[:, 1]
#                 # else:
#                 #     imag = imag + out[:, 1]

#                 # real = real + self.real_channel_wise_fc(z[:, 0], w)
#                 # imag = imag + self.real_channel_wise_fc(z[:, 1], w)

#             outputs.append(real)
#             outputs.append(imag)

#         out = torch.cat(outputs, dim=1)
#         out = out * self.output_scales.view(1, -1, 1)
#         return out
        
#     def __repr__(self):
#         lines = []
#         lines.append(
#             f"{self.__class__.__name__}("
#         )
#         lines.append(
#             f"  mmax={self.mmax}, "
#             f"lmax={self.lmax}, "
#             f"channels={self.num_channels}, "
#             f"weights={self.weight_numel}"
#         )
#         lines.append("")
#         lines.append("  instructions:")
#         total_paths = 0
#         for m3, paths in enumerate(self.instructions):
#             total_paths += len(paths)
#             path_strs = []
#             for m1, m2, mode in paths:
#                 if mode == "sum":
#                     expr = f"{m1}+{m2}"
#                 else:
#                     expr = f"{m1}-{m2}"
#                 path_strs.append(expr)
#             joined = ", ".join(path_strs)
#             lines.append(
#                 f"    m={m3:<2} : "
#                 f"{len(paths):<2} paths | "
#                 f"{joined}"
#             )
#         lines.append("")
#         lines.append(f"  total_paths={total_paths}")
#         lines.append(")")
#         return "\n".join(lines)