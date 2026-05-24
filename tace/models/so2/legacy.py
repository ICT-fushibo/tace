# class dSO2uuuTensorProduct(torch.nn.Module):
#     """
#     Fully complex SO(2) tensor product.

#     Weight layout:
#         [1, 2, -1]

#     where:
#         dim=1:
#             0 -> real weight
#             1 -> imag weight

#     Internal complex representation:
#         m = 0:
#             [B, n, C]

#         m > 0:
#             [B, 2, n, C]
#     """

#     def __init__(
#         self,
#         mmax: int,
#         lmax: int,
#         num_channels: int,
#         m1m2: Union[str, None] = None,
#         internal_weights: bool = True,
#     ):
#         super().__init__()

#         self.mmax = mmax
#         self.lmax = lmax
#         self.num_channels = num_channels
#         self.m1m2 = m1m2

#         self.instructions = []

#         n = lmax + 1
#         C = num_channels

#         weight_numel = 0

#         for m3 in range(mmax + 1):

#             paths = self.enumerate_paths(m3)

#             self.instructions.append(paths)

#             weight_numel += (
#                 n * C * len(paths)
#             )

#         self.weight_numel = weight_numel

#         if internal_weights:

#             self.weight = torch.nn.Parameter(
#                 torch.randn(
#                     1,
#                     2,
#                     self.weight_numel,
#                 )
#             )

#         else:

#             self.register_buffer("weight", None)

#         self.internal_weights = internal_weights


#         output_scales = []

#         scale0 = torch.full(
#             (n,),
#             1.0 / math.sqrt(len(self.instructions[0]))
#         )

#         output_scales.append(scale0)

#         for m3 in range(1, mmax + 1):

#             scale = (
#                 1.0 / math.sqrt(len(self.instructions[m3]))
#             )

#             output_scales.append(
#                 torch.full((2 * n,), scale)
#             )

#         output_scales = torch.cat(output_scales)

#         self.register_buffer(
#             "output_scales",
#             output_scales,
#             persistent=False
#         )


#     def enumerate_paths(
#         self,
#         m3: int
#     ):
#         paths = []
#         for m1 in range(self.mmax + 1):
#             for m2 in range(self.mmax + 1):
#                 if satisfy(m1, m2, self.m1m2):
#                     # x * y
#                     if m1 + m2 == m3:
#                         paths.append((m1, m2, "sum"))
#                     # x * conj(y)
#                     elif abs(m1 - m2) == m3:
#                         paths.append((m1, m2, "diff"))
#         return paths

#     def to_list(self, x: torch.Tensor):

#         B = x.size(0)
#         out = []
#         offset = 0
#         n = self.lmax + 1

#         # m = 0
#         x0 = x[:, offset:offset + n]
#         out.append(x0)
#         offset += n

#         # m > 0
#         for m in range(1, self.mmax + 1):
#             xm = x[:, offset:offset + 2 * n]
#             xm = xm.view(B, 2, n, self.num_channels)
#             xm = xm.permute(0, 2, 3, 1).contiguous()
#             xm = torch.view_as_complex(xm)
#             out.append(xm)
#             offset += 2 * n

#         return out

#     def get_complex_weight(self, ws, offset, n, C, cdtype):

#         wr = ws[:, 0, offset:offset + n * C]
#         wi = ws[:, 1, offset:offset + n * C]

#         offset += n * C
#         wr = wr.view(-1, n, C)
#         wi = wi.view(-1, n, C)
#         w = torch.complex(wr, wi)
#         w = w.to(cdtype)

#         return w, offset


#     def forward(
#         self,
#         x: torch.Tensor,
#         y: torch.Tensor,
#         weight: Union[torch.Tensor, None] = None,
#     ):

#         xs = self.to_list(x)
#         ys = self.to_list(y)

#         if self.internal_weights:
#             ws = self.weight
#         else:
#             ws = weight

#         B = x.size(0)
#         C = self.num_channels
#         n = self.lmax + 1

#         if x.dtype == torch.float64:
#             cdtype = torch.complex128
#         else:
#             cdtype = torch.complex64

#         outputs = []

#         w_offset = 0
#         m0 = torch.zeros(
#             B, n, C,
#             dtype=cdtype,
#             device=x.device
#         )


#         # m = 0
#         for m1, m2, mode in self.instructions[0]:

#             w, w_offset = self.get_complex_weight(
#                 ws,
#                 w_offset,
#                 n,
#                 C,
#                 cdtype,
#             )

#             # 0 x 0 => 0
#             if m1 == 0 and m2 == 0:
#                 z = xs[0].to(cdtype) * ys[0].to(cdtype)
#             # m1 x m2 => 0 (m1, m2 > 0)
#             else:
#                 if m1 >= m2:
#                     z = xs[m1] * ys[m2].conj()
#                 else:
#                     z = ys[m2] * xs[m1].conj()

#             m0 = m0 + z * w

#         # invariant -> real, TODO, real and image are all invariant
#         outputs.append(m0.real)

#         # m > 0
#         for m3 in range(1, self.mmax + 1):
#             acc = torch.zeros(
#                 B, n, C,
#                 dtype=cdtype,
#                 device=x.device
#             )

#             for m1, m2, mode in self.instructions[m3]:
#                 w, w_offset = self.get_complex_weight(
#                     ws,
#                     w_offset,
#                     n,
#                     C,
#                     cdtype,
#                 )

#                 # 0 x m2 
#                 if m1 == 0:
#                     z = xs[m1].to(cdtype) * ys[m2]
#                 # m1 x 0
#                 elif m2 == 0:
#                     z = xs[m1] * ys[m2].to(cdtype)
#                 else:
#                     # sum path
#                     if mode == "sum":
#                         z = xs[m1] * ys[m2]
#                     # diff path
#                     else:
#                         if m1 >= m2:
#                             z = xs[m1] * ys[m2].conj()
#                         else:
#                             z = ys[m2] * xs[m1].conj()

#                 acc = acc + z * w

#             outputs.append(acc.real)
#             outputs.append(acc.imag)


#         out = torch.cat(outputs, dim=1)
#         out = out * self.output_scales.view(1, -1, 1)

#         return out


#     def __repr__(self):
#         lines = []
#         lines.append(f"{self.__class__.__name__}(")
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
    

# # Plain
# class SO2uuuTensorProduct(torch.nn.Module):
#     """
#     Plain Tensor Product which all paths are directly summed.
#     """
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
#             weight_numel += num_channels * (lmax+1) * len(paths) 
            
#         # self.weight_numel = weight_numel
#         self.weight_numel = 0

#         output_scales = []
#         n = lmax + 1
#         # m = 0
#         scale0 = torch.full((n,), 1.0 / math.sqrt(len(self.instructions[0])))
#         output_scales.append(scale0)
#         # m > 0
#         for m3 in range(1, mmax + 1):
#             scale = 1.0 / math.sqrt(len(self.instructions[m3]))
#             output_scales.append(torch.full((2 * n,), scale))
#         output_scales = torch.cat(output_scales)
#         self.register_buffer("output_scales", output_scales, persistent=False)

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
#         # [B, n, C] * [B, n, C] =>  [B, n, C]
#         z = x * y
#         return z
    
#     def cmul1(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
#         '''Layout damei, should be 2 in last dim'''
#         # [B, 2, n, C] * [B, 2, n, C] => [B, 2, n, C]
#         x = x.permute(0,2,3,1).contiguous()
#         y = y.permute(0,2,3,1).contiguous()
#         x = torch.view_as_complex(x)
#         y = torch.view_as_complex(y)
#         if mode == "diff":
#             y = y.conj()
#         z = x * y
#         B = z.size(0)
#         C = self.num_channels
#         z = z.reshape(B, -1, C)
#         z = torch.view_as_real(z)
#         z = z.permute(0,3,1,2)

#         return z
    
#     def cmul2(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
#         # [B, 2, n, C] * [B, 2, n, C] => [B, 2, n, C]
#         a = x[:, 0]
#         b = x[:, 1]
#         c = y[:, 0]
#         d = y[:, 1]

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
#         n = self.lmax + 1
#         # m = 0
#         out.append(x[:, offset:offset+n])
#         offset += n
#         # m > 0
#         for m in range(1, self.mmax + 1):
#             xm = x[:, offset:offset+2*n]
#             xm = xm.view(B, 2, n, self.num_channels)
#             out.append(xm)
#             offset += 2 * n
#         return out

#     def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

#         xs = self.to_list(x) #  m = 0 [B, lmax+1, C]
#         ys = self.to_list(y) #  m > 0 [B, 2, lmax+1, C]

#         outputs = []
#         # m = 0
#         m0 = 0.0
#         for m1, m2, mode in self.instructions[0]:
#             # 0 x 0
#             if m1 == 0 and m2 == 0:
#                 z = self.rmul(xs[0], ys[0])
#                 m0 = m0 + z
#             # m > 0 and m1 -m2 = 0
#             elif m1 > 0 and m2 > 0:
#                 z = self.cmul(xs[m1], ys[m2], "diff")
#                 m0 = m0 + z[:, 0] # + z[:, 1] # not scale
#         outputs.append(m0)

#         # m > 0
#         for m3 in range(1, self.mmax + 1):
#             real = 0.0
#             imag = 0.0
#             for m1, m2, mode in self.instructions[m3]:
#                 if m1 == 0:
#                     z = xs[m1].unsqueeze(1) * ys[m2]
#                 elif m2 == 0:
#                     z = xs[m1] * ys[m2].unsqueeze(1)
#                 else:
#                     if m1 < m2 and mode == 'diff':
#                         z = self.cmul(ys[m2], xs[m1], mode)
#                     else:
#                         z = self.cmul(xs[m1], ys[m2], mode)
#                 real = real + z[:, 0]
#                 imag = imag + z[:, 1]
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