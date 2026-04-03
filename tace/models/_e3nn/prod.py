###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Dict


import torch
import opt_einsum_fx
from e3nn import o3


from ..layout import LayoutTransform
from .base import Product
from .linear import Linear, ElementLinear
from .fused import uuuTensorProduct
from .matrix import MatrixTensorProduct


class SpectralLinearACE(Product):
    
    def _setup(self):

        self.aces = torch.nn.ModuleList()
        self.coefs = torch.nn.ModuleList()
        self.coefs.append(
            ElementLinear(
                self.irreps_in,
                self.irreps_hidden,
                bias=self.use_bias,
                num_elements=self.num_elements,
            )    
        )

        product_in1 = self.irreps_in
        if self.correlation == 2:
            product_out = self.irreps_out
        else:
            product_out = self.irreps_in 

        self.shapes = []

        for nu in range(2, self.correlation+1):

            this_ace = uuuTensorProduct(
                irreps_in1=product_in1,
                irreps_in2=self.irreps_in,
                irreps_out=product_out,
                l1l2=self.l1l2,
                ictp_ictc_like=self.ictp_ictc_like,
            )
            self.aces.append(this_ace)
            self.coefs.append(
                ElementLinear(
                    this_ace.irreps_out.simplify(),
                    self.irreps_hidden,
                    bias=self.use_bias,
                    num_elements=self.num_elements,
                )    
            )

            product_in1 = this_ace.irreps_out

            if nu == self.correlation-1:
                product_out = self.irreps_out
            else:
                product_out = self.irreps_in


        self.linear = Linear(
            self.irreps_hidden,
            self.irreps_out,
            bias=self.use_bias
        )    

    def forward(
            self, 
            node_feats: torch.Tensor, 
            node_attrs: torch.Tensor,
            sc: torch.Tensor,
        ) -> torch.Tensor:

        corr_feats = {
            1: node_feats,
        }
        outs = self.coefs[0](corr_feats[1], node_attrs)

        for nu in range(2, self.correlation+1):
            corr_feats[nu] = self.aces[nu-2](corr_feats[nu-1], node_feats)
            outs = outs + self.coefs[nu-1](corr_feats[nu], node_attrs)
        outs = self.linear(outs)

        if sc is not None:
            outs = outs + sc

        for k, v in corr_feats.items():
            print(k)
            print(v[0, :self.num_channel])

        return outs
    

class SpatialLinearACE(Product):

    def _setup(self):

        self.reshape1 = LayoutTransform(self.irreps_in)
        self.reshape2 = LayoutTransform(self.irreps_out)

        ls_out = [ir.l for _, ir in self.irreps_out]
        out_slices = []
        sum_l = 0
        start = 0
        for l in ls_out:
            sum_l += 2*l +1
            length = 2*l + 1
            out_slices.append(slice(start, start+length))
            start += length

        expand_index = torch.zeros(sum_l).long()
        for idx, s, in enumerate(out_slices):
            expand_index[s] = idx
        self.register_buffer("expand_index", expand_index, persistent=False)

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

        from_grid = torch.einsum(
            "am, mbi -> bai", from_s2.sha, from_s2.shb
        ).detach()
        from_grid_list = torch.split(from_grid, [2*l + 1 for l in range(self.truncation+1)], dim=-1)


        self.register_buffer(
            "to_grid", 
            torch.einsum(
                "mbi, am -> bai", to_s2.shb, to_s2.sha
            ).detach(),
            persistent=False,
        )
        self.register_buffer(
            "from_grid", 
            torch.cat([from_grid_list[l] for l in ls_out], dim=-1).contiguous(), # l = idx
            persistent=False,
        )

        self.weight = torch.nn.Parameter(
            torch.empty(
                len(ls_out),
                self.num_elements, 
                self.num_channel,
                self.correlation*self.num_channel, 
            )
        )
        trace = torch.fx.symbolic_trace(
            lambda z, w, g, x: torch.einsum('Bz, mzCc, bam, Bbac -> BmC', z, w, g, x)
        )
        # graph = (
        #     opt_einsum_fx.optimize_einsums_full(
        #         model=trace,
        #         example_inputs=(
        #             torch.randn([256, 89]),
        #             torch.randn([16, 89, 64, 128]),
        #             torch.randn([8, 9, 16]),
        #             torch.randn([256, 8, 9, 128]),
        #         ),
        #     )
        # )
        graph = (
            opt_einsum_fx.optimize_einsums_full(
                model=trace,
                example_inputs=(
                    torch.randn([32, 10]),
                    torch.randn([16, 10, 8, 16]),
                    torch.randn([8, 9, 16]),
                    torch.randn([32, 8, 9, 16]),
                ),
            )
        )
        self.coefs = graph
        self.linear = Linear(
            self.irreps_out,
            self.irreps_out,
            bias=self.use_bias
        )

        self.alpha = 1.0 / math.sqrt(self.correlation*self.num_channel)
        if self.use_bias and 0 in ls_out:
            self.bias = torch.nn.Parameter(
                torch.empty(self.num_elements, self.num_channel)
            )
        else:
            self.register_parameter("bias", None)

        self.num_padding = (self.truncation+1)**2 - (self.lmax+1)**2
        self.reset_parameters()

        def moment(f, n, dtype=None, device=None):
            r"""
            compute n th moment
            <f(z)^n> for z normal
            """
            gen = torch.Generator(device=device).manual_seed(0)
            z = torch.randn(1_000_000, generator=gen, dtype=torch.float64, device=device)
            return f(z).pow(n).mean()


        # class xn(torch.nn.Module):
        #     def __init__(self, n: int) -> None:
        #         super().__init__()
        #         self.n = n

        #     def forward(self, x: torch.Tensor):
        #         return x ** self.n
            
        cst = []
        for nu in range(1, self.correlation+1):
            cst.append(1.0)
            # cst.append(moment(xn(nu), 2).pow(-0.5).item())

        if self.trainable_scale:
            self.scale = torch.nn.Parameter(torch.tensor(cst))
        else:
            self.register_buffer("scale", torch.tensor(cst))

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
            
    def _to_grid(self, x: torch.Tensor) -> torch.Tensor:           
        return torch.einsum("bai, Bic -> Bbac", self.to_grid, x)

    def _from_grid(self, x: torch.Tensor) -> torch.Tensor:       
        return torch.einsum("bai, Bbac -> Bic", self.from_grid, x)

    def forward(
            self, 
            node_feats: torch.Tensor, 
            node_attrs: torch.Tensor,
            sc: torch.Tensor,
        ) -> torch.Tensor:

        # node_feats = self.reshape1(node_feats).transpose(-1, -2).contiguous()
        node_feats = self.reshape1(node_feats)
        pad_shape = list(node_feats.shape)
        pad_shape[1] = self.num_padding
        padding = torch.zeros(
            pad_shape,
            dtype=node_feats.dtype,
            device=node_feats.device
        )
        node_feats = torch.cat([node_feats, padding], dim=1)
        base_grid = self._to_grid(node_feats)
        corr_feats_list = [base_grid]
        grid_prev = base_grid

        for nu in range(2, self.correlation + 1):
            grid_prev = grid_prev * base_grid
            corr_feats_list.append(grid_prev)

        for nu in range(1, self.correlation + 1):
            corr_feats_list[nu-1] = corr_feats_list[nu-1] * self.scale[nu-1]

        corr_feats = torch.cat(corr_feats_list, dim=-1)

        weight = self.weight * self.alpha
        weight = torch.index_select(
            weight, dim=0, index=self.expand_index
        ) 
        outs = self.coefs(node_attrs, weight, self.from_grid, corr_feats)

        if self.bias is not None:
            bias = torch.einsum('Bz, zi -> Bi', node_attrs, self.bias)
            outs[:, 0:1, :] = outs.narrow(1, 0, 1) + bias.unsqueeze(1)
        
        outs = self.linear(self.reshape2.inverse(outs))

        if sc is not None:
            outs = outs + sc

        return outs   
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nlon={self.num_longitude}, nlat={self.num_latitude}, truncation={self.truncation})"
    

class AgnosticLinearACE(Product):
    
    
    def _setup(self):

        self.aces = torch.nn.ModuleList()
        self.coefs = torch.nn.ModuleList()
        self.coefs.append(
            Linear(
                self.irreps_in,
                self.irreps_hidden,
                bias=self.use_bias,
            )    
        )

        product_in1 = self.irreps_in
        if self.correlation == 2:
            product_out = self.irreps_out
        else:
            product_out = self.irreps_in 

        self.shapes = []

        for nu in range(2, self.correlation+1):

            this_ace = uuuTensorProduct(
                irreps_in1=product_in1,
                irreps_in2=self.irreps_in,
                irreps_out=product_out,
                l1l2=self.l1l2,
                ictp_ictc_like=self.ictp_ictc_like,
            )
            self.aces.append(this_ace)
            self.coefs.append(
                Linear(
                    this_ace.irreps_out.simplify(),
                    self.irreps_hidden,
                    bias=self.use_bias,
                )    
            )

            product_in1 = this_ace.irreps_out

            if nu == self.correlation-1:
                product_out = self.irreps_out
            else:
                product_out = self.irreps_in


        self.linear = Linear(
            self.irreps_hidden,
            self.irreps_out,
            bias=self.use_bias
        )    

    def forward(
            self, 
            node_feats: torch.Tensor, 
            node_attrs: torch.Tensor,
            sc: torch.Tensor,
        ) -> torch.Tensor:

        corr_feats = {
            1: node_feats,
        }
        outs = self.coefs[0](corr_feats[1])

        for nu in range(2, self.correlation+1):
            corr_feats[nu] = self.aces[nu-2](corr_feats[nu-1], node_feats)
            outs = outs + self.coefs[nu-1](corr_feats[nu])
        outs = self.linear(outs)

        if sc is not None:
            outs = outs + sc

        return outs


class IdentityLinearACE(Product):

    def _setup(self):

        # assert self.correlation == 1

        self.coefs = ElementLinear(
            self.irreps_in,
            self.irreps_out,
            bias=self.use_bias,
            num_elements=self.num_elements,
        )

        self.linear = Linear(
            self.irreps_out,
            self.irreps_out,
            bias=self.use_bias,
        )

    def forward(
            self, 
            node_feats: torch.Tensor, 
            node_attrs: torch.Tensor,
            sc: torch.Tensor,
        ) -> torch.Tensor:
        outs = self.linear(self.coefs(node_feats, node_attrs))
         
        if sc is not None:
            outs = outs + sc

        return outs
    

class MatrixLinearACE(Product):
    
    def _setup(self):

        self.reshape = LayoutTransform(self.irreps_in)

        self.aces = torch.nn.ModuleList(
            MatrixTensorProduct(
                L1=self.irreps_in.lmax,
                L2=self.irreps_in.lmax,
                C=self.num_channel,
            ) for _ in range(self.correlation)
        )
        self.coefs = torch.nn.ModuleList(
            ElementLinear(
                self.irreps_in,
                self.irreps_hidden,
                bias=self.use_bias,
                num_elements=self.num_elements,
            ) for _ in range(self.correlation)
        )

        self.linear = Linear(
            self.irreps_hidden,
            self.irreps_out,
            bias=self.use_bias
        )    

    def forward(
            self, 
            node_feats: torch.Tensor, 
            node_attrs: torch.Tensor,
            sc: torch.Tensor,
        ) -> torch.Tensor:

        node_feats = self.reshape(node_feats)

        corr_feats = {
            1: node_feats,
        }
        outs = self.coefs[0](self.reshape.inverse(corr_feats[1]), node_attrs)

        for nu in range(2, self.correlation+1):
            corr_feats[nu] = self.aces[nu-2](corr_feats[nu-1], node_feats)
            outs = outs + self.coefs[nu-1](self.reshape.inverse(corr_feats[nu]), node_attrs)
        outs = self.linear(outs)

        if sc is not None:
            outs = outs + sc

        return outs
    

# class GatedLinearUnitACE(Product):
    
#     def _setup(self):

#         self.aces = torch.nn.ModuleList()
#         self.coefs = torch.nn.ModuleList()
#         self.coefs.append(
#             NONLINEAR_ACE[self.nonlinear_type](
#                 self.irreps_in,
#                 self.irreps_hidden,
#                 activation=ACTIVATION[self.nonlinear_act](),
#                 bias=self.use_bias,
#                 num_elements=self.num_elements,
#             )    
#         )

#         product_in1 = self.irreps_in
#         if self.correlation == 2:
#             product_out = self.irreps_out
#         else:
#             product_out = self.irreps_in 

#         self.shapes = []

#         for nu in range(2, self.correlation+1):
#             instructions, irreps_out = generate_e3nn_paths(
#                 irreps_out=product_out,
#                 irreps_in1=product_in1,
#                 irreps_in2=self.irreps_in,
#                 l1l2=self.l1l2,
#                 ictp_ictc_like=self.ictp_ictc_like,
#                 e3nn_mode='uuu'
#             )

#             this_ace = o3.TensorProduct(
#                 irreps_in1=product_in1,
#                 irreps_in2=self.irreps_in,
#                 irreps_out=irreps_out,
#                 instructions=instructions,
#                 internal_weights=False,
#                 shared_weights=False,
#             )
#             self.aces.append(this_ace)
#             self.coefs.append(
#                 NONLINEAR_ACE[self.nonlinear_type](
#                     this_ace.irreps_out.simplify(),
#                     self.irreps_hidden,
#                     activation=ACTIVATION[self.nonlinear_act](),
#                     bias=self.use_bias,
#                     num_elements=self.num_elements,
#                 )    
#             )

#             product_in1 = this_ace.irreps_out

#             if nu == self.correlation-1:
#                 product_out = self.irreps_out
#             else:
#                 product_out = self.irreps_in

#         self.linear = Linear(
#             self.irreps_hidden,
#             self.irreps_out,
#             bias=self.use_bias
#         )    

#     def forward(
#             self, 
#             node_feats: torch.Tensor, 
#             node_attrs: torch.Tensor,
#             node_env: torch.Tensor,
#             sc: torch.Tensor,
#         ) -> torch.Tensor:

#         corr_feats = {
#             1: node_feats,
#         }
#         outs = self.coefs[0](corr_feats[1], node_attrs)

#         for nu in range(2, self.correlation+1):
#             corr_feats[nu] = self.aces[nu-2](corr_feats[nu-1], node_feats)
#             outs = outs + self.coefs[nu-1](corr_feats[nu], node_attrs)
#         outs = self.linear(outs)

#         if sc is not None:
#             outs = outs + sc

#         return outs


# class MixtralExpertsGatedLinearUnitACE(Product):
    
#     def _setup(self):

#         self.router = torch.nn.ModuleList(
#             Linear(
#                 f"{self.num_channel}x0e",
#                 f"{self.num_experts}x0e",
#                 bias=False,
#             ) for _ in range(self.correlation)
#         )
        
#         self.aces = torch.nn.ModuleList()
#         self.coefs = torch.nn.ModuleList()

#         self.coefs.append(
#             MixtralExpertsGateGatedLinearUnit(
#                 self.irreps_in,
#                 self.irreps_hidden,
#                 activation=ACTIVATION[self.nonlinear_act](),
#                 bias=self.use_bias,
#                 num_elements=self.num_elements,
#                 num_experts=self.num_experts,
#                 num_shared_experts=self.num_shared_experts,
#             )    
#         )

#         product_in1 = self.irreps_in
#         if self.correlation == 2:
#             product_out = self.irreps_out
#         else:
#             product_out = self.irreps_in 

#         self.shapes = []

#         for nu in range(2, self.correlation+1):
#             instructions, irreps_out = generate_e3nn_paths(
#                 irreps_out=product_out,
#                 irreps_in1=product_in1,
#                 irreps_in2=self.irreps_in,
#                 l1l2=self.l1l2,
#                 ictp_ictc_like=self.ictp_ictc_like,
#                 e3nn_mode='uuu'
#             )

#             this_ace = o3.TensorProduct(
#                 irreps_in1=product_in1,
#                 irreps_in2=self.irreps_in,
#                 irreps_out=irreps_out,
#                 instructions=instructions,
#                 internal_weights=False,
#                 shared_weights=False,
#             )
#             self.aces.append(this_ace)
#             self.coefs.append(
#                 MixtralExpertsGateGatedLinearUnit(
#                     this_ace.irreps_out.simplify(),
#                     self.irreps_hidden,
#                     bias=self.use_bias,
#                     activation=ACTIVATION[self.nonlinear_act](),
#                     num_elements=self.num_elements,
#                     num_experts=self.num_experts,
#                     num_shared_experts=self.num_shared_experts,
#                 )    
#             )

#             product_in1 = this_ace.irreps_out

#             if nu == self.correlation-1:
#                 product_out = self.irreps_out
#             else:
#                 product_out = self.irreps_in

#         self.linear = Linear(
#             self.irreps_hidden,
#             self.irreps_out,
#             bias=self.use_bias
#         )

#     def forward(
#             self, 
#             node_feats: torch.Tensor, 
#             node_attrs: torch.Tensor,
#             node_env: torch.Tensor,
#             sc: torch.Tensor,
#         ) -> torch.Tensor:
#         gate_logits = self.router[0](node_env)
#         gate_probs = torch.softmax(gate_logits, dim=-1)
#         topk_probs, topk_idx = torch.topk(gate_probs, k=self.top_k, dim=-1) 
#         gate_probs_sparse = torch.zeros_like(gate_probs)
#         gate_probs_sparse.scatter_(-1, topk_idx, topk_probs / (topk_probs.sum(-1, keepdim=True)))
#         corr_feats = {
#             1: node_feats,
#         }
#         outs = self.coefs[0](corr_feats[1], node_attrs, gate_probs_sparse)

#         for nu in range(2, self.correlation+1):
#             gate_logits = self.router[nu-1](node_env)
#             gate_probs = torch.softmax(gate_logits, dim=-1)
#             topk_probs, topk_idx = torch.topk(gate_probs, k=self.top_k, dim=-1) 
#             gate_probs_sparse = torch.zeros_like(gate_probs)
#             gate_probs_sparse.scatter_(-1, topk_idx, topk_probs / (topk_probs.sum(-1, keepdim=True)))
#             corr_feats[nu] = self.aces[nu-2](corr_feats[nu-1], node_feats)
#             outs = outs + self.coefs[nu-1](corr_feats[nu], node_attrs, gate_probs_sparse)

#         outs = self.linear(outs)

#         if sc is not None:
#             outs = outs + sc

#         return outs
    

PRODUCT: Dict[str, torch.nn.Module] = {
    "coupled": SpectralLinearACE,
    "spectral": SpectralLinearACE, # TODO, refacor
    "CGTP": SpectralLinearACE,
    "grid": SpatialLinearACE,
    "spatial": SpatialLinearACE,
    "GTP": SpatialLinearACE,
    "matrix": MatrixLinearACE,
    "MTP": MatrixLinearACE,

    # special
    "agnostic": AgnosticLinearACE,
    "identity": IdentityLinearACE,
    # "glu": GatedLinearUnitACE,
    # "moe": MixtralExpertsGatedLinearUnitACE,
}

