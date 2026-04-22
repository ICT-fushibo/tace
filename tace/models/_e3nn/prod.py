###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Dict


import torch
import opt_einsum_fx
import e3nn
from e3nn import o3
from e3nn.nn import Activation

from ..layout import LayoutTransform, LayoutTransform2
from .base import Product
from .linear import Linear, ElementLinear
from .fused import uuuTensorProduct, uuuTrainableTensorProduct
from .matrix import MatrixTensorProduct
from .nonlinear import GatedLinearUnit, NormLinearUnit, GridMLPUnit
from ..mlp import ACTIVATION


class CgtpACE(Product):
    """
    The most expressive ACE implementation based on Clebsch-Gordan tensor products.

    This class computes all possible many-body tensor product paths and couples
    all channels, forming a highly expressive product basis.

    Note:
        It is recommended to use no more than 64 channels, as increasing the
        number of channels beyond this does not necessarily lead to better
        performance and may introduce unnecessary computational overhead.
    """

    def _setup(self):

        self.linear_up = Linear(
            self.irreps_in,
            self.irreps_hidden1,
            bias=self.use_bias,
        ) if self.num_channel != self.num_hidden_channel else torch.nn.Identity()

        for_coefs = {
            "irreps_out": self.irreps_nonlinear,
            "bias": self.use_bias,
        }
        if self.agnostic:
            coefs_cls = Linear
        else:
            for_coefs['num_elements'] = self.num_elements
            coefs_cls = ElementLinear
            
        self.aces = torch.nn.ModuleList()
        self.coefs = torch.nn.ModuleList()
        self.coefs.append(coefs_cls(self.irreps_hidden1, **for_coefs))

        product_in1 = self.irreps_hidden1

        if self.correlation == 2:
            product_out = self.irreps_out
        else:
            product_out = self.irreps_hidden1

        for nu in range(2, self.correlation+1):
            this_ace = uuuTensorProduct(
                irreps_in1=product_in1,
                irreps_in2=self.irreps_hidden1,
                irreps_out=product_out,
                l1l2=self.l1l2,
                l3s=self.l3s,
                ictp_ictc_like=self.ictp_ictc_like,
            )
            self.aces.append(this_ace)
            self.coefs.append(coefs_cls(this_ace.irreps_out.simplify(), **for_coefs))
            product_in1 = this_ace.irreps_out

            if nu == self.correlation-1:
                product_out = self.irreps_hidden2
            else:
                product_out = self.irreps_hidden1

        if self.nonlinear_type is not None:
            if self.nonlinear_type == 'norm':
                self.nonlinearity = NormLinearUnit(
                    self.irreps_hidden2,
                    activation=ACTIVATION[self.nonlinear_act](),
                )
            elif self.nonlinear_type == 'grid':
                self.nonlinearity = GridMLPUnit(
                    self.irreps_hidden2,
                    activation=ACTIVATION[self.nonlinear_act](),
                    bias=False,
                ) # will introduct higher freq
            elif self.nonlinear_type == 'e3nngate':
                irreps_scalars = o3.Irreps(
                    [(mul, ir) for mul, ir in self.irreps_hidden2 if ir.l == 0]
                )
                irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden2 if ir.l > 0])
                irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in irreps_gated)
                activation_fn = torch.nn.functional.silu
                act_gates_fn = torch.nn.functional.sigmoid
                self.nonlinearity = e3nn.nn.Gate(
                    irreps_scalars=irreps_scalars,
                    act_scalars=[activation_fn for _ in irreps_scalars],
                    irreps_gates=irreps_gates,
                    act_gates=[act_gates_fn] * len(irreps_gates),
                    irreps_gated=irreps_gated,
                )
            elif self.nonlinear_type == 'gate':
                irreps_gated = self.irreps_hidden2
                irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in self.irreps_hidden2)
                self.nonlinearity = GatedLinearUnit(
                    irreps_gates=irreps_gates,
                    act_gates=[ACTIVATION[self.nonlinear_act]()] * len(irreps_gates),
                    irreps_gated=irreps_gated,
                )
            else:
                assert False, "Unkown Nonlinear"
        else:
            self.nonlinearity = torch.nn.Identity()


        self.linear = Linear(
            self.irreps_hidden2,
            self.irreps_out,
            bias=self.use_bias
        )    
        
    def forward(
            self, 
            node_feats: torch.Tensor, 
            node_attrs: torch.Tensor,
            sc: torch.Tensor,
        ) -> torch.Tensor:

        node_feats = self.linear_up(node_feats)

        corr_feats = {
            1: node_feats,
        }
        if self.agnostic:
            outs = self.coefs[0](corr_feats[1])
        else:
            outs = self.coefs[0](corr_feats[1], node_attrs)

        if self.agnostic:
            for nu in range(2, self.correlation+1):
                corr_feats[nu] = self.aces[nu-2](corr_feats[nu-1], node_feats)
                outs = outs + self.coefs[nu-1](corr_feats[nu])
        else:
            for nu in range(2, self.correlation+1):
                corr_feats[nu] = self.aces[nu-2](corr_feats[nu-1], node_feats)
                outs = outs + self.coefs[nu-1](corr_feats[nu], node_attrs)

        outs = self.linear(self.nonlinearity(outs))

        if sc is not None:
            outs = outs + sc

        return outs

# TODO, refactor
class MtpACE(Product):
    """
    An ACE implementation based on matrix tensor products.

    This module performs many-body expansion using matrix tensor products.
    This approach is inspired by the FusedTensor in e3x, and the idea was
    motivated by a reviewer's suggestion in one of my papers.

    Similar to GTP-based methods, it averages over paths.
    However, it includes antisymmetric interactions, and its computational
    scaling at higher correlation orders is significantly more efficient
    than CGTP_ACE.
    """

    def _setup(self):

        for_coefs = {
            "irreps_in": self.irreps_in,
            "irreps_out": self.irreps_nonlinear,
            "bias": self.use_bias,
        }
        if self.agnostic:
            coefs_cls = Linear
        else:
            for_coefs['num_elements'] = self.num_elements
            coefs_cls = ElementLinear

        self.reshape = LayoutTransform(self.irreps_in)

        self.aces = torch.nn.ModuleList(
            MatrixTensorProduct(
                L1=self.irreps_in.lmax,
                L2=self.irreps_in.lmax,
                C=self.num_hidden_channel,
            ) for _ in range(self.correlation-1)
        )
        self.coefs = torch.nn.ModuleList(
            coefs_cls(**for_coefs) for _ in range(self.correlation)
        )

        if self.nonlinear_type is not None:
            if self.nonlinear_type == 'norm':
                self.nonlinearity = NormLinearUnit(
                    self.irreps_hidden,
                    activation=ACTIVATION[self.nonlinear_act](),
                )
            elif self.nonlinear_type == 'grid':
                self.nonlinearity = GridMLPUnit(
                    self.irreps_hidden,
                    activation=ACTIVATION[self.nonlinear_act](),
                    bias=False,
                ) # will introduct higher freq
            elif self.nonlinear_type == 'e3nngate':
                irreps_scalars = o3.Irreps(
                    [(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0]
                )
                irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0])
                irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in irreps_gated)
                activation_fn = torch.nn.functional.silu
                act_gates_fn = torch.nn.functional.sigmoid
                self.nonlinearity = e3nn.nn.Gate(
                    irreps_scalars=irreps_scalars,
                    act_scalars=[activation_fn for _ in irreps_scalars],
                    irreps_gates=irreps_gates,
                    act_gates=[act_gates_fn] * len(irreps_gates),
                    irreps_gated=irreps_gated,
                )
            elif self.nonlinear_type == 'gate':
                irreps_gated = self.irreps_hidden
                irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in self.irreps_hidden)
                self.nonlinearity = GatedLinearUnit(
                    irreps_gates=irreps_gates,
                    act_gates=[ACTIVATION[self.nonlinear_act]()] * len(irreps_gates),
                    irreps_gated=irreps_gated,
                )
            else:
                raise
            
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
        if self.agnostic:
            outs = self.coefs[0](self.reshape.inverse(corr_feats[1]))
        else:
            outs = self.coefs[0](self.reshape.inverse(corr_feats[1]), node_attrs)

        for nu in range(2, self.correlation+1):
            corr_feats[nu] = self.aces[nu-2](corr_feats[nu-1], node_feats)
            if self.agnostic:
                outs = outs + self.coefs[nu-1](self.reshape.inverse(corr_feats[nu]))
            else:
                outs = outs + self.coefs[nu-1](self.reshape.inverse(corr_feats[nu]), node_attrs)    

        if hasattr(self, 'nonlinearity'):
            outs = self.nonlinearity(outs)
            
        outs = self.linear(outs)

        if sc is not None:
            outs = outs + sc

        return outs
      
# TODO, refactor
class GtpACE(Product):
    """
    An ACE implementation based on Gaunt tensor products.

    This module uses Gaunt tensor products to perform many-body expansions.
    However, this approach introduces equivariance errors (though typically small),
    lacks antisymmetric interactions, and averages over multiple many-body
    expansion paths.

    As a result, increasing the correlation order does not always lead to improved
    accuracy. This module is subject to future redesign and refinement.
    """

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
    


# class OamACE(Product):

#     def _setup(self):

#         self.sigmoid = torch.nn.Sigmoid()
#         self.silu = Activation(self.irreps_hidden1[:1], [torch.nn.SiLU()])

#         self.scalar_linear_up = Linear(
#             self.irreps_in[:1],
#             (self.irreps_hidden1[:1] * 2).regroup(),
#             bias=self.use_bias,
#         )
#         self.tensor_linear_up = Linear(
#             self.irreps_in,
#             (self.irreps_hidden1 * 2).regroup(),
#             bias=self.use_bias,
#         )

#         self.reshape = LayoutTransform2(self.tensor_linear_up.irreps_out)
#         self.tensor_ace  = uuuTensorProduct(
#             irreps_in1=self.irreps_hidden1,
#             irreps_in2=self.irreps_hidden1,
#             irreps_out=self.irreps_hidden2,
#             l1l2=self.l1l2,
#             l3s=self.l3s,
#             ictp_ictc_like=self.ictp_ictc_like,
#         )
#         self.coefs = Linear(
#             self.tensor_ace.irreps_out.simplify(),
#             self.irreps_hidden2,
#             bias=self.use_bias,
#         )

#         self.linear_gate = Linear(
#             self.irreps_in[:1],
#             f"{len(self.irreps_hidden2) * self.num_hidden_channel}x0e",
#         )

#         irreps_gated = self.irreps_hidden2
#         irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in self.irreps_hidden2)
#         self.nonlinearity = GatedLinearUnit(
#             irreps_gates=irreps_gates,
#             act_gates=[ACTIVATION[self.nonlinear_act]()] * len(irreps_gates),
#             irreps_gated=irreps_gated,
#         )

#         self.linear = Linear(
#             self.irreps_hidden2,
#             self.irreps_out,
#             bias=self.use_bias
#         )    
        
#     def forward(
#             self, 
#             node_feats: torch.Tensor, 
#             node_attrs: torch.Tensor,
#             sc: torch.Tensor,
#         ) -> torch.Tensor:

#         N = node_feats.size(0)
#         C1 = self.num_channel
#         C2 = self.num_hidden_channel

#         salar = self.scalar_linear_up(node_feats.narrow(1, 0, C1))
#         salar1, salar2 = torch.split(salar, C2, dim=1)
#         scalar_corr_feats2 = self.silu(salar1) * salar2

#         tensor = self.reshape(self.tensor_linear_up(node_feats))
#         tensor1, tensor2 = torch.split(tensor, C2, dim=1)
#         tensor_corr_feats2 = self.tensor_ace(tensor1.view(N, -1), tensor2.view(N, -1))
#         tensor_corr_feats2 = self.coefs(tensor_corr_feats2)
#         tensor_corr_feats2 = self.nonlinearity(tensor_corr_feats2, self.linear_gate(node_feats.narrow(1, 0, C1)))


#         tensor_corr_feats2[:, :C2] = tensor_corr_feats2[:, :C2] + scalar_corr_feats2
#         outs = self.linear(tensor_corr_feats2)

#         if sc is not None:
#             outs = outs + sc

#         return outs
    

class OamACE(Product):
    def _setup(self):

        self.linear_up = Linear(
            self.irreps_in,
            self.irreps_hidden1,
            bias=self.use_bias,
        ) # if self.num_channel != self.num_hidden_channel else torch.nn.Identity()

        self.aces = torch.nn.ModuleList()
        self.coefs = torch.nn.ParameterList()

        self.aces.append(
            uuuTrainableTensorProduct(
                irreps_in1=self.irreps_hidden1,
                irreps_in2=self.irreps_hidden1[:1],
                irreps_out=self.irreps_hidden2,
                l1l2=self.l1l2,
                l3s=self.l3s,
                ictp_ictc_like=self.ictp_ictc_like,
            ) 
        )
        self.coefs.append(torch.randn(self.num_elements, self.aces[-1].weight_numel))
        self.aces.append(
            uuuTrainableTensorProduct(
                irreps_in1=self.irreps_hidden1,
                irreps_in2=self.irreps_hidden1,
                irreps_out=self.irreps_hidden2,
                l1l2=self.l1l2,
                l3s=self.l3s,
                ictp_ictc_like=self.ictp_ictc_like,
            ) 
        )
        self.coefs.append(torch.randn(self.num_elements, self.aces[-1].weight_numel))

        irreps_gated = self.irreps_hidden2
        irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in self.irreps_hidden2)
        self.nonlinearity = GatedLinearUnit(
            irreps_gates=irreps_gates,
            act_gates=[ACTIVATION[self.nonlinear_act]()] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )

        self.linear_g1 = Linear(
            self.aces[0].irreps_out.simplify(),
            self.nonlinearity.irreps_in,
            bias=self.use_bias,
        )

        self.linear_g2 = Linear(
            self.aces[1].irreps_out.simplify(),
            self.nonlinearity.irreps_in,
            bias=self.use_bias,
        )

        self.linear_down = Linear(
            self.irreps_hidden2,
            self.irreps_out,
            bias=self.use_bias
        )    
        
    def forward(
            self, 
            node_feats: torch.Tensor, 
            node_attrs: torch.Tensor,
            sc: torch.Tensor,
        ) -> torch.Tensor:

        node_feats = self.linear_up(node_feats)

        corr_feats_1 = self.aces[0](
            node_feats, 
            torch.ones_like(node_feats.narrow(1, 0, self.num_hidden_channel)), 
            torch.einsum('bz, zi -> bi', node_attrs, self.coefs[0]),
        )

        corr_feats_2 = self.aces[1](
            node_feats, 
            node_feats, 
            torch.einsum('bz, zi -> bi', node_attrs, self.coefs[1]),
        )

        outs = self.linear_down(self.nonlinearity(
                self.linear_g1(corr_feats_1)
                + self.linear_g2(corr_feats_2)
            )
        )

        if sc is not None:
            outs = outs + sc

        return outs
    
class Oam2ACE(Product):

    def _setup(self):

        self.sigmoid = torch.nn.Sigmoid()

        self.coef1 = ElementLinear(
            self.irreps_in,
            self.irreps_hidden2,
            bias=self.use_bias,
            num_elements=self.num_elements,
        )

        self.linear_up2 = Linear(
            self.irreps_in,
            (self.irreps_hidden1 * 2).regroup(),
            bias=self.use_bias,
        )
        self.reshape = LayoutTransform2(self.linear_up2.irreps_out)
        self.ace2  = uuuTensorProduct(
            irreps_in1=self.irreps_hidden1,
            irreps_in2=self.irreps_hidden1,
            irreps_out=self.irreps_hidden2,
            l1l2=self.l1l2,
            l3s=self.l3s,
            ictp_ictc_like=self.ictp_ictc_like,
        )
        self.coefs2 = Linear(
            self.ace2.irreps_out.simplify(),
            self.irreps_hidden2,
            bias=self.use_bias,
        )

        self.linear_gate = Linear(
            self.irreps_in[:1],
            f"{len(self.irreps_hidden2) * self.num_hidden_channel}x0e",
        )

        irreps_gated = self.irreps_hidden2
        irreps_gates = o3.Irreps([mul, (0, 1)] for mul, _ in self.irreps_hidden2)
        self.nonlinearity = GatedLinearUnit(
            irreps_gates=irreps_gates,
            act_gates=[ACTIVATION[self.nonlinear_act]()] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )

        self.linear = Linear(
            self.irreps_hidden2,
            self.irreps_out,
            bias=self.use_bias
        )    
        
    def forward(
            self, 
            node_feats: torch.Tensor, 
            node_attrs: torch.Tensor,
            sc: torch.Tensor,
        ) -> torch.Tensor:

        N = node_feats.size(0)
        C1 = self.num_channel
        C2 = self.num_hidden_channel

        # nu = 1
        corr1_feats = self.coef1(node_feats, node_attrs)

        # nu = 2
        corr2_feats = self.reshape(self.linear_up2(node_feats))
        f1, f2 = torch.split(corr2_feats, C2, dim=1)
        corr2_feats = self.ace2(f1.view(N, -1), f2.view(N, -1))
        corr2_feats = self.coefs2(corr2_feats)
        corr2_feats = self.nonlinearity(corr2_feats, self.linear_gate(node_feats.narrow(1, 0, C1)))

        outs = self.linear(corr1_feats + corr2_feats)

        if sc is not None:
            outs = outs + sc

        return outs
    
PRODUCT: Dict[str, torch.nn.Module] = {
    "coupled": CgtpACE,
    "spatial": CgtpACE,
    "cgtp": CgtpACE,
    "nonlinear": CgtpACE,
    "glu": CgtpACE,

    "spectral": GtpACE,
    "grid": GtpACE,
    "gtp": GtpACE,

    "matrix": MtpACE,
    "mtp": MtpACE,

    "oam": OamACE,
    "oam2": Oam2ACE,
}
