###############################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Dict, List


import torch
import opt_einsum_fx
from e3nn import o3
from cartnn import ICTD, Irreps, SymmetricContraction


from .utils import add_dict_to_left
from .paths import satisfy, generate_prod_paths
from .linear import Linear, ElementLinear
from .einsum import ProdEinsumTC
from .base import Product

PATH = 4
BATCH = 5
CHANNEL = 6


class SpectralACE(Product):
    def _setup(self) -> None:

        # === ICT ===
        for r in self.irreps_in:
            DS = ICTD(r, r)[1]
            self.register_buffer(f"D_{r}_{r}_1", DS[0].to(torch.get_default_dtype()))
            del DS
        
        # === used in init === 
        correlation = self.correlation

        # === init ===
        self.correlation = correlation
        self.lmax_in = max(self.irreps_in)
        self.ls_out = self.irreps_out
       
        # === prod ===
        self.paths_list_list, self.exprs_list_list = generate_prod_paths(
            self.lmax_in, self.ls_out, self.correlation, self.l1l2, None, None,
        )
        
        self.aces = torch.nn.ModuleList()
        for v in range(self.correlation - 1):
            ace = torch.nn.ModuleList()
            for comb, expr in zip(
                self.paths_list_list[v], self.exprs_list_list[v]
            ):
                ace.append(ProdEinsumTC((comb)))
            self.aces.append(ace)

        # === l3_count for each nu ===
        # nu = 1 
        nu_l3_count = {1: {l3: 0 for l3 in range(self.lmax_in + 1)}}  
        for l3 in range(self.lmax_in + 1):
            nu_l3_count[1][l3] += 1

        # nu > 1
        for nu in range(2, self.correlation + 1):
            nu_l3_count[nu] = {l3: 0 for l3 in range(self.lmax_in + 1)} 
            for l1 in range(self.lmax_in + 1):
                for l2 in range(self.lmax_in+ 1):
                    for l3 in range(abs(l1 - l2), min(self.lmax_in, l1 + l2) + 1, 2):
                        k = (l1 + l2 - l3) // 2
                        if satisfy(l1, l2, self.l1l2):
                            nu_l3_count[nu][l3] += nu_l3_count[nu-1][l1]


        self.coefs = torch.nn.ModuleDict()
        for nu in range(1, self.correlation+1):
            channels_in = [] 
            channels_out = []    
            for l in self.ls_out:
                channels_in.append(self.num_channel * sum([nu_l3_count[nu][l]]))
                channels_out.append(self.num_hidden_channel)
            self.coefs[str(nu)] = ElementLinear(
                    self.ls_out,
                    channels_in,
                    channels_out,
                    bias=self.use_bias,
                    num_elements=self.num_elements,
                )
             
        self.linear = Linear(
            self.ls_out,
            self.num_hidden_channel,
            self.num_channel,
            bias=self.use_bias,
        )


    def D(self, l: int):
            return dict(self.named_buffers())[f"D_{l}_{l}_1"]
    
    def forward(
        self,
        node_feats: Dict[int, torch.Tensor],
        node_attrs: torch.Tensor,
        sc: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        
        corr_feats = {
            0: {
                l: [node_feats[l]] for l in node_feats
            }
        }

        for nu, this_ace in enumerate(self.aces):
            corr_feats[nu + 1] = {l: [] for l in range(self.lmax_in + 1)}
            for idx, ace in enumerate(this_ace):
                l1, l2, l3, _ = self.paths_list_list[nu][idx]
                tmp = ace(
                    torch.stack(corr_feats[nu][l1], dim=0),
                    node_feats[l2],
                )

                P = tmp.size(0); B = tmp.size(1); C = tmp.size(-1)
                
                tmp = tmp.reshape(P, B, -1, C)
                tmp = torch.einsum(
                    "abic, ij -> abjc",  
                    tmp, self.D(l3)
                ).reshape((P, B) + (3,) * l3 + (C,))

                tmp = torch.unbind(tmp, dim=0)
                corr_feats[nu + 1][l3].extend(tmp)

        out_dict = {}
        for nu_str, linear in self.coefs.items():
            nu = int(nu_str)
            tmp_dict = {}
            for l3 in self.ls_out:
                tmp_dict[l3] = torch.cat([t for t in corr_feats[nu-1][l3]], dim=-1)
            out_dict = add_dict_to_left(out_dict, linear(tmp_dict, node_attrs))
            
        return add_dict_to_left(self.linear(out_dict), sc)


# class CartesiannjContraction(Product):
#     def __init__(
#         self,
#         num_channel: int = 64,
#         num_channel_hidden: int = 64,
#         lmax_in: int = 3,
#         ls_out: List[int] = 2,
#         atomic_numbers: List[int] = [],
#         prod: Dict = {},
#         bias: bool = False,
#         layer: int = -1,
#         num_layers: int = 2,
#     ) -> None:
        
#         '''
#         This product basis is different from original TACE's SelfContraction.
#         Simply put, 
#         Self contraction: tensor product + tensor contraction;
#         PrecomputedSelfContraction: tensor product and precomute cartesian_nj.
#         '''
#         super().__init__()

#         self.lmax_in = lmax_in
#         self.ls_out = ls_out
#         self.num_channel = num_channel
#         self.num_channel_hidden = num_channel_hidden

#         if isinstance(prod.get("element_aware", True), bool):
#             element_aware = {}
#             for l in ls_out:
#                 element_aware[l] = prod.get("element_aware", True)
#         else:
#              element_aware = prod["element_aware"]
#         if isinstance(prod.get("coupled_channel", True), bool):
#             coupled_channel = {}
#             for l in ls_out:
#                 coupled_channel[l] = prod.get("coupled_channel", True)
#         else:
#             coupled_channel = prod["coupled_channel"]

#         correlation = prod.get("correlation", [3,] * num_layers)[layer]
#         assert correlation == 2, "Only nu=2 precomputed product basis are useful in Cartesian space "
    
#         node_feats_irreps = "+".join(
#             f"{num_channel_hidden}x{l}e" for l in range(lmax_in + 1)
#         )
#         target_irreps = "+".join(f"{num_channel_hidden}x{l}e" for l in ls_out)

#         self.symmetric_contractions = SymmetricContraction(
#             irreps_in=Irreps(node_feats_irreps),
#             irreps_out=Irreps(target_irreps),
#             correlation=correlation,
#             num_elements=len(atomic_numbers),
#             element_aware=element_aware,
#             coupled_channel=coupled_channel,
#         )

#         self.linear = SelfInteraction(
#             in_channel=num_channel_hidden,
#             out_channel=num_channel,
#             ls=ls_out,
#             bias=bias and layer == num_layers -1,
#         )

#     def forward(
#         self,
#         node_feats: Dict[int, torch.Tensor],
#         node_attrs: torch.Tensor,
#         sc: Dict[int, torch.Tensor],
#     ) -> torch.Tensor:
        
#         B = node_feats[0].size(0)
#         C = node_feats[0].size(1)

#         node_feats_list = []
#         for l in range(self.lmax_in + 1):
#             node_feats_list.append(node_feats[l].view(B, C, -1))
#         node_feats = torch.cat(node_feats_list, dim=-1)
#         node_feats = self.symmetric_contractions(node_feats, node_attrs)

#         out_dict = {}
#         for idx, l in enumerate(self.ls_out):
#             out_dict[l] = node_feats[idx].view(*((B, self.num_channel) + (3,) * l))

#         return add_dict_to_left(self.linear(out_dict), sc)
    


class SpatialACE(Product):

    def _setup(self):

        normalization="component"

        to_s2 = o3.ToS2Grid(
            self.truncation, 
            (self.num_latitude, self.num_longitude), 
            normalization=normalization,
        )
        from_s2 = o3.FromS2Grid(
            (self.num_latitude, self.num_longitude), 
            self.truncation, 
            normalization=normalization,
        )

        sph_to_grid = torch.einsum(
                "mbi, am -> bai", to_s2.shb, to_s2.sha
            ).detach()
        
        cart_to_grid_list = []

        start = 0
        for l in self.irreps_in: 
            this_grid = sph_to_grid[:, :, start:start+2*l+1]
            start = start+2*l+1
            PS, DS, CS, SS = ICTD(l)
            C = CS[0]
            cart_to_grid_list.append(
                torch.einsum('bam, nm -> ban', this_grid, C) # [beta, alpha, 3**l]
            )
        
        self.register_buffer(
            "to_grid", 
            torch.cat(cart_to_grid_list, dim=-1),
            persistent=False,
        )

        cart_from_grid_list = []
        sph_from_grid = torch.einsum(
                "am, mbi -> bai", from_s2.sha, from_s2.shb
            ).detach()
        start = 0
        for l in self.irreps_in: 
            this_grid = sph_from_grid[:, :, start:start+2*l+1]
            start = start+2*l+1
            PS, DS, CS, SS = ICTD(l)
            CT = SS[0]
            cart_from_grid_list.append(
                torch.einsum('bam, mn -> ban', this_grid, CT,) # [beta, alpha, 3**l]
            )

        self.register_buffer(
            "from_grid", 
            torch.cat(cart_from_grid_list, dim=-1),
            persistent=False,
        )

        self.linear = Linear(
            self.irreps_out,
            self.num_channel,
            self.num_channel,
            bias=self.use_bias,
        )

        self.slices = []
        self.coefs = torch.nn.ModuleList()

        start = 0
        for l in self.irreps_out:
            trace = torch.fx.symbolic_trace(
                lambda a, b, d: torch.einsum('Bz, zCc, BCi -> Bci', a, b, d)
            )
            graph = (
                opt_einsum_fx.optimize_einsums_full(
                    model=trace,
                    example_inputs=(
                        torch.randn([256, 89]),
                        torch.randn([89, 128, 64]),
                        torch.randn([256, 128, 40]),
                    ),
                )
            )
            self.coefs.append(graph)
            self.slices.append(slice(start, start+3**l))
            start = start+3**l

        self.weight = torch.nn.Parameter(
            torch.empty(
                len(self.irreps_out),
                self.num_elements, 
                self.correlation*self.num_channel, 
                self.num_channel,
            )
        )
        self.alpha = 1.0 / math.sqrt(self.correlation*self.num_channel)
        if self.use_bias and 0 in self.irreps_out:
            self.bias = torch.nn.Parameter(
                torch.empty(self.num_elements, self.num_channel)
            )
        else:
            self.register_parameter("bias", None)

        if self.trainable_scale:
            self.scale = torch.nn.Parameter(torch.ones(self.correlation-1))
        else:
            self.register_buffer("scale", torch.ones(self.correlation-1), persistent=False)

        
        def _sum(lmax: int) -> int:
            _sum = 0
            for l in range(lmax+1):
                _sum += 3**l
            return _sum


        self.num_padding = _sum(self.truncation) - _sum(self.lmax)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def _to_grid(self, x: torch.Tensor):           
        return torch.einsum("bai, Bci -> Bcba", self.to_grid, x)

    def _from_grid(self, x):
        return torch.einsum("bai, Bcba -> Bci", self.from_grid, x)

    def forward(
        self,
        node_feats: Dict[int, torch.Tensor],  
        node_attrs: torch.Tensor,
        sc: Dict[int, torch.Tensor] | None = None
    ):
        dtype = node_feats[0].dtype
        device = node_feats[0].device

        B, C = node_feats[0].shape[:2]
        node_feats_list = []
        for r, v in node_feats.items():
            node_feats_list.append(v.view(B, C, -1))
        flatten_node_feats = torch.cat(node_feats_list, dim=-1)
    
        dim = flatten_node_feats.size(-1)
        pad_shape = list(flatten_node_feats.shape)
        pad_shape[-1] = self.num_padding
        padding = torch.zeros(
            pad_shape,
            dtype=dtype,
            device=device
        )
        node_feats = torch.cat([flatten_node_feats, padding], dim=-1)
        base_grid = self._to_grid(node_feats)
        corr_feats_list = [base_grid]
        grid_prev = base_grid


        for nu in range(2, self.correlation + 1):
            grid_prev = grid_prev * base_grid
            corr_feats_list.append(grid_prev * self.scale[nu-2])

        corr_feats = torch.cat(corr_feats_list, dim=1)
        corr_feats = torch.einsum('bai, BCba -> BCi', self.from_grid, corr_feats)
        corr_feats = corr_feats[:, :, :dim]

        outs = {}
        for idx, (sl, coef, l) in enumerate(zip(self.slices, self.coefs, self.irreps_out)):
            outs[l] = (
                coef(node_attrs, self.weight[idx], corr_feats[:, :, sl]) * self.alpha
            ).view(B, C, *(3,)*l)
            
    
        if self.bias is not None:
            bias = torch.einsum('Bz, zi -> Bi', node_attrs, self.bias)
            _0e = outs[0] + bias
            outs[0] = _0e
        
        outs = self.linear(outs)

        if sc is not None:
            outs = add_dict_to_left(outs, sc)
        return outs   
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nlon={self.num_longitude}, nlat={self.num_latitude})"
    
PRODUCT: Dict[str, torch.nn.Module] = {
    "coupled": SpectralACE,
    "spectral": SpectralACE,
    "grid": SpatialACE,
    "spatial": SpatialACE,
}