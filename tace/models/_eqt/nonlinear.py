################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import torch
import equitorch as eqt
from equitorch.irreps import check_irreps
from e3nn import o3


from ..mlp import MLP


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

