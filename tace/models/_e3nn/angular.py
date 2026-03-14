r"""Spherical Harmonics as polynomials of x, y, z"""

from typing import Union, List, Any

import math

import torch

from e3nn.o3._irreps import Irreps
from e3nn import get_optimization_defaults
from e3nn.util.jit import compile_mode
from e3nn.o3._spherical_harmonics import _spherical_harmonics


class SphericalHarmonics(torch.nn.Module):
    """
    Copy from e3nn for torch.save
    """

    normalize: bool
    normalization: str
    _ls_list: List[int]
    _lmax: int
    _is_range_lmax: bool
    _prof_str: str

    def __init__(
        self,
        irreps_out: Union[int, List[int], str, Irreps],
        normalize: bool,
        normalization: str = "integral",
        irreps_in: Any = None,
    ) -> None:
        super().__init__()
        self.normalize = normalize
        self.normalization = normalization
        assert normalization in ["integral", "component", "norm"]

        if isinstance(irreps_out, str):
            irreps_out = Irreps(irreps_out)
        if isinstance(irreps_out, Irreps) and irreps_in is None:
            for mul, (l, p) in irreps_out:
                if l % 2 == 1 and p == 1:
                    irreps_in = Irreps("1e")
        if irreps_in is None:
            irreps_in = Irreps("1o")

        irreps_in = Irreps(irreps_in)
        if irreps_in not in (Irreps("1x1o"), Irreps("1x1e")):
            raise ValueError(
                f"irreps_in for SphericalHarmonics must be either a vector (`1x1o`) or a pseudovector (`1x1e`), "
                f"not `{irreps_in}`"
            )
        self.irreps_in = irreps_in
        input_p = irreps_in[0].ir.p  # pylint: disable=no-member

        if isinstance(irreps_out, Irreps):
            ls = []
            for mul, (l, p) in irreps_out:
                if p != input_p**l:
                    raise ValueError(
                        f"irreps_out `{irreps_out}` passed to SphericalHarmonics asked for an output of l = {l} with parity "
                        f"p = {p}, which is inconsistent with the input parity {input_p} — the output parity should have been "
                        f"p = {input_p**l}"
                    )
                ls.extend([l] * mul)
        elif isinstance(irreps_out, int):
            ls = [irreps_out]
        else:
            ls = list(irreps_out)

        irreps_out = Irreps([(1, (l, input_p**l)) for l in ls]).simplify()
        self.irreps_out = irreps_out
        self._ls_list = ls
        self._lmax = max(ls)
        self._is_range_lmax = ls == list(range(max(ls) + 1))
        self._prof_str = f"spherical_harmonics({ls})"

        _lmax = 11
        if self._lmax > _lmax:
            raise NotImplementedError(
                f"spherical_harmonics maximum l implemented is {_lmax}, send us an email to ask for more"
            )

        # if get_optimization_defaults()["jit_mode"] == "script":
        #     self.sph_func = torch.jit.script(_spherical_harmonics)
        # elif get_optimization_defaults()["jit_mode"] == "compile":
        #     self.sph_func = torch.compile(_spherical_harmonics, fullgraph=True)
        # else:
        #     self.sph_func = _spherical_harmonics
        self.sph_func = _spherical_harmonics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # - PROFILER - with torch.autograd.profiler.record_function(self._prof_str):
        if self.normalize:
            x = torch.nn.functional.normalize(x, dim=-1)  # forward 0's instead of nan for zero-radius

        sh = self.sph_func(self._lmax, x[..., 0], x[..., 1], x[..., 2])

        if not self._is_range_lmax:
            sh = torch.cat([sh[..., l * l : (l + 1) * (l + 1)] for l in self._ls_list], dim=-1)

        if self.normalization == "integral":
            sh.div_(math.sqrt(4 * math.pi))
        elif self.normalization == "norm":
            sh.div_(
                torch.cat(
                    [math.sqrt(2 * l + 1) * torch.ones(2 * l + 1, dtype=sh.dtype, device=sh.device) for l in self._ls_list]
                )
            )

        return sh