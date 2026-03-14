################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import List


import numpy as np
import torch
import ase.data
from omegaconf import ListConfig
from scipy.optimize import brentq
from scipy.special import jv
from torch_scatter import scatter_sum


def jn_taylor(n, x, terms=5):
    x = x.to(torch.float64)
    result = torch.zeros_like(x)
    for k in range(terms):
        coeff = ((-1) ** k) / (math.factorial(k) * math.factorial(2 * k + n + 1))
        term = coeff * x ** (2 * k + n)
        result += term
    result /= 2**n
    return result


def compute_jn_zeros(n: int, k: int) -> np.ndarray:
    def spherical_bessel_jn(r, order):
        return np.sqrt(np.pi / (2 * r)) * jv(
            order + 0.5, r
        )  # from first bessel to first spherical bessel

    zeros = []
    guess_points = np.arange(1, k + 20) * np.pi

    found = 0
    i = 0
    while found < k and i < len(guess_points) - 1:
        a, b = guess_points[i], guess_points[i + 1]
        try:
            root = brentq(
                spherical_bessel_jn, a, b, args=(n)
            )  # search roots for spherical_bessel_jn in [a, b]
            zeros.append(root)
            found += 1
        except ValueError:
            pass
        i += 1
    return np.array(zeros, dtype=np.float64)


class j0SphericalBesselBasis(torch.nn.Module):
    """The Bessel Basis is proposed in the DimeNet: https://www.cs.cit.tum.de/daml/dimenet/"""

    def __init__(
        self, cutoff: float = 6.0, num_basis: int = 8, trainable: bool = False
    ) -> None:
        super().__init__()
        self.num_basis = num_basis
        bessel_roots = (
            math.pi
            / cutoff
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_roots)
        else:
            self.register_buffer("bessel_weights", bessel_roots)

        self.register_buffer(
            "cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(math.sqrt(2.0 / cutoff), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(cutoff={self.cutoff}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )


class jnSphericalBesselBasis(torch.nn.Module):
    "arbitrary order n >= 0"

    def __init__(
        self,
        cutoff: float = 6.0,
        order: int | List[int] = 0,
        num_basis: int | List[int] = 8,
        trainable: bool = False,
    ) -> None:
        super().__init__()

        # assert torch.get_default_dtype() == torch.float64, f"This class requires float64 precision, but got {torch.get_default_dtype()}"

        num_zero = num_basis
        if isinstance(order, int):
            if order < 0:
                raise ValueError("order must be a nonnegative integer")
            order = [order]
        else:
            if not isinstance(order, (List, ListConfig)):
                raise TypeError("order must be a list of nonnegative integer")
            if not all(isinstance(x, int) and x >= 0 for x in order):
                raise ValueError("All elements of order must be nonnegative integer")

        if isinstance(num_zero, int):
            if num_zero <= 0:
                raise ValueError("num_zero must be a positive integer")
            num_zero = [num_zero]
        else:
            if not isinstance(order, (List, ListConfig)):
                raise TypeError("num_zero must be a list of positive integers")
            if not all(isinstance(x, int) and x > 0 for x in num_zero):
                raise ValueError("All elements of num_zero must be positive integers")

        if len(order) != len(num_zero):
            raise ValueError(
                f"order and num_zero must have the same length, "
                f"but got {len(order)} and {len(num_zero)}"
            )

        zeros = []
        for o, n in zip(order, num_zero):
            zeros.append(compute_jn_zeros(o, n))

        normalizer = self._compute_normalizer(cutoff, order, zeros)

        self.register_buffer(
            "normalizer",
            torch.tensor(
                [y for x in normalizer for y in x], dtype=torch.get_default_dtype()
            ).unsqueeze(0),
        )  # (1, sum(order*zeros))
        if trainable:
            self.zeros = torch.nn.Parameter(
                torch.tensor(
                    [y for x in zeros for y in x], dtype=torch.get_default_dtype()
                ).unsqueeze(0)
            )
        else:
            self.register_buffer(
                "zeros",
                torch.tensor(
                    [y for x in zeros for y in x], dtype=torch.get_default_dtype()
                ).unsqueeze(0),
            )
        self.register_buffer(
            "cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )
        self.order = order
        self.num_zero = num_zero

    def torch_jn(self, order, x):
        if order == 0:
            return torch.sin(x) / x
        elif order == 1:
            return torch.sin(x) / x**2 - torch.cos(x) / x
        elif order == 2:
            return (3 / x**3 - 1 / x) * torch.sin(x) - (3 * torch.cos(x) / x**2)
        elif order == 3:
            return (15 / x**4 - 6 / x**2) * torch.sin(x) - (
                15 / x**3 - 1 / x
            ) * torch.cos(x)
        elif order == 4:
            return (105 / x**5 - 45 / x**3 + 1 / x) * torch.sin(x) - (
                105 / x**4 - 10 / x**2
            ) * torch.cos(x)
        elif order == 5:
            return (945 / x**6 - 420 / x**4 + 15 / x**2) * torch.sin(x) - (
                945 / x**5 - 105 / x**3 + 1 / x
            ) * torch.cos(x)
        elif order == 6:
            return (10395 / x**7 - 4725 / x**5 + 210 / x**3 - 1 / x) * torch.sin(
                x
            ) - (10395 / x**6 - 1260 / x**4 + 21 / x**2) * torch.cos(x)
        elif order == 7:
            return (
                135135 / x**8 - 62370 / x**6 + 3150 / x**4 - 28 / x**2
            ) * torch.sin(x) - (
                135135 / x**7 - 17325 / x**5 + 378 / x**3 - 1 / x
            ) * torch.cos(
                x
            )
        elif order == 8:
            return (
                2027025 / x**9 - 945945 / x**7 + 51975 / x**5 - 630 / x**3 + 1 / x
            ) * torch.sin(x) - (
                2027025 / x**8 - 270270 / x**6 + 6930 / x**4 - 36 / x**2
            ) * torch.cos(
                x
            )
        elif order == 9:
            return (
                34459425 / x**10
                - 16216200 / x**8
                + 945945 / x**6
                - 13860 / x**4
                + 45 / x**2
            ) * torch.sin(x) - (
                34459425 / x**9 - 4729725 / x**7 + 135135 / x**5 - 990 / x**3 + 1 / x
            ) * torch.cos(
                x
            )
        else:

            N = (
                order + 100
            )  # Starting point for backward recursion (Miller's algorithm)
            device, dtype = x.device, x.dtype
            j = torch.zeros(N + 1, *x.shape, dtype=dtype, device=device)

            j[N] = 1e-40
            j[N - 1] = 0.0

            for n in range(N - 1, 0, -1):
                j[n - 1] = (2 * n + 1) / x * j[n] - j[n + 1]

            # Normalize using accurately computed j0
            j0_ground_truth = torch.sin(x) / x
            scale_factor = j0_ground_truth / j[0]  # shape: (...,)
            j_normalized = j * scale_factor.unsqueeze(0)

            return j_normalized[order]

    def forward(self, r: torch.Tensor):  # [..., 1]
        orig_dtype = r.dtype
        r = r.to(torch.float64)
        cutoff = self.cutoff.to(torch.float64)
        zeros = self.zeros.to(torch.float64)
        normalizer = self.normalizer.to(torch.float64)

        r = zeros * (r / cutoff)

        basis = []
        idx = 0
        for o, n in zip(self.order, self.num_zero):
            r_order = r[..., idx : idx + n]
            jn = self.torch_jn(o, r_order)
            basis.append(jn)
            idx += n
        basis = torch.cat(basis, dim=-1)

        out = basis * normalizer

        return out.to(orig_dtype)

    def _compute_normalizer(self, cutoff, order, zeros):
        normalizer = []
        for o, zero in zip(order, zeros):
            o_normalizer = []
            for i in range(len(zero)):
                z = zero[i]
                j_next = np.sqrt(np.pi / (2 * z)) * jv(o + 1 + 0.5, z)
                norm = np.sqrt(2 / (cutoff**3 * j_next**2))
                o_normalizer.append(norm)
            normalizer.append(o_normalizer)
        return normalizer

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(cutoff={self.cutoff}, order={self.order},  num_zero={self.num_zero}, "
            f"trainable={self.zeros.requires_grad})"  # zeros={self.zeros.tolist()}
        )


class GaussianBasis(torch.nn.Module):
    """
    From fairchem
    """
    def __init__(self, cutoff: float, num_basis: int = 64, width: float = 2.0):
        super().__init__()
        self.num_basis = num_basis
        offset = torch.linspace(0, cutoff, num_basis)
        self.coeff = -0.5 / (width * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset, persistent=False)

    def forward(self, r_ij: torch.Tensor) -> torch.Tensor:
        r_ij = r_ij - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(r_ij, 2))
    

class CosineCutoff(torch.nn.Module):
    """
    The fourth derivative and above are discontinuous
    """
    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.register_buffer(
            "cutoff",
            torch.tensor(cutoff, dtype=torch.get_default_dtype()),
        )

    def forward(self, r_ij: torch.Tensor) -> torch.Tensor:
        return self.calculate_envelope(r_ij, self.cutoff)
    
    @staticmethod
    def calculate_envelope(
        r_ij: torch.Tensor, cutoff: torch.Tensor
    ) -> torch.Tensor:
        x = r_ij / cutoff
        envelope = (0.5 * torch.cos(torch.pi * x) + 0.5).unsqueeze(-1)
        return envelope * (r_ij < cutoff)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff={self.cutoff})"


class MollifierCutoff(torch.nn.Module):
    """
    Derivatives of any order are continuous
    """
    def __init__(self, cutoff: float, eps: float = 1e-9):
        super().__init__()
        self.register_buffer(
            "cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )
        self.eps = eps
        
    def forward(self, r_ij: torch.Tensor) -> torch.Tensor:
        return self.calculate_envelope(r_ij, self.cutoff, self.eps)

    @staticmethod
    def calculate_envelope(
        r_ij: torch.Tensor, cutoff: torch.Tensor, eps: float = 1e-9,
    ) -> torch.Tensor:
        x = r_ij / cutoff
        envelope = torch.exp(1-1/(1-x**2+eps))
        return envelope * (r_ij < cutoff)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff={self.cutoff})"
    

class C2PolynomialCutoff(torch.nn.Module):
    """
    f(1) = f'(1) = f''(1) = 0
    Envelope function(PolynomialCutoff funciton) is proposed 
    in the DimeNet: https://www.cs.cit.tum.de/daml/dimenet/
    """

    def __init__(self, cutoff: float, p: int = 5):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.calculate_envelope(x, self.cutoff, self.p.to(torch.int))

    @staticmethod
    def calculate_envelope(
        r_ij: torch.Tensor, cutoff: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        x = r_ij / cutoff
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p)
            + p * (p + 2.0) * torch.pow(x, p + 1)
            - (p * (p + 1.0) / 2) * torch.pow(x, p + 2)
        )
        return envelope * (r_ij < cutoff)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, cutoff={self.cutoff})"


class C3PolynomialCutoff(torch.nn.Module):
    """
    Considering matbench has f'''(x), the undetermined coefficient can yield c3
    f(1) = f'(1) = f''(1) = f'''(1) = 0
    """

    def __init__(self, cutoff: float, p: int = 6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.calculate_envelope(r, self.cutoff, self.p.to(torch.int))

    @staticmethod
    def calculate_envelope(
        r_ij: torch.Tensor, cutoff: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        x = r_ij / cutoff

        # coefficients
        a = -((p + 1.0) * (p + 2.0) * (p + 3.0) / 6.0)
        b = (p * (p + 2.0) * (p + 3.0) / 2.0)
        c = -(p * (p + 1.0) * (p + 3.0) / 2.0)
        d = (p * (p + 1.0) * (p + 2.0) / 6.0)

        envelope = (
            1.0
            + a * torch.pow(x, p)
            + b * torch.pow(x, p + 1)
            + c * torch.pow(x, p + 2)
            + d * torch.pow(x, p + 3)
        )

        return envelope * (r_ij < cutoff)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, cutoff={self.cutoff})"


class AgnesiTransform(torch.nn.Module):
    """
    From MACE https://github.com/ACEsuit/mace https://doi.org/10.1063/5.0158783
    """

    def __init__(
        self,
        q: float = 0.9183,
        p: float = 4.5791,
        a: float = 1.0805,
        trainable=False,
    ):
        super().__init__()
        self.register_buffer("q", torch.tensor(q, dtype=torch.get_default_dtype()))
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer("a", torch.tensor(a, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )
        if trainable:
            self.a = torch.nn.Parameter(torch.tensor(1.0805, requires_grad=True))
            self.q = torch.nn.Parameter(torch.tensor(0.9183, requires_grad=True))
            self.p = torch.nn.Parameter(torch.tensor(4.5791, requires_grad=True))

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        source = edge_index[0]
        target = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        ) 
        Z_u = node_atomic_numbers[source]
        Z_v = node_atomic_numbers[target]
        r_0: torch.Tensor = 0.5 * (
            self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
        ) 
        r_over_r_0 = x / r_0
        return (
            1
            + (
                self.a
                * torch.pow(r_over_r_0, self.q)
                / (1 + torch.pow(r_over_r_0, self.q - self.p))
            )
        ).reciprocal_()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(a={self.a:.4f}, q={self.q:.4f}, p={self.p:.4f})"
        )


class SoftTransform(torch.nn.Module):
    """
    From MACE https://github.com/ACEsuit/mace https://doi.org/10.1063/5.0158783
    """

    def __init__(self, a: float = 0.2, b: float = 3.0, trainable=False):
        super().__init__()
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )
        if trainable:
            self.a = torch.nn.Parameter(torch.tensor(a, requires_grad=True))
            self.b = torch.nn.Parameter(torch.tensor(b, requires_grad=True))
        else:
            self.register_buffer("a", torch.tensor(a))
            self.register_buffer("b", torch.tensor(b))

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        source = edge_index[0]
        target = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        )
        Z_u = node_atomic_numbers[source]
        Z_v = node_atomic_numbers[target]
        r_0 = (self.covalent_radii[Z_u] + self.covalent_radii[Z_v]) / 4
        r_over_r_0 = x / r_0
        y = (
            x
            + (1 / 2) * torch.tanh(-r_over_r_0 - self.a * torch.pow(r_over_r_0, self.b))
            + 1 / 2
        )
        return y

    def __repr__(self):
        return f"{self.__class__.__name__}(a={self.a.item()}, b={self.b.item()})"

   
class ZBLBasis(torch.nn.Module):
    """
    Metal units
    From LAMMPS pair_zbl_const.h
    Code from MACE
    """

    p: torch.Tensor

    def __init__(self, cutoff_fn: str, trainable: bool = False, p: int = 5):
        super().__init__()
        # Pre-calculate the p coefficients for the ZBL potential
        self.register_buffer(
            "c",
            torch.tensor(
                [0.1818, 0.5099, 0.2802, 0.02817], dtype=torch.get_default_dtype()
            ),
        )
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )
        if trainable:
            self.a_exp = torch.nn.Parameter(torch.tensor(0.300, requires_grad=True))
            self.a_prefactor = torch.nn.Parameter(
                torch.tensor(0.4543, requires_grad=True)
            )
        else:
            self.register_buffer("a_exp", torch.tensor(0.300))
            self.register_buffer("a_prefactor", torch.tensor(0.4543))

        self.is_polynomial_cutoff = False
        if cutoff_fn == 'mollifier':
            self.cutoff_fn = MollifierCutoff.calculate_envelope
        elif cutoff_fn == 'cosine': 
            self.cutoff_fn = CosineCutoff.calculate_envelope
        elif cutoff_fn == 'c3poly': 
            self.is_polynomial_cutoff = True
            self.cutoff_fn = C3PolynomialCutoff.calculate_envelope
        else:
            self.is_polynomial_cutoff = True
            self.cutoff_fn = C2PolynomialCutoff.calculate_envelope
        
    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        source = edge_index[0]
        target = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(-1)
        Z_u = node_atomic_numbers[source]
        Z_v = node_atomic_numbers[target]
        a = (
            self.a_prefactor
            * 0.529
            / (torch.pow(Z_u, self.a_exp) + torch.pow(Z_v, self.a_exp))
        )
        r_over_a = x / a
        phi = (
            self.c[0] * torch.exp(-3.2 * r_over_a)
            + self.c[1] * torch.exp(-0.9423 * r_over_a)
            + self.c[2] * torch.exp(-0.4028 * r_over_a)
            + self.c[3] * torch.exp(-0.2016 * r_over_a)
        )
        v_edges = (14.3996 * Z_u * Z_v) / x * phi
        r_max = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
        if self.is_polynomial_cutoff:
            envelope = self.cutoff_fn(x, r_max, self.p)
        else:
            envelope = self.cutoff_fn(x, r_max)    
        v_edges = 0.5 * v_edges * envelope
        V_ZBL = scatter_sum(v_edges, target, dim=0, dim_size=node_attrs.size(0))
        return V_ZBL.squeeze(-1)

    def __repr__(self):
        return f"{self.__class__.__name__}(c={[float(f'{c:.4f}') for c in self.c.tolist()]})"


class RadialBasis(torch.nn.Module):
    def __init__(
        self,
        cutoff: float = 6.0,
        num_basis: int = 8,
        polynomial_cutoff: int = 5,
        radial_basis: str = "j0",
        distance_transform=None,
        order: int | List[int] = [0],
        trainable: bool = False,
        apply_cutoff: bool = True,
        cutoff_fn: str ='mollifier', # ['cosine', 'mollifier', 'polynomial']
        gaussian_width: float = 2.0,
    ):
        super().__init__()
        if radial_basis == "bessel" or radial_basis == "j0":
            self.radial_fn = j0SphericalBesselBasis(
                cutoff=cutoff, num_basis=num_basis, trainable=trainable
            )
        elif radial_basis == "jn":
            self.radial_fn = jnSphericalBesselBasis(
                cutoff=cutoff,
                order=order,
                num_basis=num_basis,
                trainable=trainable,
            )
        # elif radial_basis == "chebychev":
        #     self.radial_fn = ChebyshevTBasis(
        #         cutoff=cutoff,
        #         num_basis=num_basis,
        #     )
        elif radial_basis == "gaussian":
            self.radial_fn = GaussianBasis(
                cutoff=cutoff,
                num_basis=num_basis,
                width=gaussian_width,
            )
        else:
            raise ValueError(f"Unknown radial_basis: {radial_basis}")
        
        if distance_transform == "Agnesi":
            self.distance_transform = AgnesiTransform()
        elif distance_transform == "Soft":
            self.distance_transform = SoftTransform()
        
        if cutoff_fn == 'mollifier':
            self.cutoff_fn = MollifierCutoff(cutoff=cutoff)
        elif cutoff_fn == 'cosine': 
            self.cutoff_fn = CosineCutoff(cutoff=cutoff)
        elif cutoff_fn == 'c3poly':
            self.cutoff_fn = C3PolynomialCutoff(cutoff=cutoff, p=polynomial_cutoff)
        else:
            self.cutoff_fn = C2PolynomialCutoff(cutoff=cutoff, p=polynomial_cutoff)

        if not isinstance(num_basis, int):
            num_basis = sum(num_basis)
            self.out_dim = num_basis
            self.num_basis = num_basis
        else:
            self.out_dim = num_basis
            self.num_basis = num_basis

        self.apply_cutoff = apply_cutoff

    def forward(
        self,
        edge_length: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        cutoff = self.cutoff_fn(edge_length)
        if hasattr(self, "distance_transform"):
            edge_length = self.distance_transform(
                edge_length, node_attrs, edge_index, atomic_numbers
            )
        radial = self.radial_fn(edge_length)
        if self.apply_cutoff:
            return radial * cutoff, None
        else:
            return radial, cutoff
    
# # draw
# import torch
# import matplotlib.pyplot as plt
# import numpy as np

# torch.set_default_dtype(torch.float64)
# cutoff = 6.0
# num_basis = 8
# num_basis_list = [15] # [8, 9, 10]

# j0 = j0SphericalBesselBasis(cutoff=cutoff, num_basis=num_basis)
# jn = jnSphericalBesselBasis(cutoff=cutoff, order=8, num_basis=num_basis_list)

# x = torch.linspace(0, cutoff, 200).unsqueeze(-1)

# a = j0(x)  # shape: [200, 8]
# b = jn(x)  # shape: [200, 8]


# x_np = x.squeeze().numpy()        # shape: [200]
# a = a.detach().numpy()         # shape: [200, 8]
# b = b.detach().numpy()         # shape: [200, 8]


# plt.figure(figsize=(10, 5))

# for i in range(a.shape[1]):
#     # plt.plot(x_np, a[:, i], label=f'j0_basis_{i}', linestyle='-')
#     plt.plot(x_np, b[:, i], label=f'jn_basis_{i}', linestyle='--')

# plt.xlabel("x")
# plt.ylabel("Basis value")
# plt.title("Comparison of j0 vs jn spherical Bessel basis")
# plt.legend(fontsize='small', ncol=2)
# plt.grid(True)
# plt.tight_layout()
# plt.show()