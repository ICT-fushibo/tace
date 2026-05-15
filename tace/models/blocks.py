################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import List, Dict


import torch
from e3nn import o3


from ..utils.torch_scatter import scatter_sum

def format_list(obj, ndigits=4):
    if isinstance(obj, int):
        return str(obj)
    elif isinstance(obj, float):
        return f"{obj:.{ndigits}f}"
    elif isinstance(obj, (list, tuple)):
        return "[" + ", ".join(format_list(x, ndigits) for x in obj) + "]"
    else:
        return str(obj)
  
  
class OneHotToAtomicEnergy(torch.nn.Module):
    def __init__(self, atomic_energies: List[Dict[int, float]]) -> None:
        super().__init__()
        assert atomic_energies is not None
        atomic_energy_list = []
        for atomic_energy in atomic_energies:
            atomic_energy_list.append(
                [float(v) for _, v in atomic_energy.items()]
            )
        self.register_buffer(
            "atomic_energy",
            torch.tensor(
                atomic_energy_list,
                dtype=torch.get_default_dtype(),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return torch.matmul(x, self.atomic_energy.T)

    def __repr__(self):
        s = f"{self.__class__.__name__}(\n"
        s += "  atomic_energies = {\n"

        data = self.atomic_energy.detach().cpu().numpy()

        for i in range(data.shape[0]):
            s += f"    fidelity {i}: {format_list(data[i].tolist(), 4)}\n"

        s += "  }\n"
        s += ")"
        return s


class ScaleShift(torch.nn.Module):
    def __init__(
        self,
        scale_dicts: List[Dict[int, float]] = [],
        shift_dicts: List[Dict[int, float]] = [],
        scale_trainable: bool = False,
        shift_trainable: bool = False,
        all_atoms: bool = False,
    ):
        super().__init__()

        self.all_atoms = all_atoms
        self.has_scale = len(scale_dicts) > 0
        self.has_shift = len(shift_dicts) > 0
        self.num_fidelities = max(len(scale_dicts), len(shift_dicts))
        atomic_numbers = sorted(
            set().union(*[d.keys() for d in scale_dicts] if scale_dicts else [])
            | set().union(*[d.keys() for d in shift_dicts] if shift_dicts else [])
        )
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )

        if self.has_scale:
            scale_list = []
            for d in scale_dicts:
                scale_list.append([d.get(z, 1.0) for z in atomic_numbers])
            scale_tensor = torch.tensor(scale_list, dtype=torch.get_default_dtype())
            if scale_trainable:
                self.scale = torch.nn.Parameter(scale_tensor)
            else:
                self.register_buffer("scale", scale_tensor)

        if self.has_shift:
            shift_list = []
            for d in shift_dicts:
                shift_list.append([d.get(z, 0.0) for z in atomic_numbers])
            shift_tensor = torch.tensor(shift_list, dtype=torch.get_default_dtype())
            if shift_trainable:
                self.shift = torch.nn.Parameter(shift_tensor)
            else:
                self.register_buffer("shift", shift_tensor)

    def forward(self, node_energy, node_attrs, ptr, edge_index, batch, node_fidelity):
        if not (self.has_scale or self.has_shift):
            return node_energy

        num_graphs = ptr.numel() - 1
        num_nodes = ptr[1:] - ptr[:-1]

        if self.all_atoms:
            if self.has_scale:
                node_scale = (node_attrs * self.scale[node_fidelity]).sum(dim=-1)
                node_energy = node_energy * node_scale

            if self.has_shift:
                node_shift = (node_attrs * self.shift[node_fidelity]).sum(dim=-1)
                node_energy = node_energy + node_shift
        else:
            if edge_index.numel() == 0:
                num_edges = torch.zeros(num_graphs, dtype=torch.int64, device=node_energy.device)
            else:
                edge_batch = batch[edge_index[1]]
                num_edges = torch.bincount(edge_batch, minlength=num_graphs)

            isolated_mask = (num_nodes == 1) & (num_edges == 0)

            if self.has_scale:
                node_scale = (node_attrs * self.scale[node_fidelity]).sum(dim=-1)
                if isolated_mask.any():
                    isolated_nodes = torch.isin(batch, torch.where(isolated_mask)[0])
                    node_scale[isolated_nodes] = 0.0
                node_energy = node_energy * node_scale

            if self.has_shift:
                node_shift = (node_attrs * self.shift[node_fidelity]).sum(dim=-1)
                if isolated_mask.any():
                    isolated_nodes = torch.isin(batch, torch.where(isolated_mask)[0])
                    node_shift[isolated_nodes] = 0.0
                node_energy = node_energy + node_shift

        return node_energy

    def __repr__(self):

        s = f"{self.__class__.__name__}(\n"
        s += f"  atomic_numbers = {self.atomic_numbers.tolist()}\n"

        if self.has_scale:
            s += "  scale = {\n"
            for lvl in range(self.scale.shape[0]):
                data = self.scale[lvl].detach().cpu().numpy().tolist()
                s += f"    fidelity {lvl}: {format_list(data, 4)}\n"
            s += "  }\n"
        else:
            s += "  scale = None\n"

        if self.has_shift:
            s += "  shift = {\n"
            for lvl in range(self.shift.shape[0]):
                data = self.shift[lvl].detach().cpu().numpy().tolist()
                s += f"    fidelity {lvl}: {format_list(data, 4)}\n"
            s += "  }\n"
        else:
            s += "  shift = None\n"

        s += f"all_atoms={self.all_atoms}\n"
        s += ")"
        return s
    
    @classmethod
    def build_from_config(cls, statistics, cfg: Dict):
        required_keys = [
            "scale_type",
            "shift_type",
            "scale_trainable",
            "shift_trainable",
        ]
        assert all(
            k in cfg for k in required_keys
        ), f"Missing keys in scale_shift config: {required_keys}"

        scale_key = cfg["scale_type"]
        shift_key = cfg["shift_type"]

        scale_dicts = []
        shift_dicts = []

        for stats in statistics:
            scale_stat = {z: 1.0 for z in stats["atomic_numbers"]}
            shift_stat = {z: 0.0 for z in stats["atomic_numbers"]}

            if scale_key is not None:
                assert scale_key in stats, f"{scale_key} not found in statistics"
                scale_stat = stats[scale_key]

            if shift_key is not None:
                assert shift_key in stats, f"{shift_key} not found in statistics"
                shift_stat = stats[shift_key]

            scale_dict = {int(k): float(v) for k, v in scale_stat.items()}
            shift_dict = {int(k): float(v) for k, v in shift_stat.items()}
            scale_dicts.append(scale_dict)
            shift_dicts.append(shift_dict)

        return cls(
            scale_dicts=scale_dicts,
            shift_dicts=shift_dicts,
            scale_trainable=cfg["scale_trainable"],
            shift_trainable=cfg["shift_trainable"],
            all_atoms=cfg['all_atoms']
        )


def has_no_isolated_atoms(edge_index: torch.Tensor, num_atoms: int):
    if torch.all(
        scatter_sum(
            torch.ones((num_atoms,))[edge_index[0]], 
            edge_index[1], 
            dim_size=num_atoms, 
            dim=0,
        )
    ):
        return True
    else:
        return False
       
# class kSphericalHarmonics(torch.nn.Module):
#     def __init__(
#         self,
#         num_sample: int,
#         lmax: int
#     ):
#         super().__init__()

#         sh = SphericalHarmonics(
#             irreps_out=o3.Irreps.spherical_harmonics(lmax, p=-1),
#             normalize=False,
#         )
#         k = FibonacciLattice.generate(num_sample) # [num_sammple, 3]
#         ksh = sh(k)
#         self.register_buffer("k")



#     def forward(self, node_energy, node_attrs, ptr, edge_index, batch, node_fidelity):

import torch
import torch.nn as nn
from collections import defaultdict
from e3nn.o3 import Irreps


class IrrepsLSortedSum(nn.Module):
    """
    x: irreps_in1
    y: irreps_in2

    输出：
    - l 升序
    - p = -1 -> +1
    - 同 l 部分相加
    """

    def __init__(self, irreps_in1: Irreps, irreps_in2: Irreps):
        super().__init__()

        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)

        self.blocks1 = self._build_blocks(self.irreps_in1)
        self.blocks2 = self._build_blocks(self.irreps_in2)

        self.common_ls = sorted(set(self.blocks1.keys()) & set(self.blocks2.keys()))

    def _build_blocks(self, irreps):
        """
        return:
            dict[l] -> list of (p, start, end)
        """
        out = defaultdict(list)
        idx = 0

        for mul, ir in irreps:
            l = ir.l
            p = ir.p
            dim = ir.dim

            out[l].append((p, idx, idx + mul * dim))
            idx += mul * dim

        # -1 → +1
        for l in out:
            out[l].sort(key=lambda x: x[0])

        return out

    def _extract(self, x, blocks):
        out = {}
        for l, items in blocks.items():
            parts = []
            for p, s, e in items:
                parts.append(x[..., s:e])
            out[l] = torch.cat(parts, dim=-1)
        return out

    def forward(self, x, y):
        x_l = self._extract(x, self.blocks1)
        y_l = self._extract(y, self.blocks2)

        outputs = []

        for l in sorted(self.common_ls):

            x_feat = x_l[l]
            y_feat = y_l[l]

            d = min(x_feat.shape[-1], y_feat.shape[-1])

            outputs.append(x_feat[..., :d] + y_feat[..., :d])

        if len(outputs) == 0:
            raise ValueError("No common l between irreps_in1 and irreps_in2")

        return torch.cat(outputs, dim=-1)