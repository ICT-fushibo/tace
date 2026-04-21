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
    ):
        super().__init__()

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