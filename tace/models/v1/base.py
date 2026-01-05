import torch
from typing import Sequence, Dict

class ElementAwareMixin:
    def _init_elements(self, atomic_numbers: Sequence[int]):
        zs = sorted(list(set(int(z) for z in atomic_numbers)))
        self.atomic_numbers = zs
        self.num_elements = len(zs)
        # z -> element index
        self._z_to_idx: Dict[int, int] = {
            z: i for i, z in enumerate(zs)
        }

    def get_element_parameter(
            self, 
            param: torch.Tensor, 
            atomic_numbers: Sequence[int],
        ) -> torch.Tensor:

        param_list = []
        for z in atomic_numbers:
            param_list.append(param[self._z_to_idx[z]])

        return torch.stack(param_list, dim=0)
