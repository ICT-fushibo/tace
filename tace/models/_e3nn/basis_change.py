################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Type


import torch


from ..ictd import ICTD


class DirectPolarizability(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        PS, DS, CS, SS = ICTD(2)
        self.register_buffer(
            "zero", 
            SS[2].view(3, 3).to(dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "two", 
            SS[0].view(5, 3, 3).to(dtype=torch.get_default_dtype()),
        )
        del PS, DS, CS, SS

    def forward(self, t0: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        t0 = torch.einsum('b, ij -> bij', t0, self.zero)
        t2 = torch.einsum('bk, kij -> bij', t2, self.two)
        return t0 + t2
    
class DirectVirials(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        PS, DS, CS, SS = ICTD(2)
        self.register_buffer(
            "zero", 
            SS[2].view(3, 3).to(dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "two", 
            SS[0].view(5, 3, 3).to(dtype=torch.get_default_dtype()),
        )
        del PS, DS, CS, SS

    def forward(self, t0: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        t0 = torch.einsum('b, ij -> bij', t0, self.zero)
        t2 = torch.einsum('bk, kij -> bij', t2, self.two)
        return t0 + t2


PropertyBasisChange: Dict[str, Type[torch.nn.Module]] = {
    'direct_virials': DirectVirials,
    'direct_polarizability': DirectPolarizability,
}



