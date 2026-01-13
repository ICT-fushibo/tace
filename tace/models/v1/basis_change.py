################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Type


import torch


from cartnn import ICTD


class DirectPolarizability(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        PS, DS, CS, SS = ICTD(2)
        self.register_buffer(
            "one_e", 
            SS[2].view(3, 3).to(dtype=torch.get_default_dtype()),
        )
        del PS, DS, CS, SS

    def forward(self, t0: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        t0 = torch.einsum('b, ij -> bij', t0, self.one_e)
        return t0 + t2
    
class DirectVirials(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        PS, DS, CS, SS = ICTD(2)
        self.register_buffer(
            "one_e", 
            SS[2].view(3, 3).to(dtype=torch.get_default_dtype()),
        )
        del PS, DS, CS, SS

    def forward(self, t0: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        t0 = torch.einsum('b, ij -> bij', t0, self.one_e)
        return t0 + t2

class DirectHessians(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        PS, DS, CS, SS = ICTD(2)
        self.register_buffer(
            "one_e", 
            SS[2].view(3, 3).to(dtype=torch.get_default_dtype()),
        )
        del PS, DS, CS, SS

    def forward(self, t0: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        t0 = torch.einsum('b, ij -> bij', t0, self.one_e)
        return t0 + t2
    
PropertyBasisChange: Dict[str, Type[torch.nn.Module]] = {
    'direct_virials': DirectVirials,
    'direct_polarizability': DirectPolarizability,
    'direct_hessians': DirectHessians,
}



