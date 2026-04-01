################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import numpy as np
import ase

WTC_U_SHIFT = {
    23: -2.271,  # V
    24: -2.458,  # Cr
    25: -1.602,  # Mn
    26: -1.925,  # Fe
    27: -2.004,  # Co
    28: -2.586,  # Ni
    42: -4.895,  # Mo
    74: -5.875,  # W
}

def apply_wtc_u_shift(atoms: ase.Atoms, energy: float, minus: bool = True) -> float:
    Z = atoms.get_atomic_numbers()
    if not (np.any(Z == 8) or np.any(Z == 9)):
        return energy 

    shift = 0.0
    for elem, dE in WTC_U_SHIFT.items():
        n = np.sum(Z == elem)
        if n > 0:
            shift += n * dE

    if minus:
        return energy - shift
    return energy + shift