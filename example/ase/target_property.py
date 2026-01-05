################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch

from ase.io import read, write

from tace.interface.ase import TACEAseCalc, add_dispersion

dtype = 'float32'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
level = 0  # first fidelity
model = "ckpt/average_model.pt"  # Your Model
atoms = read('../unrelaxed.xyz', index=0)


# The training property of TACE-v1-OAM-M.pt are ['energy', 'forces', 'stress'],
# but we can also predict properties such as hessians, atomic_stresses

target_property = ["energy", "forces", "stress", "atomic_stresses", "hessians"]
calc = TACEAseCalc(
    model=model,
    dtype=dtype,
    device=device,
    level=level,
    target_property=target_property,
)
atoms.calc = calc
energy = atoms.get_potential_energy()
calc_results = atoms.calc.results
print(calc_results.keys())
print(energy)
print(calc_results['energy'])
print(calc_results['forces'].shape)
print(calc_results['stress'].shape)
print(calc_results['stresses'].shape)
print(calc_results['hessians'].shape)





