################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from pathlib import Path

import torch
from ase.io import read, write
from sella import IRC

from tace.interface.ase import TACEAseCalc
from tace.foundations import tace_foundations

model = tace_foundations["TACE-v1-LES-REICO-5-PdAgCHO.pt"]
dtype = "float32"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
calc = TACEAseCalc(
    model=model,
    dtype=dtype,
    device=device,
    level=0,
)

TS = list(Path("O5").rglob("*xyz")) # Use your own approximate transition state structure


fmax = 0.01
steps = 1000

for file in TS:
    file = Path(str(file))
    print(f">>> Running IRC {file}")
    atoms = read(file, -1)   

    print(">>> Running IRC (forward direction)")
    atoms_fwd = atoms.copy()
    atoms_fwd.calc = calc
    irc_fwd = IRC(
        atoms_fwd,
        logfile="-",
        trajectory=str(file.with_suffix(".fwd.traj")),
        dx=0.1,    # default 0.1, unit Angstrom * sqrt(amu), larger, faster, less true traj
        eta=1e-4,  # default 1e-4
        gamma=0.1, # default 0.1
        # gamma=0.4, # default 0.1
        keep_going=False, 
    )
    irc_fwd.run(fmax=fmax, steps=steps, direction='forward')


    print(">>> Running IRC (backward direction)")
    atoms_bwd = atoms.copy()
    atoms_bwd.calc = calc
    irc_bwd = IRC(
        atoms_bwd,
        logfile="-",
        trajectory=str(file.with_suffix(".bwd.traj")),
        dx=0.1,    # default 0.1, unit Angstrom * sqrt(amu), larger, faster, less true traj
        eta=1e-4,  # default 1e-4
        gamma=0.1, # default 0.1
        # gamma=0.4, # default 0.1
        keep_going=False, 
    )
    irc_bwd.run(fmax=fmax, steps=steps, direction='reverse')

