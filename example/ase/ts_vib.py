################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from pathlib import Path

import torch
from ase.io import read, write
from ase.vibrations import Vibrations

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

for file in TS:
    file = str(file)
    atoms = read(file, -1)
    atoms.calc = None   
    atoms.calc = calc   
    print(f">>> Running Vibration Analysis: {file}")
    vib = Vibrations(
        atoms,
        name='tmp',
        delta=0.015,
        nfree=2
    )
    vib.run()
    vib.summary()
    vib.clean()