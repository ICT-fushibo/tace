################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from pathlib import Path

import torch
from ase.io import read, write
from sella import Sella

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

TS = list(Path("O5").rglob("TS*")) # Use your own approximate transition state structure

for f in TS:
    f = Path(str(f))
    atoms = read(f, 0)
    atoms.calc = calc   
    print(f">>> Relaxing {f}")
    ts_opt = Sella(
        atoms,
        trajectory=str(f.with_suffix(".traj")),
        logfile='-',
        # eta=1e-4,        # Finite difference step size
        # gamma=0.4,       # Convergence criterion for iterative diagonalization
        # delta0=1.3e-3,   # Initial trust radius
        # rho_inc=1.035,   # Threshold for increasing trust radius
        # rho_dec=5.0,     # Threshold for decreasing trust radius
        # sigma_inc=1.15,  # Trust radius increase factor
        # sigma_dec=0.65,  # Trust radius decrease factor
        # order=1,
    )
    ts_opt.run(fmax=0.01)
