################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
'''
This is an example of molecular dynamics simulation for extended systems.
'''
import torch
import torch_sim as ts
integrator_cls = {
    "nve": ts.Integrator.nve,
    "nvt_langevin": ts.Integrator.nvt_langevin,
    "nvt_nose_hoover": ts.Integrator.nvt_nose_hoover,
    "npt_langevin": ts.Integrator.npt_langevin,
    "npt_nose_hoover": ts.Integrator.npt_nose_hoover,
}
import ase
from ase.io import read


from tace.interface.torchsim import TACETorchSimCalc

# === model ===
dtype = 'float32'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

water = read('liquid-64.xyz', '0')
init_conf = water
# init_conf = water.repeat((2,2,2))

init_atomsList = [init_conf] * 5
model = "../TACE-v1-OMat24-L.pt" # Your Model
level = 0  # first fidelity
model = TACETorchSimCalc(
    model,
    level=level,
    device=device,
    dtype=dtype, 
    compute_forces=True,
    compute_stress=True,
)

# === torchSim ===
integrator = "nvt_nose_hoover"
T = 300 # in K
TIME_STEP = 0.001  # in ps
# TOTAL_STEP = 300 * 1000
TOTAL_STEP = 10000
SAVE_FREQ = 10

# === md traj ===
filenames = [f"batchMdTraj{i}.h5" for i in range(len(init_atomsList))]
prop_calculators = {
    SAVE_FREQ: {
        "potential_energy": lambda state: state.energy,
        "kinetic_energy": lambda state: ts.calc_kinetic_energy(
            momenta=state.momenta, masses=state.masses
        ),
        "forces": lambda state: state.forces,
    # can add more SAVE_FREQ
    }
}
batch_reporter = ts.TrajectoryReporter(
    filenames, 
    state_frequency=SAVE_FREQ,
    prop_calculators=prop_calculators,
)

# Automatically manage the memory of multiple Gpus to full capacity
final_state = ts.integrate(
    system=init_atomsList, 
    model=model,  
    integrator=integrator_cls[integrator], 
    n_steps=TOTAL_STEP,  
    temperature=T, # in K
    timestep=TIME_STEP,
    trajectory_reporter=batch_reporter,
    autobatcher=True,
    pbar=True,
    # external_pressure=0.0,
)

# === convert to Atoms ===
# final_atoms: list[ase.Atoms] = final_state.to_atoms()

# # === data processing === 
# final_energies_per_atom = []
# for sys_idx, filename in enumerate(filenames):
#     with ts.TorchSimTrajectory(filename) as traj:
#         final_energy = traj.get_array("potential_energy")[-1].item()
#         n_atoms = len(traj.get_atoms(-1))
#         final_energies_per_atom.append(final_energy / n_atoms)
#         print(
#             f"System {sys_idx}: {final_energy:.6f} eV, {final_energy / n_atoms:.6f} eV/atom"
#         )

