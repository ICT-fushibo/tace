ASE Calculator Tutorial
=======================

This tutorial demonstrates how to use a TACE model as a calculator within ASE (Atomic Simulation Environment).

ASE Calculator documentation: `ASE Calculator <https://wiki.fysik.dtu.dk/ase/ase/calculators/calculator.html>`_

For detailed usage and scripts (e.g., ``opt``, and other scripts), see  
`https://github.com/xvzemin/tace/tree/main/example/ase <https://github.com/xvzemin/tace/tree/main/example/ase>`_

.. code-block:: python

    from ase.io import read
    from tace.interface.ase import TACEAseCalc, add_dispersion

    device = 'cuda'            # Compute device, e.g., 'cpu' or 'cuda'
    dtype = 'float32'          # model dtype 'float32' or 'float64'
    MODEL_PATH = '.pt'         # Path to the model checkpoint, file ends with .pt, .pth or .ckpt
    fidelity_idx = 0  # first fidelity
    atoms = read('*.xyz', 0)   #  Any ase readable files

    dispersion = False

    calc = TACEAseCalc(
        MODEL_PATH,
        device=device,
        dtype=dtype,
        fidelity_idx = fidelity_idx,
    )
    if dispersion: # pip install torch-dftd
        calc = add_dispersion(
            base_calc=calc,
            damping= "bj",  # choices: ["zero", "bj", "zerom", "bjm"]
            dispersion_xc="pbe",
            dispersion_cutoff= 40.0 * units.Bohr,
        )
    atoms.calc = calc

.. autoclass:: tace.interface.ase.calculator.TACEAseCalc
   :no-members:
   :show-inheritance: