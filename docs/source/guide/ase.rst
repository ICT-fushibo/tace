ASE Calculator Tutorial
=======================

This tutorial demonstrates how to use a TACE model as a calculator within ASE (Atomic Simulation Environment).

ASE Calculator documentation: `ASE Calculator <https://wiki.fysik.dtu.dk/ase/ase/calculators/calculator.html>`_

For optimization, molecular dynamics, and other examples, see the
`TACE ASE examples <https://github.com/xvzemin/tace/tree/main/example/ase>`_.

.. code-block:: python

    from ase import units
    from ase.io import read
    from tace.interface.ase import TACEAseCalc, add_dispersion

    device = "cuda"           # Use "cpu" when CUDA is unavailable
    dtype = "float32"         # "float32" or "float64"
    model_path = "model.pt"   # .pt, .pth, .ckpt, or compatible .pt2
    fidelity_idx = 0
    atoms = read("structure.xyz", index=0)

    dispersion = False

    calc = TACEAseCalc(
        model_path,
        device=device,
        dtype=dtype,
        fidelity_idx=fidelity_idx,
    )
    if dispersion:  # Requires: pip install torch-dftd
        calc = add_dispersion(
            base_calc=calc,
            damping="bj",  # choices: ["zero", "bj", "zerom", "bjm"]
            dispersion_xc="pbe",
            dispersion_cutoff=40.0 * units.Bohr,
        )
    atoms.calc = calc

.. autoclass:: tace.interface.ase.calculator.TACEAseCalc
   :no-members:
   :show-inheritance:
