.. _acceleration-tutorial:

Acceleration Tutorial
=====================

TACE supports three complementary forms of acceleration:

* external equivariant kernels, such as OpenEquivariance and
  cuEquivariance, accelerate the most expensive edge-level operations;
* TACE Triton operators fuse selected SO2 operations and reduce intermediate
  memory use;
* PyTorch compilation accelerates a larger part of the model and can either
  run inside the current Python process or produce an AOTInductor package for
  later deployment.

The acceleration backend must be selected before the model is constructed.
The same settings can be used during training, validation, testing, and model
export, subject to the backend limitations described below.

Kernel Backends
---------------

The following kernel backends are available:

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Backend
     - Supported
     - Environment variable
   * - Equitorch
     - Yes
     - ``TACE_USE_EQT=1``
   * - OpenEquivariance
     - Yes
     - ``TACE_USE_OEQ=1``
   * - cuEquivariance
     - Yes
     - ``TACE_USE_CUE=1``
   * - TACE Triton uuSO2 scatter
     - Yes
     - ``TACE_USE_TRITON=1``

For example:

.. code-block:: bash

   export TACE_USE_OEQ=1

Only enable one equivariant kernel backend at a time. OpenEquivariance is the
recommended default for supported NVIDIA GPUs. Equitorch is mainly useful for
models with ``correlation > 2`` and is not normally required.

The acceleration environment can also be configured through one Python
interface before constructing or loading the model:

.. code-block:: python

   from tace.utils.env import enable_acceleration

   enable_acceleration(enable_oeq=True, enable_triton=True)

By default, this interface only enables the requested backends and preserves
existing environment settings. Pass ``force=True`` to explicitly write every
backend setting and disable unselected backends.

The ASE and TorchSim calculators expose the same backends as constructor
options. For example:

.. code-block:: python

   from tace.interface.ase import TACEAseCalc

   calc = TACEAseCalc(
       model="model.pt",
       device="cuda",
       enable_oeq=True,
       enable_triton=True,
   )

.. note::

   Environment variables can replace compatible modules while a checkpoint or
   state-dict package is being loaded. Once the complete Python model has been
   serialized, its modules are already fixed. Set the required acceleration
   variables before exporting a full model or a LAMMPS model.

PyTorch Compilation
-------------------

TACE provides two different compilation workflows.

.. list-table::
   :header-rows: 1
   :widths: 28 34 38

   * - Workflow
     - How to enable it
     - Intended use
   * - In-process compilation
     - ``TACE_USE_COMPILE=1``
     - Training, validation, and inference in the current Python process
   * - AOTInductor (AOTI)
     - ``tace-export-eval --backend aoti`` or
       ``tace-export-lammps --backend aoti``
     - Ahead-of-time deployment without compiling again at startup

In-process compilation caches compiled graphs in memory and does not create a
deployment artifact:

.. code-block:: bash

   export TACE_USE_COMPILE=1
   tace-train -cn tace.yaml

AOTI produces a ``.pt2`` package containing compiled native code. Loading the
package does not call ``torch.compile`` again.

.. important::

   TACE AOTInductor compilation and export require **PyTorch 2.11 or newer**.
   Earlier PyTorch versions are not supported for AOTI export.

Compilation currently supports energy and its force, stress, and virial
derivatives, together with their direct prediction variants. Models using
unsupported output properties or LES cannot be exported through the current
AOTI path.

AOTI packages contain machine-specific native code. Compile on the deployment
machine, or on a machine with a compatible operating system, PyTorch/CUDA ABI,
and GPU architecture. A package compiled for CUDA cannot be loaded on CPU.

.. _tace-export-tutorial:

TACE Export Tutorial
--------------------

Use the export command that matches the target workflow:

* ``tace-export-train`` creates an editable model package for continued
  training, fine-tuning, or transfer learning;
* ``tace-export-eval`` creates a native PyTorch inference model or an AOTI
  graph package;
* ``tace-export-lammps`` creates a LAMMPS ML-IAP model, optionally backed by
  AOTI.

The commands accept ``.ckpt``, state-dict ``.pt``/``.pth`` packages, and
serialized full models as input. Use ``-f`` to select a fidelity and ``--dtype``
to change model precision during export.

Export for Training
~~~~~~~~~~~~~~~~~~~

Use this form when the result must remain editable by TACE:

.. code-block:: bash

   tace-export-train -m model.ckpt

The default output is ``model.ckpt-state.pt``. It stores the state dictionary
and the model configuration required by ``load_tace`` and training utilities.
The command loads EMA parameters when they are available.

An explicit output, fidelity, and precision can also be selected:

.. code-block:: bash

   tace-export-train \
     -m model.ckpt \
     -o model-fidelity-1.pt \
     -f 1 \
     --dtype float32

Export for Native PyTorch Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default ``state_dict`` backend is portable and reconstructs the model from
its saved configuration:

.. code-block:: bash

   tace-export-eval -m model.ckpt --backend state_dict --device cpu

The default output is ``model.ckpt-state_dict.pt``. This is the recommended
non-compiled format for normal evaluation and Python deployment.

The ``full_model`` backend serializes the complete Python module with
``torch.save``:

.. code-block:: bash

   tace-export-eval -m model.ckpt --backend full_model --device cpu

The default output is ``model.ckpt-full_model.pt``. It is convenient, but more
tightly coupled to the TACE and PyTorch versions used during export.

Both formats are loaded through the same API:

.. code-block:: python

   from tace.lightning import load_tace

   model = load_tace("model.ckpt-state_dict.pt", device="cuda")
   model.eval()

Export Eager or AOTI for ASE and TorchSim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set external-kernel variables before export, then select the ``aoti`` backend:

.. code-block:: bash

   export TACE_USE_OEQ=1

   tace-export-eval \
     -m model.ckpt \
     --backend aoti \
     --device cuda

No sample structure is required. TACE automatically builds a synthetic
two-graph input and exports dynamic node, edge, and graph dimensions, so the
resulting package can be used with different structures and batch sizes.
``--sample`` remains available as an optional advanced override, but is not
needed for normal ASE or TorchSim deployment.

The default output is ``model.pt2``. ``tace-compile`` is an alias for
``tace-export-eval`` and accepts the same options. The equivalent short command
is:

.. code-block:: bash

   tace-compile -m model.ckpt --backend aoti --device cuda

The graph ``.pt2`` package can be loaded with ``load_tace`` and shared by native
PyTorch consumers, including the ASE and TorchSim integrations:

.. code-block:: python

   from tace.lightning import load_tace

   model = load_tace("model.pt2", device="cuda")
   outputs = model(batch)

Export Eager or AOTI for LAMMPS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The regular ML-IAP backend serializes the eager model:

.. code-block:: bash

   export TACE_USE_OEQ=1
   tace-export-lammps -m model.pt --backend mliap --device cuda

This creates ``model.pt-lammps_mliap.pt`` by default.

To compile the LAMMPS tensor graph ahead of time:

.. code-block:: bash

   export TACE_USE_OEQ=1
   tace-export-lammps \
     -m model.pt \
     --backend aoti \
     --device cuda

The AOTI backend creates two files:

* ``model.pt-lammps_aoti.pt2`` is the compiled AOTInductor package;
* ``model.pt-lammps_aoti.pt`` is the ``MLIAPUnified`` loader used by LAMMPS.

LAMMPS ML-IAP loads a pickled Python interface, so ``pair_style`` must point to
the ``.pt`` loader rather than directly to the ``.pt2`` package. The loader
contains the package bytes and loads the compiled model without recompilation:

.. code-block:: text

   pair_style mliap unified model.pt-lammps_aoti.pt 0
   pair_coeff * * H C N

Use ``--aoti-package`` to choose the package path and ``-o`` to choose the
ML-IAP loader path. TACE ML-IAP currently requires the CUDA Kokkos backend,
and multi-rank runs require CUDA-aware MPI. Native CPU inference remains
available through the ASE and TorchSim interfaces. Detailed LAMMPS setup is
covered in :doc:`lammps`.
