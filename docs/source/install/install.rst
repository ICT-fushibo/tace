.. _installation:

Installation
============

Requirements
------------

TACE requires Python 3.9 or newer and PyTorch 2.4 through 2.13. AOTInductor
export additionally requires PyTorch 2.11 or newer. We recommend installing
TACE in a clean environment:

.. code-block:: bash

   micromamba create -n tace python=3.13 -y
   micromamba activate tace

Install TACE
------------

Install the latest release from PyPI:

.. code-block:: bash

   pip install tace

To install the current source tree instead:

.. code-block:: bash

   git clone https://github.com/xvzemin/tace.git
   cd tace
   pip install .

The core installation uses the standard e3nn implementation. Acceleration
libraries and simulation interfaces are optional and can be installed
independently as described below. When working from a source checkout, replace
``tace[extra]`` with ``.[extra]`` in the commands.

OpenEquivariance (OEQ)
----------------------

OEQ provides optimized CUDA or HIP equivariant kernels:

.. code-block:: bash

   pip install "tace[oeq]"

Enable it before constructing or loading a configurable model:

.. code-block:: bash

   export TACE_USE_OEQ=1

cuEquivariance (CUE)
--------------------

Install the package matching the CUDA major version used by PyTorch. CUDA 12
and CUDA 13 use different kernel packages:

.. code-block:: bash

   # CUDA 12
   pip install "tace[cueq12]"

   # CUDA 13
   pip install "tace[cueq13]"

Check ``torch.version.cuda`` if the correct CUDA variant is unclear, then
enable the backend with:

.. code-block:: bash

   python -c "import torch; print(torch.version.cuda)"
   export TACE_USE_CUE=1

EquiTorch (EQT)
---------------

The EQT implementation used by TACE is bundled with TACE, so ordinary EQT
usage does not require installing a separate EquiTorch package:

.. code-block:: bash

   export TACE_USE_EQT=1

The sparse higher-order product path for models with ``correlation > 2`` may
also require ``torch-scatter``. Install a wheel matching the exact PyTorch and
CUDA versions in the environment. For example, for PyTorch 2.11 and CUDA 13.0:

.. code-block:: bash

   pip install torch-scatter \
     -f https://data.pyg.org/whl/torch-2.11.0+cu130.html

Use the `PyTorch Geometric installation guide
<https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_
to select a different PyTorch or CUDA wheel.

TACE Triton Operators
---------------------

TACE's fused Triton operators use the Triton installation supplied by a
compatible CUDA-enabled PyTorch build; no additional TACE extra is required:

.. code-block:: bash

   export TACE_USE_TRITON=1

TorchSim
--------

Install the optional TorchSim interface with:

.. code-block:: bash

   pip install "tace[torchsim]"

TACE requires ``torch-sim-atomistic>=0.6.1`` and does not impose an upper
version bound. Compatibility with newer releases follows the upstream TorchSim
API; when that API changes, use mutually compatible TACE and TorchSim releases.
See the :doc:`../guide/torchSim` tutorial for calculator usage.

Acceleration Selection
----------------------

OEQ, CUE, and EQT are alternative equivariant kernel backends; enable only one
of them at a time. The TACE Triton operators and PyTorch compilation are
separate acceleration layers. See the :ref:`acceleration-tutorial` for backend
selection, Python interfaces, compilation, and AOTI export.
