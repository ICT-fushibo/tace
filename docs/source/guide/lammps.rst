LAMMPS ML-IAP Tutorial
======================

TACE runs in LAMMPS through the unified Python interface of
``pair_style mliap`` and its Kokkos accelerator variant ``mliap/kk``. The
same exported TACE model can be used by one or more MPI ranks. Each rank owns
one spatial subdomain, loads one model instance, and exchanges ghost features
between interaction layers.

The example input files are available in
`example/lammps <https://github.com/xvzemin/tace/tree/main/example/lammps>`__.
The relevant upstream references are the
`LAMMPS ML-IAP documentation <https://docs.lammps.org/pair_mliap.html>`__,
`Kokkos package guide <https://docs.lammps.org/Speed_kokkos.html>`__, and
`LAMMPS command-line options <https://docs.lammps.org/Run_options.html>`__.

Requirements and limitations
----------------------------

* Build LAMMPS with ``KOKKOS``, ``ML-IAP``, ``ML-SNAP``, and ``PYTHON``.
  ``ML-SNAP`` is a build dependency of ``ML-IAP`` even though TACE does not
  use a SNAP descriptor.
* LAMMPS and TACE must use the same Python environment. That environment must
  contain TACE, PyTorch, Cython, and the CuPy package matching the CUDA major
  version.
* TACE ML-IAP currently requires the CUDA Kokkos backend. Native TACE inference
  through PyTorch, ASE, and TorchSim can run on CPU, but LAMMPS host-side
  ML-IAP ghost exchange is not currently supported.
* One MPI rank on one GPU does not require CUDA-aware MPI. Any run with more
  than one rank requires a CUDA-aware MPI implementation.
* AOTI export requires PyTorch 2.11 or newer. AOTI packages are specific to
  their deployment software stack and accelerator target, so export on a
  machine compatible with the deployment nodes.

Build LAMMPS
------------

The following example builds LAMMPS for an NVIDIA RTX 4090. Replace
``Kokkos_ARCH_ADA89`` with the architecture of the deployment GPU, such as
``Kokkos_ARCH_AMPERE80`` for A100/A800 or ``Kokkos_ARCH_HOPPER90`` for
H100/H200/H20. The complete architecture list is maintained in the
`LAMMPS Kokkos build documentation
<https://docs.lammps.org/Build_extras.html#kokkos>`__.

.. code-block:: bash

   conda activate tace
   pip install cython cupy-cuda12x

   git clone https://github.com/lammps/lammps.git
   cd lammps

   cmake -S cmake -B build-tace \
     -D CMAKE_BUILD_TYPE=Release \
     -D CMAKE_INSTALL_PREFIX="$PWD/install-tace" \
     -D CMAKE_C_COMPILER=gcc \
     -D CMAKE_CXX_COMPILER=g++ \
     -D CMAKE_CUDA_HOST_COMPILER=g++ \
     -D Python_EXECUTABLE="$(which python)" \
     -D Python3_EXECUTABLE="$(which python)" \
     -D BUILD_MPI=ON \
     -D BUILD_SHARED_LIBS=ON \
     -D PKG_KOKKOS=ON \
     -D PKG_ML-IAP=ON \
     -D PKG_ML-SNAP=ON \
     -D PKG_PYTHON=ON \
     -D MLIAP_ENABLE_PYTHON=ON \
     -D Kokkos_ENABLE_CUDA=ON \
     -D Kokkos_ENABLE_SERIAL=ON \
     -D Kokkos_ARCH_ADA89=ON

   cmake --build build-tace --parallel 8
   cmake --install build-tace

   export PATH="$PWD/install-tace/bin:$PATH"
   export LD_LIBRARY_PATH="$PWD/install-tace/lib:$LD_LIBRARY_PATH"

For multi-rank GPU runs, configure with the compiler wrappers from the
CUDA-aware MPI installation:

.. code-block:: bash

   -D MPI_C_COMPILER=/path/to/cuda-aware-mpi/bin/mpicc
   -D MPI_CXX_COMPILER=/path/to/cuda-aware-mpi/bin/mpicxx

Check both the enabled packages and the MPI linked by the resulting executable:

.. code-block:: bash

   lmp -h
   ldd "$(which lmp)" | grep mpi

The help output should list ``KOKKOS``, ``ML-IAP``, ``ML-SNAP``, and
``PYTHON``. Do not mix ``mpirun`` from one MPI installation with a LAMMPS
binary linked against another.

Export a TACE model
-------------------

The eager ML-IAP export is portable between compatible CUDA devices and does
not compile the model:

.. code-block:: bash

   tace-export-lammps \
     -m ~/.cache/tace/TACE-OAM-7M.pt \
     --backend mliap \
     --device cuda

This writes ``TACE-OAM-7M.pt-lammps_mliap.pt`` by default. Acceleration
backends such as OEQ must be selected before export:

.. code-block:: bash

   export TACE_USE_OEQ=1
   tace-export-lammps -m model.pt --backend mliap --device cuda

For AOTI deployment:

.. code-block:: bash

   export TACE_USE_OEQ=1
   tace-export-lammps -m model.pt --backend aoti --device cuda

The command creates:

* ``model.pt-lammps_aoti.pt2``: the AOTInductor package;
* ``model.pt-lammps_aoti.pt``: the serialized ML-IAP loader.

Point LAMMPS at the ``.pt`` loader. It embeds the AOTI package bytes and loads
the compiled model in every rank without recompiling. Keep the ``.pt2`` file
as the standalone build artifact, but it is not passed directly to
``pair_style``.

LAMMPS input
------------

The execution mode is selected by the launch command, so the same input file
can be shared by single-GPU and distributed runs:

.. code-block:: text

   units           metal
   atom_style      atomic
   boundary        p p p
   newton          on

   read_data       structure.lammps-data

   mass 1 1.00794
   mass 2 12.0107
   mass 3 14.0067

   pair_style      mliap unified TACE-OAM-7M.pt-lammps_mliap.pt 0
   pair_coeff      * * H C N

   neighbor        2.0 bin
   thermo_style    custom step pe ke etotal temp press vol fmax fnorm
   thermo          10

   velocity        all create 300 5463576
   fix             1 all nvt temp 300 300 0.1
   timestep        0.001
   run             10000

The final ``0`` in ``pair_style`` disables neighbor lists centered on ghost
atoms. TACE instead exchanges ghost features after each interaction layer.
Keep ``newton on``; it is required by ML-IAP. The element order in
``pair_coeff`` must match the LAMMPS atom types.

Single GPU
----------

Use one MPI rank and one visible GPU:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 \
   lmp -k on g 1 -sf kk \
     -pk kokkos newton on neigh half \
     -in in.lmp

``-k on g 1`` enables one Kokkos GPU and ``-sf kk`` selects accelerated
styles. Half neighbor lists with Newton communication are a useful starting
point for Pascal and newer NVIDIA GPUs, but should still be benchmarked for
the target system.

Single node, multiple GPUs
--------------------------

The recommended mapping is one MPI rank per physical GPU. For two GPUs:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0,1 \
   mpirun --bind-to none -np 2 \
     lmp -k on g 2 -sf kk \
       -pk kokkos newton on neigh half \
       -in in.lmp

OpenMPI defines ``OMPI_COMM_WORLD_LOCAL_RANK`` for each process, which LAMMPS
uses to assign distinct GPUs. Other supported launchers provide equivalent
local-rank variables. ``-np`` is the total number of ranks, while ``g 2`` is
the number of GPUs available on each node.

TACE has been tested with two ranks on two RTX 4090 GPUs using CUDA-aware
OpenMPI. Every rank loads its own model, so model memory usage scales with the
number of ranks.

One GPU, multiple workers
-------------------------

Multiple MPI ranks can share one GPU:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 \
   mpirun --bind-to none -np 2 \
     lmp -k on g 1 -sf kk \
       -pk kokkos newton on neigh half \
       -in in.lmp

This mode is functional with TACE, but is not the default recommendation.
Each worker creates a CUDA context and loads another model copy, increasing
memory use. The LAMMPS Kokkos guide recommends NVIDIA CUDA MPS when multiple
ranks share a GPU. It is most useful when non-Kokkos work leaves the GPU
underutilized; for small TACE systems it can be slower than one rank per GPU.

Multiple nodes
--------------

TACE uses the MPI domain decomposition and ghost communication supplied by
LAMMPS, so the interface also supports multi-node execution. The deployment
requirements are:

* the same TACE/PyTorch/LAMMPS environment on every node;
* a model path visible on every node;
* one compatible GPU architecture and software stack for an AOTI artifact;
* CUDA-aware MPI with a transport configured for the cluster network;
* one rank per GPU as the initial process mapping.

For two nodes with two GPUs per node, an OpenMPI host file could contain:

.. code-block:: text

   node01 slots=2
   node02 slots=2

Launch four ranks, two on each node:

.. code-block:: bash

   mpirun --hostfile hosts -np 4 \
     --map-by ppr:2:node --bind-to none \
     lmp -k on g 2 -sf kk \
       -pk kokkos newton on neigh half \
       -in in.lmp

With Slurm, the equivalent allocation is typically:

.. code-block:: bash

   srun --nodes=2 --ntasks-per-node=2 \
     --gpus-per-task=1 --gpu-bind=single:1 \
     lmp -k on g 2 -sf kk \
       -pk kokkos newton on neigh half \
       -in in.lmp

Exact launcher, binding, and network options depend on the cluster MPI and
scheduler. Verify rank placement with the launcher's binding-report option
before a production run. The multi-node commands above describe the supported
configuration but have not been exercised by the TACE maintainers on the
current test machine.

Independent simulations
-----------------------

For independent replicas, it is usually simpler to launch one single-rank
LAMMPS process per GPU:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 lmp -k on g 1 -sf kk -in replica-0.lmp &
   CUDA_VISIBLE_DEVICES=1 lmp -k on g 1 -sf kk -in replica-1.lmp &
   wait

This differs from multiple workers: independent processes run separate
simulations, while multiple MPI workers cooperate on one spatially decomposed
simulation.

Troubleshooting
---------------

Segmentation fault with multiple ranks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Confirm that MPI is CUDA-aware and that ``mpirun`` matches the library linked
to ``lmp``. LAMMPS documents that an MPI implementation without GPU-aware
support may produce a warning or a segmentation fault in Kokkos runs.

The general LAMMPS workaround ``-pk kokkos gpu/aware off`` is not suitable for
distributed TACE ML-IAP: TACE exchanges PyTorch CUDA tensors directly and
therefore requires device-aware communication when more than one rank is
used.

Wrong GPU assignment
~~~~~~~~~~~~~~~~~~~~

Check the local-rank variables inside each worker and make sure the number of
MPI ranks per node matches ``g Ng``. For OpenMPI:

.. code-block:: bash

   mpirun -np 2 sh -c \
     'echo rank=$OMPI_COMM_WORLD_RANK local=$OMPI_COMM_WORLD_LOCAL_RANK'

Out of memory
~~~~~~~~~~~~~

Every MPI rank loads a separate eager or AOTI model. Reduce ranks per GPU,
disable unnecessary acceleration backends, use a smaller model, or compile
for a lower precision after validating the resulting accuracy.

CPU execution
~~~~~~~~~~~~~

TACE detects a host-side ML-IAP data object and exits with a clear error before
LAMMPS enters its currently unsupported Kokkos host ghost-exchange path. Use
the ASE or TorchSim interface for native CPU inference.

.. autoclass:: tace.interface.lammps.mliap.TACELammpsCalc
   :no-members:
   :show-inheritance:
