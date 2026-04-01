LAMMPS-ML-IAP Tutorial
======================

This tutorial demonstrates how to use a TACE model in LAMMPS.

An example can be found at:

https://github.com/xvzemin/tace/tree/main/example/lammps


We provide an interface for LAMMPS-MLIAP that supports multi-node and multi-GPU parallelization. 
To use any MLIP (such as TACE, NequIP, MACE, Allegro, etc.) in LAMMPS, there are two steps:

Step 1: Install LAMMPS
----------------------

In principle, since the MLIAP interface is provided for Python, compiling LAMMPS should be straightforward on most modern systems. 
As long as your system is not very old, the installation process is generally easy.

.. code-block:: bash

   conda activate tace  # activate your tace environment

   git clone https://github.com/lammps/lammps.git --depth 1
   cd lammps
   mkdir build-mliap
   cd build-mliap
   cp ../cmake/presets/kokkos-cuda.cmake .

   # conda install -c conda-forge gcc_linux-64 gxx_linux-64 cmake

   # cuda 12
   pip install cython cupy-cuda12x

   # # cuda 13
   # pip install cython cupy-cuda13x

   # Use your own GPU arch
   # -D Kokkos_ARCH_Hopper70=ON \ # A800
   # -D Kokkos_ARCH_ADA89=ON \    # 4090
   # -D Kokkos_ARCH_HSX90=ON \    # H100
   cmake -C kokkos-cuda.cmake \
   -D CMAKE_BUILD_TYPE=Release \
   -D CMAKE_INSTALL_PREFIX=$(pwd) \
   -D BUILD_MPI=ON \
   -D PKG_ML-IAP=ON \
   -D PKG_ML-SNAP=ON \
   -D MLIAP_ENABLE_PYTHON=ON \
   -D PKG_PYTHON=ON \
   -D BUILD_SHARED_LIBS=ON \
   -D Kokkos_ARCH_Hopper70=ON \
   ../cmake

   make -j 8
   make install-python

   # After compilation, you get the LAMMPS executable file
   # You can either put it into your PATH, or directly use it
   export PATH=$PATH:$(pwd)
   cp lmp ~



Step 2: Train the Model and Export
----------------------------------

Once you have trained your model, you will obtain a checkpoint file, for example `.ckpt`.  
To convert this checkpoint into a format readable by LAMMPS, execute the following command:

.. code-block:: bash

   # Use your own model, or tace foundational model
   cp ~/.cache/tace/TACE-v1-OMat24-M.pt .

   # tace-export -h 
   tace-export -m TACE-v1-OMat24-M.pt --backend lammps 

After running `tace-export`, you will get a file with extension `*-lammps_mliap.pt`.  
This is the exported model file that you will use in LAMMPS.  

.. code-block:: yaml

   # an example lammps input file, we do not provide input structure, modify by yourself
   # To use TACE in lammps, you only need to modify the  ``pair_style`` and replace the model file with your own.
   # Other parameters follow the same usage as in LAMMPS.

   # MoS2 test system
   units           metal
   atom_style      atomic
   processors      * * 1
   boundary        p p p
   newton          on

   box             tilt large
   read_data       in-mzl.lammps-data
   change_box      all triclinic

   mass 1 1.00794  # H
   mass 2 12.0107  # C
   mass 3 14.0067  # N

   pair_style      mliap unified TACE-v1-OMat24-M.pt-lammps_mliap.pt 0
   pair_coeff      * * H C N

   neighbor        2.0 bin
   thermo_style    custom step pe ke etotal temp press vol fmax fnorm
   thermo          10
   dump            1 all custom 100 traj.dump id type element x y z fx fy fz vx vy vz
   dump_modify     1 element H C N

   min_style       cg
   minimize        1e-4 1e-4 100 100

   velocity        all create 348 5463576
   fix             1 all nvt temp 348 348 0.1
   timestep        0.001

   run             2000000

.. code-block:: bash
   
   # When your input files are ready, start the simulation with the following command
   lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lmp

   
.. autoclass:: tace.interface.lammps.mliap.TACELammpsCalc
   :no-members:
   :show-inheritance:
