USPEX Tutorial
==============

This tutorial demonstrates how to use a TACE model within USPEX.

Since USPEX currently does not provide a direct external machine-learning
potential interface, we integrate TACE through the following workflow::

    USPEX -> LAMMPS-MLIAP -> TACE

At present (as of May 8, 2026), we restrict support to USPEX-v10.5, since
USPEX-2025 does not yet provide LAMMPS support.

USPEX can be obtained from the official website:

https://uspex-team.org/en

To use this workflow, both USPEX and LAMMPS must be installed. Currently,
only NVIDIA GPUs (not allow cpu) are supported.

An example can be found at:

https://github.com/xvzemin/tace/tree/main/example/uspex