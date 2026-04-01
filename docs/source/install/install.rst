Installation
============

.. note::
    It is recommended to use ``mamba/conda`` to create a clean environment.

    .. code-block:: bash
        
        # python < 3.14
        micromamba create -n tace python=3.13.11 -y 
        micromamba activate tace 

    .. If your system is too old and the GCC version is outdated, installing the following
    .. packages first and then installing TACE may help save you from trouble.

    .. .. code-block:: bash

        micromamba install -c conda-forge compilers openblas cmake pkg-config hdf5 h5py
        

You can install the package in ways as described below.

Install from Source (recommended)
---------------------------------

.. code-block:: bash

    git clone https://github.com/xvzemin/tace.git
    cd tace
    pip install .

    # Optional packages for accelerating computations, highly recommended.
    # For equitroch, we maintain a local version.

    pip install openequivariance

    # cuda12
    cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12

    # cuda13
    cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12

Install via pip (not recommended)
---------------------------------

.. code-block:: bash

    pip install tace 

    # Optional packages for accelerating computations, highly recommended.
    # For equitroch, we maintain a local version.

    pip install openequivariance

    # cuda12
    cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12

    # cuda13
    cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu12

