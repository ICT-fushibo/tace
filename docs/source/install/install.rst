Installation
============

.. note::
   The basic energy/force/stress are essentially bug-free, but some advanced features may still contain bugs.
   Therefore, it is recommended to always use the latest commit on GitHub.

    It is recommended to use ``conda/mamba`` to create a clean environment.

    .. code-block:: bash
        
        # python < 3.14
        conda create -n tace python=3.12.11 -y 
        conda activate tace 

    If your system is too old and the GCC version is outdated, installing the following
    packages first and then installing TACE may help save you from trouble.

    .. code-block:: bash

        # micromamba install -c conda-forge compilers openblas cmake pkg-config hdf5 h5py
        conda install -c conda-forge compilers openblas cmake pkg-config hdf5 h5py
        

You can install the package in ways as described below.

Install from Source (recommended)
---------------------------------

.. code-block:: bash

    git clone https://github.com/xvzemin/tace.git
    cd tace
    pip install .

Install via pip (not recommended)
---------------------------------

.. code-block:: bash

    pip install tace 




