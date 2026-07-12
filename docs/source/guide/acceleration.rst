Acceleration Tutorial
=====================

TACE provides two primary acceleration methods. Unless your machine is 
particularly old, you should enable both, as doing so can deliver nearly a ``5x`` 
performance improvement. All acceleration methods are controlled through 
``environment variables`` and can be used during ``train/valid/test``.

- The most computationally expensive part of equivariant models is typically 
  edge-level computation. Therefore, the core idea behind external acceleration
  libraries ``oeq`` and ``cueq`` is to avoid computing and storing all 
  edge-level tensors simultaneously.  
  
- You can also use PyTorch compilation techniques to accelerate the entire model. 
  This can provide a 2-3x speedup. Currently, compilation supports predictions 
  only for ``energy, forces, stress, and virials``. Support for additional 
  physical quantities will be added gradually in future releases.


External kernel
---------------

The following table summarizes the available external kernel for TACE:

+--------------------+-------------+
| Backend            | Support ?   |
+====================+=============+
.. | cartnn             | ✔           |
.. +--------------------+-------------+
.. | e3nn               | ✔           |
+--------------------+-------------+
| equitroch          | ✔           |
+--------------------+-------------+
| openequivariance   | ✔           |
+--------------------+-------------+
| cuequivariance     | ✔           | 
+--------------------+-------------+
| compile            | ✔           | 
+--------------------+-------------+

Set the corresponding environment variable to 1 to enable it, and 0 to disable it.

   .. code-block:: bash

      export TACE_USE_OEQ=1       # openequivariance
      export TACE_USE_CUE=1       # cuequivariance
      export TACE_USE_EQT=1       # equitroch      

      export TACE_USE_COMPILE=1   # compile whole model

.. note::

  - If your model is a parameter dictionary, the relevant environment variables 
    can be used to replace or modify the model at runtime. However, once the model 
    has been serialized, it can no longer be modified through environment variables. 
    Therefore, you should set the required environment variables before exporting 
    the model, for example, before exporting a model for use with LAMMPS.

  - ``eqt`` is primarily intended to accelerate models with ``correlation > 2`` 
    and is generally not recommended for typical use cases. 


.. 1. ``tace.models.e3nnTACE``
   
..    Support ``oeq``, ``cue`` and ``eqt``.
   
.. 2. ``tace.models.cartTACE``

..    This model does not support acceleration.

..    The Cartesian version of TACE is currently being refactored for better future compatibility.

..    If you need to use the Cartesian version at this time, please refer to version v0.1.0:

..    https://github.com/xvzemin/tace/tree/v0.1.0