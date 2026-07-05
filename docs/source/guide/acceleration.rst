Acceleration Tutorial
=====================

TACE provides two main approaches for acceleration:

- The most computationally expensive part of equivariant models is typically 
  edge-level computation. Therefore, the core idea behind external acceleration 
  libraries is to avoid computing and storing all edge-level tensors simultaneously.

- You can also use PyTorch compilation techniques to accelerate the entire model. 
  This can provide a 2-3x speedup. However, because TACE currently supports a wide range 
  of physical quantities and parts of the model code are relatively complex, 
  this feature is planned for version 0.3.0.

External kernel
---------------

The following table summarizes the available external kernel for TACE:

+--------------------+-------------+
| Backend            | Support ?   |
+====================+=============+
| cartnn             | ✔           |
+--------------------+-------------+
| e3nn               | ✔           |
+--------------------+-------------+
| equitroch          | ✔           |
+--------------------+-------------+
| openequivariance   | ✔           |
+--------------------+-------------+
| cuequivariance     | ✔           |
+--------------------+-------------+

Set the corresponding environment variable to 1 to enable it, and 0 to disable it.

   .. code-block:: bash

      export TACE_USE_OEQ=1   # openequivariance
      export TACE_USE_CUE=1   # cuequivariance
      export TACE_USE_EQT=1   # equitroch      


.. note::

   ``eqt`` is primarily intended to accelerate models with ``correlation > 2`` 
   and is generally not recommended for typical use cases. 
   The main acceleration backends are ``oeq`` and ``cueq``.

.. 1. ``tace.models.e3nnTACE``
   
..    Support ``oeq``, ``cue`` and ``eqt``.
   
.. 2. ``tace.models.cartTACE``

..    This model does not support acceleration.

..    The Cartesian version of TACE is currently being refactored for better future compatibility.

..    If you need to use the Cartesian version at this time, please refer to version v0.1.0:

..    https://github.com/xvzemin/tace/tree/v0.1.0