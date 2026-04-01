Acceleration Tutorial
=====================

Backend Types
-------------

The following table summarizes the available backends for TACE:

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
| cuequivariance     | (in test)   |
+--------------------+-------------+


Acceleration Methods for Specific Models
----------------------------------------

We control acceleration for training and inference through environment variables.

Openequivariance is used to accelerate tensor product computations at the edge level

Equitroch is used to accelerate tensor product computations at the node level.

Set the corresponding environment variable to 1 to enable it, and 0 to disable it.

   .. code-block:: bash
      
      export TACE_USE_OEQ=1   # openequivariance
      export TACE_USE_EQT=1   # equitroch      
      export TACE_USE_CUEQ=1  # cuequivariance, not support now

1. ``tace.models.e3nnTACE``
   
   Support ``oeq`` and ``eqt``.
   
3. ``tace.models.cartTACE``

   This model does not support acceleration.

   The Cartesian version of TACE is currently being refactored for better future compatibility.

   If you need to use the Cartesian version at this time, please refer to version v0.1.0:

   https://github.com/xvzemin/tace/tree/v0.1.0