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
| cuequivariance     | ✔ (in test) |
+--------------------+-------------+


Acceleration Methods for Specific Models
----------------------------------------

We control acceleration for training and inference through environment variables.

Only one acceleration engine can be enabled at a time. 
Set the corresponding environment variable to 1 to enable it, and 0 to disable it.

   .. code-block:: bash
      
      export TACE_USE_OEQ=1   # openequivariance
      export TACE_USE_CUEQ=1  # cuequivariance


1. ``tace.models.e3nnTACE``
   
   Supports both ``oeq`` and ``cueq``.

2. ``tace.models.eqtTACE``

   Supports both ``oeq`` and ``cueq``.
   
3. ``tace.models.cartTACE``

   This model does not support acceleration.