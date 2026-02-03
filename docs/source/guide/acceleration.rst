Acceleration Tutorial
=====================

Note
----

This feature is prepared for **Release 0.2.0**.  
If you can see this note, the acceleration functionality has not been released yet.


Backend Types
-------------

The following table summarizes the available backends for TACE:

+--------------------+-----------+
| Backend            | Support ? |
+====================+===========+
| cartnn             | ✔         |
+--------------------+-----------+
| equitroch          | ✔         |
+--------------------+-----------+
| e3nn               | ✖         |
+--------------------+-----------+
| openequivariance   | ✔         |
+--------------------+-----------+
| cueqequivariance   | ✖         |
+--------------------+-----------+


Acceleration Methods for Specific Models
----------------------------------------

1. `tace.models.eqtTACE`

   This model uses **equitroch** as its backend. The Triton kernel provides significant acceleration, 
   especially for the coupled product basis.

   However, Equitroch currently does not support edge-level operator fusion. In such cases, you can 
   replace Equitroch's TensorProduct with openequivariance by setting the environment variable:

   .. code-block:: bash
      # Supports both training and inference stage
      export TACE_USE_OEQ=1

   Note that using oeq can significantly reduce memory usage, although it may slightly reduce speed.

2. `tace.models.cartTACE`

   This model currently does not support acceleration.
