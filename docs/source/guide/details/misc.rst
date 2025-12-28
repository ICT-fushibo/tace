misc
====

The `misc` section contains miscellaneous configuration options for training and execution.

Example
-------

.. code-block:: yaml
  
  misc:
    project_name: omat24_large
    global_seed: 42 
    device: cuda # cpu or cuda
    allow_tf32: true
    ignore_warning: true 
    env: # You can specify here the environment variables that should always be used.
      WANDB_MODE: offline

