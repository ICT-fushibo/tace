logger
======

The recommended logger is ``wandb``, and by default, we have set the storage of wandb to local. 
If this is not the effect you want, manually modify the environment variables of wandb in ``cfg.misc.env.WANDB_MODE`` to ``online``

Example
-------
.. code-block:: yaml

    logger:
    _target_: lightning.pytorch.loggers.WandbLogger
    project: ${misc.project_name} 
    name: ${misc.project_name} 
    # entity: xxxxxx-nanjing-university 
    log_model: final  # all, final
    # tags: ["baseline", "v0.0.1"]
    # group: "experiments"
    save_dir: wandb_logs  

.. note::

   We do not guarantee that the parameters recorded by Weights & Biases (wandb)
   are always correct. We recommend directly referring to the errors printed in
   the standard output. At the end of each epoch, all error metrics are
   automatically printed.
