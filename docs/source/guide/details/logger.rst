logger
======

The recommended loggers are ``lightning``. 
By default, wandb is configured to store logs locally. 
If you prefer to use wandb online, manually modify misc.env.WANDB_MODE in ``tace.yaml``.


.. note::

   We do not guarantee that the parameters recorded by Weights & Biases (wandb)
   are always correct. We recommend directly referring to the errors printed in
   the standard output. At the end of each epoch, all error metrics are
   automatically printed.


Example
-------
.. code-block:: yaml

    # Comment out the section below to use Lightning, or uncomment it to use wandb.

    # You can specify the logger here, for example, the built-in logger provided 
    # by Lightning or a third-party logger such as WandB. 
    logger:
    _target_: pytorch_lightning.loggers.WandbLogger
    project: ${misc.project_name} 
    name: ${misc.project_name} 
    # entity: xuzemin-nanjing-university 
    log_model: final  # all, final
    # tags: ["baseline", "v0.0.1"]
    # group: "experiments"
    save_dir: wandb_logs  

