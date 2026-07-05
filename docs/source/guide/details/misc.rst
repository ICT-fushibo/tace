misc
====

The `misc` section contains miscellaneous configuration options for training and execution, such as tf32 training.

Example
-------

.. code-block:: yaml
  
  misc:
    project_name: Example
    global_seed: 42 
    device: cuda # cpu or cuda
    allow_tf32: true
    ignore_warning: true 
    log_level: debug # info or debug
    env: # You can specify here the environment variables that should always be used.
      WANDB_MODE: offline # For the convenience of Chinese users
    # Generally, there is no need to enable it. If training is unstable, prioritize using Layer Norm.
    LossSkipController:
      enable: false 
      manual_threshold: 1e6
      start_step: 1000
      ema_window: 1000
      multiplier: 1e6
      skip_nan: true
      skip_large: true

