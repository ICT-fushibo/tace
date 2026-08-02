trainer
=======

For a complete list of parameters, see the `Lightning Trainer documentation
<https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.

Example
-------

.. code-block:: yaml

  trainer:
    # As long as your system has NVIDIA GPUs, the following configuration will automatically detect and use all available GPUs for training.
    # If you wish to control which GPUs are visible, you can do so by setting the environment variable like export CUDA_VISIBLE_DEVICES=0,2. 
    _target_: lightning.Trainer
    num_nodes: 1   
    accelerator: auto
    devices: auto        # auto or list
    max_time: '90:00:00:00' # 90 days
    max_epochs: 20000
    min_epochs: 1        # null or int > 1
    precision: 32        # only 32 or 64 are allowed, not allow bf16, fp16 ....
    strategy: auto # For single-GPU/multi-node training, use auto
    # strategy: # Recommended for single-node training. 
    #   _target_: tace.utils.strategy.SimpleDDPStrategy                                                  
    gradient_clip_val: 10.0 
    enable_progress_bar: true # always set this to true to show progress_bar
    log_every_n_steps: 1000

    # Generally, no modification is required
    enable_model_summary: true
    enable_checkpointing: true 
    check_val_every_n_epoch: 1
    detect_anomaly: false
    inference_mode: false
    deterministic: false  
    # benchmark: true  
