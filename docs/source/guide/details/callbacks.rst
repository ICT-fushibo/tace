callbacks
=========

A callback refers to a set of functions that are automatically invoked after the 
completion of each training epoch. At a minimum, you need at least one callback for saving the model.
If you have any special requirements, or if you want to use any built-in features provided by Lightning, 
you can add them here as callbacks.


Example
-------

.. code-block:: yaml

    callbacks:
    early_stopping:
        _target_: lightning.pytorch.callbacks.EarlyStopping
        verbose: true
        log_rank_zero_only: true
        monitor: ${synth_metric.monitor_metric_name}
        # min_delta: 1e-5 
        patience: 50

    ema: # ema is always recommended
        _target_: tace.utils.callbacks.EMACallback
        decay: 0.999 # 0.99 - 0.999
        use_num_updates: true

    checkpoint_epoch: # at leas one checkpoint is required
        _target_: lightning.pytorch.callbacks.ModelCheckpoint
        dirpath: checkpoints_epoch
        filename: TACE-{${misc.project_name}-{epoch}-{step}-{${synth_metric.monitor_metric_name}:.4f}
        # monitor: ${synth_metric.monitor_metric_name}
        save_top_k: -1
        save_last: true
        every_n_epochs: 10
        # mode: min
        save_weights_only: false
        auto_insert_metric_name: false
        verbose: false

    # # If you also need to save checkpoints based on training steps, you can uncomment this section.
    # checkpoint_step: 
    #   _target_: lightning.pytorch.callbacks.ModelCheckpoint
    #   dirpath: checkpoints_step
    #   filename: TACE-{${misc.project_name}-{epoch}-{step}
    #   save_top_k: -1
    #   save_last: false
    #   every_n_train_steps: 10000
    #   save_weights_only: false
    #   auto_insert_metric_name: false
    #   verbose: false
