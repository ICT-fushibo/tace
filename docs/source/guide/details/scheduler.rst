scheduler
=========

Types of Learning Rate Schedulers
---------------------------------

There are generally two types of learning rate schedulers:

1. **Validation-based schedulers**  
   Adjust the learning rate based on performance on the validation set.  
   Example: ``torch.optim.lr_scheduler.ReduceLROnPlateau``

2. **Fixed-step schedulers**  
   Reduce the learning rate in a predefined manner.  

In this document, we take ``torch.optim.lr_scheduler.ReduceLROnPlateau`` as an example.  
For other schedulers, please check the `official PyTorch documentation <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.

Custom Learning Rate Schedulers
-------------------------------

In addition to the official PyTorch learning rate schedulers,  
you can also use custom schedulers implemented in the codebase.  

If you define your own scheduler in ``tace.utils.lr_scheduler``,  
you only need to modify the ``_target_`` field accordingly.

Example
-------

.. code-block:: yaml

   scheduler:
   _target_: torch.optim.lr_scheduler.ReduceLROnPlateau # validaton-based
   mode: min
   factor: 0.5
   # min_lr: 1e-6
   patience: 25
   # threshold: 1e-4 
   extra:
      monitor: ${synth_metric.monitor_metric_name}
      interval: epoch
      frequency: 1

   # _target_: tace.utils.lr_scheduler.CosineAnnealingWarmupRestarts
   # first_cycle_steps: 400000 # total step, one batch = one step
   # cycle_mult: 1.0 # restart factor
   # max_lr: 2e-4 
   # min_lr: 2e-6
   # warmup_steps: ${floor:${mul:${scheduler.first_cycle_steps}, 0.05}} # 5 % first step (total here)
   # gamma: 1.0 # decay factor
   # last_epoch: -1
   # extra:
   #   interval: step
   #   frequency: 1

   # _target_: tace.utils.lr_scheduler.WarmupStableDecay
   # num_warmup_steps: 19691
   # num_stable_steps: 177219
   # num_decay_steps: 393820
   # min_lr_ratio: 0.001
   # num_cycles: 0.5
   # cooldown_type: cosine # [cosine, 1-sqrt, linear, 1-square]
   # extra:
   #   monitor: ${synth_metric.monitor_metric_name}
   #   interval: step
   #   frequency: 1


