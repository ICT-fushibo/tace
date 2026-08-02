optimizer
=========

Any optimizer supported by PyTorch or third party can be used.

.. note::

   The safest choice is to use Adam or AdamW. Although newer optimizers such as
   Muon and SOAP may improve convergence speed and in-domain accuracy, they can
   potentially reduce the model's extrapolation ability.

Example
-------

.. code-block:: yaml

  optimizer:
    _target_: torch.optim.AdamW
    lr: 1e-3
    weight_decay: 1e-8
  
    # _target_: torch.optim.Adam
    # lr: 1e-3
    # weight_decay: 1e-8
    # amsgrad: true

    # # pip install pytorch-optimizer before use SOAP
    # _target_: pytorch_optimizer.SOAP
    # lr: 5e-3
    # betas: [0.95, 0.95]
    # weight_decay: 1e-3
    # precondition_frequency: 10
    # max_precondition_dim: 256
    # merge_dims: false
    # precondition_1d: false
    # normalize_gradient: false

    # # You must `export TACE_USE_MATRIX_WEIGHT=1` before use this optimizer
    # _target_: tace.utils.optimizer.HybridMuonOptimizer
    # lr: 1e-3
    # weight_decay: 1e-8
    # adam_betas: [0.9, 0.98]
