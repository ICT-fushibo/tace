loss
====

We support two major categories of loss functions, as well as some
other custom losses. For other loss functions, please refer to the source code.

Example
-------
.. code-block:: yaml

    # If the dataset is not very large and there are no particular outliers, use this
    loss:
      _target_: tace.utils.loss.NormalLoss
      loss_property: [energy, forces, stress]
      loss_function_name: 
        - mse_energy_per_atom
        - mse_forces
        - mse_stress
      loss_property_weights: [1.0, 8.0, 8.0]

    # huber loss used for outliers, used to train uMLIPs
    # loss:
    #   _target_: tace.utils.loss.OMat24sAlexMPtrjLoss
    #   loss_property: [energy, forces, stress]
    #   energy_weight: 1.0
    #   forces_weight: 8.0
    #   stress_weight: 8.0
    #   energy_huber_delta: 0.01
    #   forces_huber_delta: 0.01
    #   stress_huber_delta: 0.01

    # The performance is mediocre and cannot achieve full convergence, but it can be used as a toy loss.
    # loss:
    #   _target_: tace.utils.loss.UncertaintyLoss
    #   loss_property: [energy, forces, stress]
    #   loss_function_name: 
    #     - mse_energy_per_atom
    #     - mse_forces
    #     - mse_stress
    #   loss_property_weights: [1.0, 1.0, 1.0]

Notes
-----

- **Choice of loss function name**:  
  - For properties that are already *per-atom* quantities,  
    the ``per_atom`` suffix is not required and is not supported.

- **Choice of loss function name**:  
  - In general, mean squared error (MSE) performs better than mean absolute
    error (MAE) during training. Therefore, MAE is not implemented by default,
    but you may add it yourself if needed. For training on large datasets,
    we recommend using the Huber loss.
