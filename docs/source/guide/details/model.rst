model
=====

.. note::
    You can view detailed descriptions of some parameters here. We also provide detailed explanations in the example 
    input files, code, so you can use these resources together for a better understanding.

Example
-------

.. code-block:: yaml

  config:
    _target_: tace.models.TACEV1 # only TACEV1 are avaliable now
    wrapper:
        _target_: tace.models.WrapModelV1 # only WrapModelV1 are avaliable now

    cutoff: 5.0 # (float), recommend 6.0
    max_neighbors: null # (null, int) if you want to use tace in lammps, do not use max_neighbors
    atomic_numbers: null # (null, list), null = read from dataset.
    atomic_energies: # (list[dict[int, float]], null) null = auto caculate, for each computational level
        - {1: -13.587222780835477, 6: -1029.4889999855063, 7: -1484.9814568572233, 8: -2041.9816003861047}
    num_levels: 1
    use_multi_head: false # (bool)
    use_multi_fidelity: false # (bool)
    Lmax: 2 # (int, list), Truncation for node, 2 is recommended. If >2, recommend set `l1l2 = <=`
    lmax: 3 # (int, list), Truncation for edge, 3 is recommended, 2 for higher speed. If >3, recommend set `l1l2 = <=`
    num_layers: 2 # (int), 2 for correlation >=2, can be higher when correlation=2
    bias: true # (bool), false is more safe, but true sometimes give better results
    num_channel: 64 # (int), recommended 48-64, 64's accuracy is enough to achieve SOTA
    num_channel_hidden: ${model.config.num_channel} # (int), equal num_channel is the best

    radial_basis:
        radial_basis: j0   # [j0, jn, n_j0]
        num_radial_basis: 8 # (int) 8-10 is recommended 
        distance_transform: null # (null, str) choices: [null, Agnesi, Soft], this plug-in from mace, if you do not know what it means, never use
        polynomial_cutoff: 5 # (int, float)
        order: 0 # (int), specify order for jn, for j0, it is negligible, for jn, you should use float64
        trainable: false # (bool)
        apply_cutoff: false # # (bool) true: cutoff was use in before radial_mlp, false: after radial_mlp

    angular_basis:
        traceless: true # (bool), always set this to true
        norm: true      # (bool), always set this to true

    radial_mlp:
        hidden: [64, 64, 64, 1024] # (list(int), list(list(int)))
        act: 'silu'   # (str, null)
        bias: false   # (bool)

    inter:
        l1l2: null # (null | str | list), choices: [null, <=], restriction for each layer
        conv_weights: [edge_ij] # list, subset of [edge_ij, node_j, node_i]
        normalizer: avg_num_neighbors  # str, choices: [avg_num_neighbors, density_v1]
        nonlinearity:
        type: null # (null | str), choices: [null, norm, gated]
        gate: 'silu' # (null | str)
        kernel: scatter # (str), choices: [scatter, torch_fusion] 
        sc: # self connection
          use_first_sc: false    # (bool), false is recommended
          from: current_message  # (str), choices: [current_message, last_product]
        use_resnet: false        # (bool), recommend true when use nonlinearity gate, else false

    prod:
        l1l2: null  # (null | str | list), restriction when produce combination for each layer, choices: [null, <=]
        l3l1: null  # (null | str | list), restriction when produce combination for each layer, choices: [null, <=, ==]
        correlation: 3 # (int | list) body-order, 3 is recommended
        element_aware: true # (bool) always set this to true
        coupled_channel: true # (bool) always set this to true
    
    readout_emlp:
        hidden: [16] # (list(int)) always set this to [16]
        act: silu # (null | str) for l=0, always set this to silu
        gate: silu # (null | str) for l>0
        bias: false # (bool) only useful for l=0, recommended false
        use_all_layer: true # (bool) recommended true, if true, evergy layer has readout, else, only last layer has
        enable_uie_readout: false # (bool), whether use universal invariant embedding readout
        
    scale_shift:
        scale_type: rms_forces # (null | str) add by_element can scale for each element
        shift_type: mean_delta_energy_per_atom # (null | str)  add by_element can shift for each element,  for energy only, set ``null`` or ``std_energy``
        scale_trainable: false # (bool) 
        shift_trainable: false # (bool) 
        # # Manually modify the scale and shift parameters per element using the automatically generated statistics yaml file and ignore this parameter
        scale_dict: auto 
        shift_dict: auto

    short_range:
        zbl: # metal units, for molecular dynamics, true is recommended
        enable: false
        trainable: false # recommended always false

    long_range:
        les: # for introduction to the arguments, see the official repo: https://github.com/ChengUCB/les
        enable: false
        les_arguments:
            n_layers: 3
            n_hidden': [32, 16]
            add_linear_nn: true
            output_scaling_factor: 0.1
            sigma: 1.0
            dl: 2.0
            remove_mean: true
            epsilon_factor: 1.0
            use_atomwise: false
            compute_bec: false
            bec_output_index: null

    conservation: # only one of enable_* can be true for each property
        charges:
        method: lagrangian # (str) choices: [lagrangian, uniform_distribution] 

    universal_embedding:
        invariant:
        level:
            enable: false
            num_embeddings: 5 # (int) the number of different DFT computational levels
        
        spin_multiplicity:
            enable: false
            num_embeddings: 2 # (int) 
        
        charges:
            enable: false
            act: 'silu'   # (str, null)

        total_charge:
            enable: false
            act: 'silu'   # (str, null)

        collinear_magmoms:
            enable: false
            act: 'silu'   # (str, null)
        
        temperature:
            enable: false
            act: 'silu'   # (str, null)

        electron_temperature:
            enable: false
            act: 'silu'   # (str, null)

        equivariant:
        noncollinear_magmoms:
            enable: false
            normalizer: 1.0 # (float)

        electric_field:
            enable: false
            normalizer: 1.0 # (float)

        magnetic_field:
            enable: false
            normalizer: 1.0 # (float)

