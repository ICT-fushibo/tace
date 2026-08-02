dataset
=======

This field is used to store information about the sources of the train/valid/test set.
It also controls whether the constructed graphs are saved locally, whether they are 
loaded from specified files, and which keyscare used to read training labels from the input data.


.. note::
  
   - The priority order is:  ``no_valid_set`` > ``valid_file`` > ``valid_from_index`` > ``valid_ratio``.


Example
-------

.. code-block:: yaml

  dataset:
    type: ase # ase or ase-db
    split_seed: ${misc.global_seed} # this random seed is useful if auto split
    train_file: ../data/BaTiO3.xyz
    valid_file: null                
    # test_files: # (null, str, list) recoomend to use tace-eval --test 1 to print test errors to avoild possible bugs
    #   - dataset/test_300K.xyz
    #   - dataset/test_600K.xyz
    #   - dataset/test_1200K.xyz
    #   - dataset/test_dih.xyz
    valid_ratio: 0.1 # auto split from train file if valid file is null, The priority order is:  no_valid_set > valid_file > valid_from_index > ``valid_ratio.
    valid_from_index: false # split train and val from train.index and valid.index in current directory
    no_valid_set: false # The prerequisite for enabling this is that you are using a learning rate scheduler that does not depend on the validation set.
    neighborlist_backend: matscipy # [ase, matscipy, vesin, alchemiops] recommend matscipy
    storage_mode: memory # [memory, lmdb], if your dataset is large (> 100w), the recommended approach is to use lmdb, as this avoids repeatedly constructing the graph.  
    shard_dirs: # if lmdb model, specify a list of path where you save you graph
      - graphCache
    # If using LMDB mode, you must allocate a reasonable pre-storage size.  
    # Generally, if the number of neighboring atoms is around 60, about 200 KB per graph is sufficient.
    shard_size: 100000 # number of graphs stored per file in LMDB mode, should be large, here is just an example, if small, it will be slow and may be error
    cache_size: 1024 # cache for faster load data from dataloader
    avg_graph_size_in_KB: 200 # in KB, the total disk usage equals the size of this multiplied by the total number of your structures.
    lmdb_wait_timeout: 86400 # in seconds; when training with multi-GPU, the maximum waiting time.
    force_dtype: null # 32 or 64, if your data grapg saved in lmdb, but original float64, you can force to float32 for convenience
    keys: # all default key name is the property name
      fidelity_idx_key: fidelity_idx
      energy_key: energy
      forces_key: forces
      stress_key: stress
      virials_key: virials
      charges_key: charges
      total_charge_key: total_charge
      direct_forces_key: forces
      direct_stress_key: stress
      direct_virials_key: virials
      polarization_key: polarization
      direct_dipole_key: dipole
      conservative_dipole_key: dipole
      direct_polarizability_key: polarizability
      conservative_polarizability_key: polarizability
      born_effective_charges_key: born_effective_charges
      electric_field_key: electric_field
      magnetic_field_key: magnetic_field
      initial_collinear_magmoms_key: initial_collinear_magmoms
      initial_noncollinear_magmoms_key: initial_noncollinear_magmoms
      final_collinear_magmoms_key: final_collinear_magmoms
      final_noncollinear_magmoms_key: final_noncollinear_magmoms
      total_collinear_magmoms_key: total_collinear_magmoms
      total_noncollinear_magmoms_key: total_noncollinear_magmoms
      collinear_magnetic_forces_key: collinear_magnetic_forces
      noncollinear_magnetic_forces_key: noncollinear_magnetic_forces

    train_dataloader:
      _target_: torch_geometric.loader.DataLoader
      batch_size: 8
      shuffle: true # always true
      drop_last: false
      num_workers: 4 # If the batch size is large, you need to set up multiple workers

    valid_dataloader:
      _target_: torch_geometric.loader.DataLoader
      batch_size: 4
      shuffle: false
      drop_last: false
      num_workers: 2 # If the batch size is large, you need to set up multiple workers

    test_dataloader: ${dataset.valid_dataloader}
