RADIAL_BASIS = {
    "radial_basis": "j0",
    "num_radial_basis": 8,
    "distance_transform": None,
    "polynomial_cutoff": 5,
    "order": 0,
    "trainable": False,
    "apply_cutoff": True,
}


ANGULAR_BASIS = {
    "type": "ictd",
    "norm": True,
}


RADIAL_MLP = {
    "hidden": [
        [64, 64, 64],
        [64, 64, 64],
    ],
    "act": "silu",
    "bias": False,
    "enable_layer_norm": False,
}


INTER = {
    "restriction": [None, None],
    "residual": False,
}


PROD = {
    "restriction": None,
    "correlation": 3,
    "element": True,
    "coupled": True,
    "add_source_target_embedding": False,
    "normalizer": {
      "type": "fixed",
      "hidden": [64],
      "act_1": 'silu',
      "act_2": 'tanh',
      "bias": False,
      "scale_shift_trainable": True,
    }

}


READOUT_EMLP = {
    "hidden": [16], 
    "act": "silu", 
    "gate": "silu",
    "bias": False,
    "use_nolinear_tensor_readout": True,
    "use_only_last_readout": False,
}


SCALE_SHIFT = {
    "scale_type": "rms_forces",
    "shift_type": "mean_delta_energy_per_atom",
    "scale_trainable": False,
    "shift_trainable": False,
    "scale_dict": "auto",
    "shift_dict": "auto",
}


SHORT_RANGE = {
    'use_zbl': False
}

LONG_RANGE = {
    'les': 
        {
            'use_les': False,
            'les_arguments': None,
        },
}


from typing import Dict, Any, List
def check_config(cfg: Dict[str, Any]):
    assert isinstance(cfg['radial_basis'], Dict), "cfg.model.config.radial_basis must be a Dict"
    assert isinstance(cfg['radial_mlp'], Dict), "cfg.model.config.radial_mlp must be a Dict"
    assert isinstance(cfg['angular_basis'], Dict), "cfg.model.config.angular_basis must be a Dict"
    assert isinstance(cfg['inter'], Dict), "cfg.model.config.inter must be a Dict"
    assert isinstance(cfg['prod'], Dict), "cfg.model.config.prod must be a Dict"
    assert isinstance(cfg['readout_emlp'], Dict), "cfg.model.config.readout_emlp must be a Dict"
    assert isinstance(cfg['scale_shift'], Dict), "cfg.model.config.scale_shift must be a Dict"
    assert cfg['short_range'] is None or isinstance(cfg['short_range'], Dict), "cfg.model.config.short_range must be a Dict or None"
    assert cfg['long_range'] is None or isinstance(cfg['long_range'], Dict), "cfg.model.config.long_range must be a Dict or None"
    assert cfg['embedding_property'] is None or isinstance(cfg['embedding_property'], List), "embedding_property must be a List or None"
    assert cfg['conservations'] is None or isinstance(cfg['conservations'], Dict), "cfg.model.config.conservations must be a Dict or None"
    assert cfg['universal_embedding'] is None or isinstance(cfg['universal_embedding'], Dict), "cfg.model.config.universal_embedding must be a Dict or None"

    # statistics
    if not isinstance(cfg['statistics'], List): 
        cfg['statistics'] = [cfg['statistics']]

    # Lmax, lmax
    # if 'max_r_1' in cfg['kwargs']:
    #     cfg['Lmax'] = cfg['kwargs']['max_r_1']
    # if 'max_r_2' in cfg['kwargs']:
    #     cfg['lmax'] = cfg['kwargs']['max_r_2']
    if isinstance(cfg['Lmax'], int):
        cfg['Lmax'] = [cfg['Lmax']] * cfg['num_layers']
    if isinstance(cfg['lmax'], int):
        cfg['lmax'] = [cfg['lmax']] * cfg['num_layers']

    # num_channel, num_channel_hidden
    # if isinstance(cfg['num_channel'], int):
    #     cfg['num_channel'] = [cfg['num_channel']] * cfg['num_layers']
    # if isinstance(cfg['num_channel_hidden'], int):
    #     cfg['num_channel_hidden'] = [cfg['num_channel_hidden']] * cfg['num_layers']

    # radial_mlp
    if isinstance(cfg['radial_mlp']['hidden'][0], list): # not safe
        pass
    else:
        cfg['radial_mlp']['hidden'] = [cfg['radial_mlp']['hidden']] * cfg['num_layers']


    # readout_emlp
    if 'readout_mlp' in cfg['kwargs']:
        cfg['readout_emlp'] = cfg['kwargs']['readout_mlp']

    # embedding property
    cfg['embedding_property'] = cfg['embedding_property'] or []

    # atomic_numbers
    cfg['atomic_numbers'] = sorted(cfg['statistics'][0]['atomic_numbers'])

    # avg_num_neighbors
    cfg['avg_num_neighbors'] = cfg['statistics'][0]['avg_num_neighbors']

    # atomic_energies TODO, check correctness when atomic_energies = None
    if "energy" in cfg['target_property']: 
        cfg['atomic_energies'] = [stats['atomic_energy'] for stats in cfg['statistics']]
    else:
        cfg['atomic_energies'] = None

    # short_range and long_range
    cfg['short_range'] = cfg['short_range'] or SHORT_RANGE
    cfg['long_range'] = cfg['long_range'] or LONG_RANGE
    cfg['use_zbl'] = cfg['short_range'].get('use_zbl', False)
    if cfg['short_range'].get('enable_zbl', False):
        cfg['use_zbl'] = True
    cfg['use_les'] = cfg['long_range'].get('les', LONG_RANGE['les']['use_les'])


    # universal_embedding
    cfg['universal_embedding'] = cfg['universal_embedding'] or {}

    # conservations
    cfg['conservations'] = cfg['conservations'] or {}

    # inter
    # if isinstance(cfg['inter'].get('restriction', None), List):
    #     cfg['inter']['l1l2'] = []
    #     for r in cfg['inter']['restriction']:
    #         cfg['inter']['l1l2'].append(r)
    if isinstance(cfg['inter']['l1l2'], str) or cfg['inter']['l1l2'] == None:
        cfg['inter']['l1l2'] = [cfg['inter']['l1l2']] * cfg['num_layers']


    cfg['inter']['ictp_lw'] = cfg['inter'].get('ictp_lw', False)
    cfg['inter']['ictc_lw'] = cfg['inter'].get('ictc_lw', False)
    cfg['inter']['ictp_hw'] = cfg['inter'].get('ictp_hw', True)
    cfg['inter']['ictc_hw'] = cfg['inter'].get('ictc_hw', True)

    # prod
    # if 'r_1_r_2' in cfg['prod'].get('restriction', {}):
    #     cfg['prod']['l1l2'] = [cfg['prod']['restriction']['r_1_r_2']] * cfg['num_layers']
    # if 'r_o_r_1' in cfg['prod'].get('restriction', {}):
    #     cfg['prod']['l3l1'] = [cfg['prod']['restriction']['r_o_r_1']] * cfg['num_layers']
    if isinstance(cfg['prod']['l1l2'], str) or cfg['prod']['l1l2'] == None:
        cfg['prod']['l1l2'] = [cfg['prod']['l1l2']] * cfg['num_layers']
    if isinstance(cfg['prod']['l3l1'], str) or cfg['prod']['l3l1'] == None:
        cfg['prod']['l3l1'] = [cfg['prod']['l3l1']] * cfg['num_layers']
    if isinstance(cfg['prod']['correlation'], int):
        cfg['prod']['correlation'] = [cfg['prod']['correlation']] * cfg['num_layers']

    return cfg