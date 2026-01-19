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
    "traceless": True,
    "norm": True,
}


RADIAL_MLP = {
    "hidden": [64, 64, 64],
    "act": "silu",
    "bias": False,
}


INTER = {
    "l1l2": None,
    "conv_weights": ["edge_ij"],
    "normalizer": "avg_num_neighbors",
    "nonlinearity": {
      "type": None,
      "gate": 'silu', 
    },
    "sc": {
      "use_first_sc": False, 
    },
    "use_resnet": False,   
}


PROD = {
    "l1l2": None,
    "l3l1": None,
    "correlation": 3,
    "element_aware": True,
    "coupled_channel": True,
    "precompute": False,
}


READOUT_EMLP = {
    "hidden": [16], 
    "act": "silu", 
    "gate": "silu",
    "bias": False,
    "use_all_layer": True,
    "enable_uie_readout": False,
}


SCALE_SHIFT = {
    "scale_type": "rms_forces",
    "shift_type": "mean_delta_energy_per_atom",
    "scale_trainable": False,
    "shift_trainable": False,
}


SHORT_RANGE = {
    "zbl": {
      "enable": False,
      "trainable": False,
    }
}

LONG_RANGE = {
    "les": {
        "enable": False,
        "les_arguments": {
            "n_layers": 3,
            "n_hidden": [32, 16],
            "add_linear_nn": True,
            "output_scaling_factor": 0.1,
            "sigma": 1.0,
            "dl": 2.0,
            "remove_self_interaction": True,
            "remove_mean": True,
            "epsilon_factor": 1.0,
            "use_atomwise": False,
            "compute_bec": False,
            "bec_output_index": None,
        },
    },
}


CONSERVATION = {
    "charges": {
        "method": "lagrangian",
    },
}


UNIVERSAL_EMBEDDING = {
    "invariant": {
        "level": {
            "enable": False,
            "num_embeddings": -1,
        },
        "spin_on": {
            "enable": False,
            "num_embeddings": 2,
        },
        "spin_multiplicity": {
            "enable": False,
            "num_embeddings": -1,
        },
        "charges": {
            "enable": False,
            "act": "silu",
        },
        "total_charge": {
            "enable": False,
            "act": "silu",
        },
        "initial_collinear_magmoms": {
            "enable": False,
            "act": "silu",
        },
        "temperature": {
            "enable": False,
            "act": "silu",
        },
        "electron_temperature": {
            "enable": False,
            "act": "silu",
        },
    },
    "equivariant": {
        "initial_noncollinear_magmoms": {
            "enable": False,
            "normalizer": 1.0,
            "gate": None,
            "linear": False,
        },
        "electric_field": {
            "enable": False,
            "normalizer": 1.0,
            "gate": None,
            "linear": False,
        },
        "magnetic_field": {
            "enable": False,
            "normalizer": 1.0,
            "gate": None,
            "linear": False,
        },
    },
}

SPECIAL = {
    "hessians":{
        'num_samples': 2
    }    
}

from typing import Dict, Any, List, Tuple

def check_model_config(cfg: Dict[str, Any]):
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
    assert cfg['conservation'] is None or isinstance(cfg['conservation'], Dict), "cfg.model.config.conservations must be a Dict or None"
    assert cfg['universal_embedding'] is None or isinstance(cfg['universal_embedding'], Dict), "cfg.model.config.universal_embedding must be a Dict or None"

    # statistics
    if not isinstance(cfg['statistics'], List): 
        cfg['statistics'] = [cfg['statistics']]

    # Lmax, lmax
    if isinstance(cfg['Lmax'], int):
        cfg['Lmax'] = [cfg['Lmax']] * cfg['num_layers']
    if isinstance(cfg['lmax'], int):
        cfg['lmax'] = [cfg['lmax']] * cfg['num_layers']

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

    if "energy" in cfg['target_property']: 
        cfg['atomic_energies'] = [stats['atomic_energy'] for stats in cfg['statistics']]
    else:
        cfg['atomic_energies'] = None

    # short_range and long_range
    cfg['short_range'] = cfg.get('short_range', SHORT_RANGE) 
    cfg['long_range'] = cfg.get('long_range', LONG_RANGE)
 
    # inter
    if isinstance(cfg['inter']['l1l2'], str) or cfg['inter']['l1l2'] == None:
        cfg['inter']['l1l2'] = [cfg['inter']['l1l2']] * cfg['num_layers']

    # prod
    if isinstance(cfg['prod']['l1l2'], str) or cfg['prod']['l1l2'] == None:
        cfg['prod']['l1l2'] = [cfg['prod']['l1l2']] * cfg['num_layers']
    if isinstance(cfg['prod']['l3l1'], str) or cfg['prod']['l3l1'] == None:
        cfg['prod']['l3l1'] = [cfg['prod']['l3l1']] * cfg['num_layers']
    if isinstance(cfg['prod']['correlation'], int):
        cfg['prod']['correlation'] = [cfg['prod']['correlation']] * cfg['num_layers']


    # === uie and uee ===
    universal_embedding = cfg.get('universal_embedding', {})
    universal_embedding['invariant'] = universal_embedding.get('invariant', {})
    universal_embedding['equivariant'] = universal_embedding.get('equivariant', {})
    universal_embedding['invariant_embedding_property'] = []
    for k, v in universal_embedding['invariant'].items():
        if v.get('enable', False):
            universal_embedding['invariant_embedding_property'].append(k)
    universal_embedding['equivariant_embedding_property'] = []
    for k, v in universal_embedding['equivariant'].items():
        if v.get('enable', False):
            universal_embedding['equivariant_embedding_property'].append(k)

    # === conservation ===
    cfg['conservation'] = cfg.get('conservation', CONSERVATION)

    # === special ===
    cfg['special'] = cfg.get('special', SPECIAL)

    return cfg