################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, List, Any


from ...dataset.quantity import PROPERTY


DEFAULT_MODEL_CONFIG = {
    "Lmax": 2,
    "lmax": 3,
    "num_channel": 64,
    "num_layers": 2,
    "target_property": ["energy", "forces"],
    "embedding_property": [],
    "fidelity": {
        "name": "PBE",
        "atomic_energy": None,
    },
    "node_embedding": {
        "type": "linear",
    },
    "edge_embedding": {
        "type": "identity",
    },
    "edge_update": {
        "type": "identity",
    },
    "radial_basis": {
        "bias": False,
        "radial_basis": "j0",
        "num_radial_basis": 8,
        "distance_transform": None,
        "cutoff_fn": 'c2poly',
        "polynomial_cutoff": 5,
        "order": 0,
        "trainable": False,
        "apply_cutoff": True,
        "hidden": [64, 64, 64],
        "gaussian_width": 2.0,
    },
    "angular_basis": {
    },
    "atomic_basis": {
        "type": "normal",
        "edge_info_type": "mlp",
        "scatter_norm": "density",
        "nonlinear": None,
        "l1l2": None,
        "ictp_ictc_like": True,
        "num_experts": None,
        "num_shared_experts": None,
        "top_k": None,
        "moe_channel": None,
    },
    "resnet": {
        "type": "BB",
        "use_first_resnet": False,
    },
    "layer_norm": {
        "pre_norm_type": None,
        "final_norm_type": None,
        "use_first_pre_norm": False,
    },
    "product_basis": {
        "type": "coupled",
        "l1l2": None,
        "correlation": 3,
        "ictp_ictc_like": True,
        "num_latitude": None,
        "num_longitude": None,
        "truncation": None,
        "trainable_scale": True,
        "num_channel": None,
        "num_experts": None,
        "num_shared_experts": None,
        "top_k": None,
        "nonlinear": None,
    },
    "readout_emlp": {
        "bias": False,
        "hidden": [16],
        "use_alllayer": True,
        "use_uie": False,
    },
    "scale_shift": {
        "scale_type": "rms_forces",
        "shift_type": None,
        "scale_trainable": False,
        "shift_trainable": False,
    },
    "short_range": {
        "zbl": {
        "enable": False,
        "trainable": False,
        }
    },
    "long_range": {
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
    },
    "universal_embedding": {
        "charges": {
            "enable": False,
            "act": "silu",
        },
        "total_charge": {
            "enable": False,
            "act": "silu",
        },
        "spin_multiplicity": {
            "enable": False,
            "num_embeddings": -1,
        },
        "initial_noncollinear_magmoms": {
            "enable": False,
            "normalizer": 1.0,
        },
        "electric_field": {
            "enable": False,
            "normalizer": 1.0,
        },
        "magnetic_field": {
            "enable": False,
            "normalizer": 1.0,
        },
    },
    "special": {
        "hessians": {
            "num_samples": 2
        },
        "charges": {
            "method": "lagrangian",
        },
    },
}


def recursive_update(
        cfg: Dict[str, Any], 
        *, 
        default: Dict[str, Any] = DEFAULT_MODEL_CONFIG,
    ) -> Dict[str, Any]:
    """
    Recursively update `default` with `cfg`.

    Rules:
    - if key not in cfg: keep default
    - if both values are dict: recurse
    - otherwise: cfg overrides default
    """
    result = {}

    for key, default_val in default.items():
        if key not in cfg:
            result[key] = default_val
            continue

        cfg_val = cfg[key]

        if isinstance(default_val, Dict) and isinstance(cfg_val, Dict):
            result[key] = recursive_update(cfg_val, default=default_val)
        else:
            result[key] = cfg_val

    # allow cfg to introduce new keys
    for key, cfg_val in cfg.items():
        if key not in result:
            result[key] = cfg_val

    return result


def get_invariant_property(ue: Dict) -> List[str]:
    invariant_property = []
    for p, v in ue.items():
        if v['enable'] and PROPERTY[p]['rank'] == 0:
            invariant_property.append(p)
    return invariant_property


def get_equivariant_property(ue: Dict) -> List[str]:
    equivariant_property = []
    for p, v in ue.items():
        if v['enable'] and PROPERTY[p]['rank'] > 0:
            equivariant_property.append(p)
    return equivariant_property


def check_model_config(cfg: Dict[str, Any]):

    # Update default config with user config
    cfg = recursive_update(cfg)

    # Update statistics info
    cfg['atomic_numbers'] = sorted(
        {z for s in cfg['statistics'] for z in s['atomic_numbers']}
    )
    cfg["avg_num_neighbors"] = (
        sum(s["avg_num_neighbors"] for s in cfg['statistics']) / len(cfg['statistics'])
    )
    cfg['atomic_energies'] = (
        [stats['atomic_energy'] for stats in cfg['statistics']]
        if 'energy' in cfg['target_property']
        else None
    )

    # Universal_embedding
    cfg['invariant_property'] = get_invariant_property(cfg['universal_embedding'])
    cfg['equivariant_property'] = get_equivariant_property(cfg['universal_embedding'])

    # Update int to list use `num_layers`
    def _to_list(x: int | List[int]):
        if isinstance(x, int):
            return [x for _ in range(cfg['num_layers'])]
        assert isinstance(x, list)
        return x

    cfg['Lmax'] = _to_list(cfg['Lmax'])
    cfg['product_basis']['correlation'] = _to_list(cfg['product_basis']['correlation'])

    return cfg