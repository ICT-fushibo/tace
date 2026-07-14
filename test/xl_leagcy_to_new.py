
import os
os.environ["TACE_USE_DENS"] = '1'
os.environ["TACE_USE_MATRIX_WEIGHT"] = '1'
import sys
from collections import Counter

import torch

from tace.lightning import load_tace, export_tace
from tace.dataset.quantity import get_target_property, get_embedding_property
from tace.lightning.model import create_model



def load_from_checkpoint(
    ckpt_path: str,
    map_location: str = "cpu",
    strict = True,
    use_ema = 1,
) -> torch.nn.Module:

    checkpoint = torch.load(
        ckpt_path, map_location=map_location, weights_only=False
    )
    dtypes = [
        v.dtype for v in checkpoint["state_dict"].values()
        if hasattr(v, "dtype") and torch.is_floating_point(v)
    ]
    dominant_dtype = Counter(dtypes).most_common(1)[0][0] # original training precision
    cfg = checkpoint['hyper_parameters']['cfg']

    del cfg['model']['config']['atomic_basis']['use_both_Bi_Bj']
    del cfg['model']['config']['atomic_basis']['separate_so2_radial']
    del cfg['model']['config']['atomic_basis']['is_so2_layout']
    cfg['model']['config']['atomic_basis']['use_graph_softmax'] = [
        False,
        True,
        False,
        True,  
        False,
        True,  
        False,
        True,  
    ]

    for k, v in cfg['model']['config']['atomic_basis'].items():
        print(k)
        print(v)
        print()

    target_property = get_target_property(cfg)
    embedding_property = get_embedding_property(cfg)
    statistics = checkpoint['hyper_parameters']['statistics']
    model = create_model(cfg, statistics, target_property, embedding_property)
    model.to(dtype=dominant_dtype)
    state_dict = {
        k[len("model.") :]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")
    }
    model.load_state_dict(state_dict, strict=strict)

    # === EMA ===
    if bool(use_ema) and "ema_state_dict" in checkpoint:
        ema_params = checkpoint['ema_state_dict']['shadow_params']
        for idx, (name, _) in enumerate(model.named_parameters()):
            state_dict[name] = ema_params[idx]
        model.load_state_dict(state_dict, strict=strict)

    return model.to(map_location)


model = load_from_checkpoint(sys.argv[1])

export_tace(model, "dir_model")


model.readout_fn.model_config['target_property'] = ['energy', 'forces', 'stress']
model.readout_fn.model_config['product_basis']['return_components'] = [0, 1, 2]
model.readout_fn.model_config['dropout']['stochastic_depth'] = 0.0
model.readout_fn.model_config['dropout']['use_first_dropout'] = False

del model.readout_fn.representation.forces_embedding
del model.readout_fn.direct_forces_readouts
del model.readout_fn.direct_virials_readout0s
del model.readout_fn.direct_virials_readout2s
del model.readout_fn.direct_virials_basis_change
del model.readout_fn.dens_noise_readouts
del model.readout_fn.representation.decouple_edge_updates
del model.readout_fn.representation.decouple_interactions
del model.readout_fn.representation.decouple_products

print(model)


export_tace(model, "con_model")


