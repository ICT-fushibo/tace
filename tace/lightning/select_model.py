################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

# TODO, refactor code


import importlib
from typing import Any, Dict, List
import logging

import torch


from tace.utils.env import get_tace_use_compile
from ..utils.utils import deep_convert


def select_wrapper(model_config: Dict, wrapper_path: str = None) -> Any:
    if wrapper_path is None:
        wrapper_path = model_config.get("wrapper", {}).get("_target_", "tace.models.TensorModel")
    module_name, class_name = wrapper_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    wrap_cls = getattr(module, class_name)
    return wrap_cls


def select_model(
    cfg: Dict,
    statistics: Dict,
    target_property: List[str],
    embedding_property: List[str],
    **kwargs,
) -> torch.nn.Module:
    
    if "model" in cfg:
        model_config = (cfg['model']['config'])
    else:
        model_config = cfg

    # === model cls ===
    if 'kwargs' in model_config:
        model_path = model_config['kwargs'].get('_target_', 'tace.models.e3nnTACE')
    else:
        model_path = model_config.get('_target_', 'tace.models.e3nnTACE')

    wrapper_path = model_config.get("wrapper", {}).get("_target_", "tace.models.TensorModel")

    if get_tace_use_compile() == "1" and model_path in {
        "tace.models.e3nnTACE",
        "tace.models._e3nn.e3nnTACE",
    }:
        model_path = "tace.models.compile.e3nnTACE"
        if wrapper_path == "tace.models.TensorModel":
            wrapper_path = "tace.models.CompileTensorModel"
    else:
        logging.warning(
            "You are not using AOTI. "
            "For acceleration options, see "
            "https://tace.readthedocs.io/en/latest/guide/acceleration.html"
        )


    # === wrapper cls ===
    WRAPPER_CLS = select_wrapper(model_config, wrapper_path)
        
    module_name, class_name = model_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    MODEL_CLS = getattr(module, class_name)
    model_config = deep_convert(model_config)
    if 'statistics' not in model_config:
        model_config['statistics'] = statistics
    if 'target_property' not in model_config:
        model_config['target_property'] = target_property
    if 'embedding_property' not in model_config:
        model_config['embedding_property'] = embedding_property
    # === instantiate ===
    try:
        MODEL = WRAPPER_CLS(
            MODEL_CLS(
                **model_config,
            )
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate the model using the provided configuration.\n"
            # f"Model config: {model_config}"
        ) from e

    return MODEL
