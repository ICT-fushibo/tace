################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import importlib
from typing import Any, Dict, Optional, List


import torch


from ..dataset.statistics import Statistics
from ..utils.utils import deep_convert


def select_wrapper(model_config: Dict) -> Any:
    wrapper_path = model_config.get("wrapper", {}).get("_target_", "tace.models.WrapModelV1")
    module_name, class_name = wrapper_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    wrap_cls = getattr(module, class_name)
    return wrap_cls


def select_model(
    cfg: Dict,
    statistics: Optional[Statistics],
    target_property: List[str],
    embedding_property: List[str],
    **kwargs,
) -> torch.nn.Module:
    
    if "model" in cfg:
        model_config = (cfg['model']['config'])
    else:
        model_config = cfg

    # === wrapper cls ===
    WRAPPER_CLS = select_wrapper(model_config)

    # === model cls ===
    model_path = model_config.get('_target_', 'tace.models.TACEV1')
    module_name, class_name = model_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    MODEL_CLS = getattr(module, class_name)
    model_config = deep_convert(model_config)
    # === instantiate ===
    try:
        MODEL = WRAPPER_CLS(
            MODEL_CLS(
                **model_config,
                target_property=target_property,
                embedding_property=embedding_property,
                statistics=statistics,
                model_config=model_config,
            )
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate the model using the provided configuration.\n"
            f"Model config: {model_config}"
        ) from e

    return MODEL
