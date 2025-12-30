################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import importlib
from typing import Any, Dict, Optional, List


import torch


from ..dataset.statistics import Statistics
from ..utils.utils import deep_convert


def select_wrapper(cfg: Dict) -> Any:
    wrapper_path = cfg["model"]["config"].get("wrapper", {}).get("_target_", "tace.models.WrapModelV1")
    module_name, class_name = wrapper_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    wrap_cls = getattr(module, class_name)
    return wrap_cls


def select_model(
    cfg: Dict,
    statistics: Optional[Statistics],
    target_property: List[str],
    embedding_property: List[str],
) -> torch.nn.Module:
    # === wrapper ===
    WRAPPER_CLS = select_wrapper(cfg)

    # === model cls ===
    model_path = cfg['model']['config'].get('_trget_', 'tace.models.TACEV1')
    if model_path == "tace.models.tace.TACE": # for compatible with earlier version
        model_path = "tace.models.TACEV1"
    module_name, class_name = model_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    MODEL_CLS = getattr(module, class_name)
    model_config = deep_convert(cfg['model']['config'])
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
