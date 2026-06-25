################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
from typing import Any

import torch

from tace.models.linear import (
    e3nnElementLinear,
    e3nnLinear,
    e3nnMoEElementLinear,
    enable_lora,
    has_lora,
    merge_lora,
    mlpLinear,
    torchLinear,
)


LORA_MODULES = (
    torchLinear,
    mlpLinear,
    e3nnLinear,
    e3nnElementLinear,
    e3nnMoEElementLinear,
)


LORA_CLASS_CONFIG_KEYS = {
    torchLinear: ("torchLinear", "torch_linear"),
    mlpLinear: ("mlpLinear", "mlp_linear"),
    e3nnLinear: ("e3nnLinear", "e3nn_linear"),
    e3nnElementLinear: ("e3nnElementLinear", "e3nn_element_linear"),
    e3nnMoEElementLinear: ("e3nnMoEElementLinear", "e3nn_moe_element_linear"),
}


def _as_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    try:
        return dict(cfg)
    except TypeError:
        return {}


def _select_lora_config(
    module: torch.nn.Module,
    lora_configs: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not lora_configs:
        return None

    for cls, keys in LORA_CLASS_CONFIG_KEYS.items():
        if type(module) is cls:
            for key in keys:
                if key in lora_configs:
                    return _as_dict(lora_configs[key])
            break

    if "default" in lora_configs:
        return _as_dict(lora_configs["default"])

    return None


def inject_lora_into_model(
    model: torch.nn.Module,
    lora_configs: dict[str, dict[str, Any]],
) -> torch.nn.Module:
    injected = 0
    for _, module in model.named_modules():
        if not isinstance(module, LORA_MODULES):
            continue
        if getattr(module, "weight", None) is None:
            continue

        cfg = _select_lora_config(module, lora_configs)
        if cfg is None:
            continue

        r = int(cfg.get("r", cfg.get("rank", 4)))
        if r <= 0:
            continue

        enable_lora(
            module,
            r=r,
            alpha=float(cfg.get("alpha", r)),
            element_aware=bool(cfg.get("element_aware", True)),
            expert_aware=bool(cfg.get("expert_aware", True)),
            freeze_base=bool(cfg.get("freeze_base", True)),
            train_bias=bool(cfg.get("train_bias", False)),
        )
        injected += 1

    logging.info(f"Injected LoRA into {injected} linear modules.")
    return model


def to_lora_model(finetune_cfg: dict, model: torch.nn.Module) -> torch.nn.Module:
    if not finetune_cfg:
        return model

    lora = _as_dict(finetune_cfg.get("lora", {}))
    if len(lora) > 0:
        inject_lora_into_model(model, lora)

    freeze = _as_dict(finetune_cfg.get("freeze", {}))
    if len(freeze) > 0:
        name_to_param = dict(model.named_parameters())
        for name, flag in freeze.items():
            if name not in name_to_param:
                logging.warning(f"Parameter '{name}' not found in model")
                continue
            param = name_to_param[name]
            assert isinstance(flag, bool)
            param.requires_grad = not flag

    logging.info(model)
    return model


def from_lora_to_merged_model(model: torch.nn.Module) -> torch.nn.Module:
    merged = 0
    for module in model.modules():
        if has_lora(module):
            merge_lora(module, remove_lora=True)
            merged += 1
    logging.info(f"Merged LoRA into {merged} linear modules.")
    return model
