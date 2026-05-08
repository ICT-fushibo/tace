################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging

import torch


def to_lora_model(finetune_cfg: dict, model: torch.nn.Module) -> torch.nn.Module:
    if not finetune_cfg: 
        return model

    # # === LoRA ===
    # lora = finetune_cfg.get('lora', {})
    # if len(lora) > 0:
    #     inject_lora_into_model(model, lora)

    # === Freeze ===
    freeze = finetune_cfg.get('freeze', {})
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