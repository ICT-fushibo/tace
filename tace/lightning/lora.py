################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
# TODO not for users now
import logging
from math import sqrt
from typing import Optional, Dict, Any, Union

import torch

from ..models import WrapModelV1, TACEV1
from ..models.v1.representation import TACEDescriptor
from ..models.v1.mlp import LinearLayer, LoRALinearLayer, MLP
from ..models.v1.linear import (
    Linear, 
    ElementLinear, 
    CWLinear, 
    ElementCWLinear,
    LoRALinear,
    LoRAElementLinear,
    LoRACWLinear,
    LoRAElementCWLinear,
)

def replace_mlp_with_lora(
    linear: LinearLayer,
    lora_config: Dict[str, int | float | bool],
) -> LoRALinearLayer:
    
    arguments = {
        'in_dim': linear.in_dim,
        'out_dim': linear.out_dim,
        'alpha': linear.alpha,
        'bias': linear.bias is not None,
        'lora_r': int(lora_config.get('r', 0)),
        'lora_alpha': float(lora_config.get('alpha', lora_config.get('r', 0))),
    }

    if arguments['lora_r'] <= 0:
        return linear
    
    lora_linear = LoRALinearLayer(**arguments)
    lora_linear = lora_linear.to(
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    with torch.no_grad():
        lora_linear.weight.copy_(linear.weight)
        if arguments['bias']:
            lora_linear.bias.copy_(linear.bias)
 
    return lora_linear


def replace_linear_with_lora(
    linear: Union[Linear, ElementLinear, CWLinear, ElementCWLinear],
    lora_config: Dict[str, int | float | bool],
) -> Union[LoRALinear, LoRAElementLinear, LoRACWLinear, LoRAElementCWLinear]:
    
    arguments = {
        'in_dim': linear.in_dim,
        'out_dim': linear.out_dim,
        'bias': linear.bias is not None,
        'l': linear.l,
        'lora_r': int(lora_config.get('r', 4)),
        'lora_alpha': float(lora_config.get('alpha', lora_config.get('r', 4))),
        'element_aware': bool(lora_config.get('element_aware', True)),
        'atomic_numbers': getattr(linear, 'atomic_numbers', None)
    }

    if arguments['lora_r'] <= 0:
        return linear
    
    if isinstance(linear, Linear):
        lora_linear = LoRALinear(**arguments)
        lora_linear = lora_linear.to(
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            lora_linear.weight.copy_(linear.weight)
            if arguments['bias']:
                lora_linear.bias.copy_(linear.bias)
    elif isinstance(linear, CWLinear):
        lora_linear = LoRACWLinear(**arguments)
        lora_linear = lora_linear.to(
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            lora_linear.weight.copy_(linear.weight)
            if arguments['bias']:
                lora_linear.bias.copy_(linear.bias)
    elif isinstance(linear, ElementLinear):
        lora_linear = LoRAElementLinear(**arguments)
        lora_linear = lora_linear.to(
            device=linear.weights.device,
            dtype=linear.weights.dtype,
        )
        with torch.no_grad():
            lora_linear.weights.copy_(linear.weights)
            if arguments['bias']:
                lora_linear.bias.copy_(linear.bias)
    elif isinstance(linear, ElementCWLinear):
        lora_linear = LoRAElementCWLinear(**arguments)
        lora_linear = lora_linear.to(
            device=linear.weights.device,
            dtype=linear.weights.dtype,
        )
        with torch.no_grad():
            lora_linear.weights.copy_(linear.weights)
            if arguments['bias']:
                lora_linear.bias.copy_(linear.bias)

    return lora_linear


def inject_lora_into_model(
    model: torch.nn.Module,
    lora_configs: Dict[str, Dict[str, int | float]],
):
    # for name, module in model.named_children():
    #     if isinstance(module, Linear):
    #         setattr(
    #             model,
    #             name,
    #             replace_linear_with_lora(module, inter_config)
    #         )
    #     else:
    #         inject_lora_into_model(module, lora_configs)

    to_replace = []
    for full_name, module in model.named_modules():
        if isinstance(module, LinearLayer):
            to_replace.append((full_name, module, replace_mlp_with_lora))
        elif isinstance(module, Linear | ElementLinear | CWLinear | ElementCWLinear):
            to_replace.append((full_name, module, replace_linear_with_lora))


    for full_name, module, replace_fn in to_replace:
        parent = model
        *path, attr = full_name.split(".")
        for p in path:
            parent = getattr(parent, p)

        if '.node_embedding.' in full_name:
            cfg = lora_configs['element_embedding']
        elif '.radial_net.' in full_name:
            cfg = lora_configs['radial_mlp']
        elif '.interactions.' in full_name:
            cfg = lora_configs['interaction']
        elif '.products.' in full_name:
            cfg = lora_configs['product']
        elif 'readout' in full_name:
            cfg = lora_configs['readout']
        else:
            raise

        setattr(
            parent,
            attr,
            replace_fn(module,  cfg)
        )

    logging.info(model)
