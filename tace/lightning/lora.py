################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
from typing import Dict, Union

import torch


from ..models.v1.mlp import LinearLayer, LoRALinearLayer
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

def to_lora_mlp(
    linear: LinearLayer,
    lora_config: Dict[str, int | float | bool],
) -> LoRALinearLayer:
    
    arguments = {
        'in_dim': linear.in_dim,
        'out_dim': linear.out_dim,
        'alpha': linear.alpha,
        'bias': linear.bias is not None,
        'lora_r': int(lora_config.get('r', -1)),
        'lora_alpha': float(lora_config.get('alpha', lora_config.get('r', -1))),
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

def from_lora_mlp(lora_linear: LoRALinearLayer) -> LinearLayer:
    arguments = {
        'in_dim': lora_linear.in_dim,
        'out_dim': lora_linear.out_dim,
        'alpha': lora_linear.alpha,
        'bias': lora_linear.bias is not None,
    }
    linear = LinearLayer(**arguments)
    linear = linear.to(
        device=lora_linear.weight.device,
        dtype=lora_linear.weight.dtype,
    )
    with torch.no_grad():
        delta_w = lora_linear.lora_A @ (lora_linear.lora_B * lora_linear.scaling)
        linear.weight.copy_(lora_linear.weight + delta_w)
        if arguments['bias']:
            linear.bias.copy_(lora_linear.bias)
    return linear

def to_lora_linear(
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

def from_lora_linear(
    lora_linear: Union[
        LoRALinear,
        LoRAElementLinear,
        LoRACWLinear,
        LoRAElementCWLinear,
    ]
) -> Union[
    Linear,
    ElementLinear,
    CWLinear,
    ElementCWLinear,
]:
    #  === Linear  ===
    if isinstance(lora_linear, LoRALinear):
        arguments = {
            'in_dim': lora_linear.in_dim,
            'out_dim': lora_linear.out_dim,
            'bias': lora_linear.bias is not None,
            'l': lora_linear.l,
        }
        linear = Linear(**arguments).to(
            device=lora_linear.weight.device,
            dtype=lora_linear.weight.dtype,
        )
        with torch.no_grad():
            delta_w = lora_linear.lora_A @ (
                lora_linear.lora_B * lora_linear.scaling
            )
            linear.weight.copy_(lora_linear.weight + delta_w)
            if arguments['bias']:
                linear.bias.copy_(lora_linear.bias)
        return linear

    # === CWLinear  ===
    elif isinstance(lora_linear, LoRACWLinear):
        arguments = {
            'in_dim': lora_linear.in_dim,
            'out_dim': lora_linear.out_dim,
            'bias': lora_linear.bias is not None,
            'l': lora_linear.l,
        }
        linear = CWLinear(**arguments).to(
            device=lora_linear.weight.device,
            dtype=lora_linear.weight.dtype,
        )
        with torch.no_grad():
            delta_w = lora_linear.lora_A @ (
                lora_linear.lora_B * lora_linear.scaling
            )
            linear.weight.copy_(lora_linear.weight + delta_w)
            if arguments['bias']:
                linear.bias.copy_(lora_linear.bias)
        return linear

    #  === ElementLinear ===
    elif isinstance(lora_linear, LoRAElementLinear):
        arguments = {
            'in_dim': lora_linear.in_dim,
            'out_dim': lora_linear.out_dim,
            'bias': lora_linear.bias is not None,
            'l': lora_linear.l,
            'atomic_numbers': lora_linear.atomic_numbers,
        }
        linear = ElementLinear(**arguments).to(
            device=lora_linear.weights.device,
            dtype=lora_linear.weights.dtype,
        )
        with torch.no_grad():
            if lora_linear.element_aware:
                delta_w = torch.einsum(
                    'zri, zor -> zoi',
                    lora_linear.lora_A,
                    lora_linear.lora_B * lora_linear.scaling,
                )
            else:
                delta_w = (lora_linear.lora_B * lora_linear.scaling) @ lora_linear.lora_A 
                delta_w = delta_w.unsqueeze(0)
            linear.weights.copy_(lora_linear.weights + delta_w)
            if arguments['bias']:
                linear.bias.copy_(lora_linear.bias)
        return linear

    # === ElementCWLinear ===
    elif isinstance(lora_linear, LoRAElementCWLinear):
        arguments = {
            'in_dim': lora_linear.in_dim,
            'out_dim': lora_linear.out_dim,
            'bias': lora_linear.bias is not None,
            'l': lora_linear.l,
            'atomic_numbers': lora_linear.atomic_numbers,
        }
        linear = ElementCWLinear(**arguments).to(
            device=lora_linear.weights.device,
            dtype=lora_linear.weights.dtype,
        )
        with torch.no_grad():
            delta_w = (lora_linear.lora_B.T @ lora_linear.lora_A.T) * lora_linear.scaling
            linear.weights.copy_(lora_linear.weights + delta_w.unsqueeze(0))
            if arguments['bias']:
                linear.bias.copy_(lora_linear.bias)
        return linear

    else:
        raise TypeError(f"Unsupported LoRA linear type: {type(lora_linear)}")


def inject_lora_into_model(
    model: torch.nn.Module,
    lora_configs: Dict[str, Dict[str, int | float]],
):
    to_replace = []
    for full_name, module in model.named_modules():
        if isinstance(module, LinearLayer):
            to_replace.append((full_name, module, to_lora_mlp))
        elif isinstance(module, Linear | ElementLinear | CWLinear | ElementCWLinear):
            to_replace.append((full_name, module, to_lora_linear))


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

def to_lora_model(finetune_cfg: Dict, model: torch.nn.Module) -> torch.nn.Module:
    if not finetune_cfg: 
        return model

    # === LoRA ===
    lora = finetune_cfg.get('lora', {})
    if len(lora) > 0:
        inject_lora_into_model(model, lora)

    # === Freeze ===
    freeze = finetune_cfg.get('freeze', {})
    if len(freeze) > 0:
        name_to_param = dict(model.named_parameters())
        for name, flag in freeze.items():
            if name not in name_to_param:
                logging.warning(f"Parameter '{name}' not found in model")
                continue
            param = name_to_param[name]
            param.requires_grad = not bool(flag)
    logging.info(model)

    return model

def from_lora_to_merged_model(model: torch.nn.Module) -> torch.nn.Module:
    def _replace(module: torch.nn.Module):
        for name, child in module.named_children():
            # === LoRA MLP ===
            if isinstance(child, LoRALinearLayer):
                new_module = from_lora_mlp(child)
                setattr(module, name, new_module)
            # === LoRA Linear ===
            elif isinstance(
                child,
                (
                    LoRALinear,
                    LoRAElementLinear,
                    LoRACWLinear,
                    LoRAElementCWLinear,
                ),
            ):
                new_module = from_lora_linear(child)
                setattr(module, name, new_module)
            else:
                # recurse
                _replace(child)
    _replace(model)
    return model
