################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
# TODO not for users now

import argparse
import yaml


from ..lightning import load_tace


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to model checkpoint (.ckpt or .pt or .pth)")
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_tace(args.model, 'cpu', strict=True, use_ema=True)
    finetune_cfg = {}
    finetune_cfg['atomic_numbers'] = None

    # === Freeze ===
    finetune_cfg['freeze'] = {}
    for name, _ in model.named_parameters():
        finetune_cfg['freeze'][name] = True

    # === LoRA ===
    finetune_cfg['lora'] = {}
    def set_lora_config(name: str, r: int, alpha: float):
        finetune_cfg['lora'][name] = {
            'r': int(r),
            'alpha': float(alpha),
        }
    set_lora_config('element_embedding', 8, 8)
    set_lora_config('radial_mlp', 8, 8)
    set_lora_config('interaction', 8, 8)
    set_lora_config('product', 8, 8)
    set_lora_config('readout', 8, 8)
    finetune_cfg['lora']['interaction']['element_aware'] = False
    finetune_cfg['lora']['product']['element_aware'] = False

    with open('finetune_config.yaml', "w") as f:
        yaml.dump(finetune_cfg, f, default_flow_style=False, sort_keys=False)
    

if __name__ == "__main__":
    main()
