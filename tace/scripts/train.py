################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import yaml
import logging
import warnings
from pathlib import Path


import hydra
from omegaconf import DictConfig, OmegaConf


from ..dataset.statistics import Statistics
from ..lightning.trainer import train
from ..lightning.lit_model import finetune, load_tace
from ..lightning.select_model import select_model
from ..dataset.dataloader import build_atomsList, compute_statistics
from ..dataset.datamodule import build_datamodule
from ..utils.hydra_resolver import register_resolvers
from ..utils.logger import set_logger
from ..utils.utils import (
    set_global_seed,
    set_precision,
    save_full_cfg,
    deep_convert,
)
from ..utils.env import set_env
from ..dataset.quantity import (
    KEYS,
    KeySpecification,
    update_keyspec_from_kwargs,
    get_target_property,
)
from ..dataset.quantity import get_embedding_property


register_resolvers()


def initialize(cfg):
    cfg = deep_convert(cfg)
    set_logger()
    if cfg['misc'].get('ignore_warning', True): 
        try:
            warnings.simplefilter("ignore", FutureWarning)
            warnings.filterwarnings(
                "ignore", module="pydantic._internal._generate_schema"
            )
        except Exception:
            pass
    save_full_cfg(cfg)
    set_env(cfg)
    set_global_seed(cfg)
    set_precision(cfg)
    return cfg


def build(cfg: DictConfig):
    cfg = initialize(OmegaConf.to_container(cfg, resolve=True, structured_config_mode="dict"))
    target_property = get_target_property(cfg)
    embedding_property = get_embedding_property(cfg)
    userKeys = KEYS
    userKeys.update(cfg['dataset'].get('keys', {}))
    keyspec = KeySpecification()
    update_keyspec_from_kwargs(keyspec, userKeys)
    num_levels = cfg['model']['config'].get("num_levels", 1)

    statistics = None
    # finetune or reusme, statistics is no need to recalculate
    if not (cfg.get("finetune_from_model", None) or cfg.get("resume_from_model", None)): 
        statistics_yaml = [Path('.') / f'statistics_{i}.yaml' for i in range(num_levels)]
        if all(yaml_file.exists() for yaml_file in statistics_yaml):
            statistics = []
            for yaml_file in statistics_yaml:
                with open(yaml_file, "r") as f:
                    statistics_data = yaml.safe_load(f)
                    statistics.append(Statistics(**statistics_data))
            for idx, yaml_file in enumerate(statistics_yaml):
                logging.info(f"Using statistics_yaml from '{str(yaml_file)}' for level {idx}")
        else:
            logging.info(f"Computing statistics information from scratch")
            element, threeAtomsList, atomic_energies = build_atomsList(
                cfg, target_property, embedding_property, keyspec, num_levels
            )
            statistics = compute_statistics(
                cfg, 
                target_property, 
                embedding_property, 
                keyspec, 
                num_levels,
                element,
                threeAtomsList,
                atomic_energies,
            )

    if cfg.get("finetune_from_model", None):
        model = finetune(cfg)
        cfg['model']['config'] = model.readout_fn.model_config
        statistics = model.readout_fn.statistics
        finetune_cfg = cfg.get('finetune', {})
        if finetune_cfg:
            logging.info(f"finetune_config field found in your main train config, use lora and parameter freezing")
        else:
            yaml_path = Path("finetune_config.yaml")
            if yaml_path.exists():
                logging.info(f"{yaml_path} found, use lora and parameter freezing")
                try:
                    finetune_cfg = yaml.safe_load(yaml_path.read_text())
                except Exception as e:
                    logging.error(f"Failed to read {yaml_path}: {e}")
            else:
                logging.warning(f"{yaml_path} not found, skipping lora and parameter freezing.")
        cfg['finetune'] = finetune_cfg
    elif cfg.get("resume_from_model", None):
        model = load_tace(
                cfg["resume_from_model"],
                device="cpu",
                strict=True,
                use_ema=True,
            )
        cfg['model']['config'] = model.readout_fn.model_config
        statistics = model.readout_fn.statistics
        cfg['finetune'] = cfg.get('finetune', {})

    else: # finetune_from_model or from scratch
        model = select_model(cfg, statistics, target_property, embedding_property)

    datamodule = build_datamodule(cfg, target_property, embedding_property, keyspec, num_levels, statistics)

    return cfg, statistics, target_property, embedding_property, model, datamodule

@hydra.main(version_base="1.3", config_path=str(Path.cwd()), config_name="tace")
def main(cfg: DictConfig):
    cfg, statistics, target_property, embedding_property, model, datamodule = build(cfg)

    train_arguments = {
        "cfg": cfg,
        "statistics": statistics,
        "target_property": target_property,
        "embedding_property": embedding_property,
        "model": model,
        "datamodule": datamodule,
    }
    train(**train_arguments)


if __name__ == "__main__":
    main()

