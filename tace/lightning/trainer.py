################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
from typing import Dict
from pathlib import Path
from datetime import datetime


import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from hydra.utils import instantiate
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import StochasticWeightAveraging


from .lit_model import LightningWrapperModel
from ..utils.callbacks import PrintMetricsCallback
from ..utils.utils import log_parameters


def build_trainer(cfg: Dict, dataloader_valid: DataLoader = None) -> L.Trainer:
    """Build and configure a PyTorch Lightning Trainer.

    Args:
        cfg: Hydra configuration object containing:
            - logger: Configuration for experiment logger
            - callbacks: Dictionary of callback configurations
            - trainer: Base Trainer configuration

    Returns:
        Fully configured PyTorch Lightning Trainer instance

    Raises:
        ValueError: For missing critical configurations
        RuntimeError: For initialization failures in components
    """

    # === Configuration Preprocessing ===
    if not isinstance(cfg, Dict):
        raise TypeError(f"Expected DictConfig type, got {type(cfg)}")

    # === Logger Configuration ===
    logger_cfg = cfg.get("logger", None)
    logger_instance = None
    if logger_cfg is not None:
        try:
            logger_instance = instantiate(logger_cfg)
        except Exception as e:
            raise RuntimeError(f"Logger initialization failed: {str(e)}") from e

    # === User Callbacks Configuration ===
    initialized_callbacks = []
    for cb_name, cb_config in cfg.get("callbacks", {}).items():
        try:
            if cb_config.get("_target_") is None:
                logging.warning(f"Skipping unconfigured callback: {cb_name}")
                continue

            callback = instantiate({k: v for k, v in cb_config.items() if k != "extra"})
            initialized_callbacks.append(callback)
            logging.debug(
                f"Successfully loaded callback: {cb_name} ({type(callback).__name__})"
            )
        except Exception as e:
            error_msg = (
                f"Callback '{cb_name}' initialization failed\n"
                f"Config: {cb_config}\n"
                f"Error: {str(e)}"
            )
            logging.error(error_msg, exc_info=True)

    # === Built-in callbacks ===
    initialized_callbacks += [PrintMetricsCallback()]

    # === Disallow SWA callback ===
    swa_cbs = [
        cb for cb in initialized_callbacks
        if isinstance(cb, StochasticWeightAveraging)
    ]
    if swa_cbs:
        raise RuntimeError(
            "StochasticWeightAveraging (SWA) is currently not allowed in TACE when used "
            "directly via PyTorch Lightning callbacks.\n\n"
            "Reason: During testing, we observed some strange issues "
            "when using Lightning's built-in SWA.\n\n"
            "Recommended alternative:\n"
            "- Save a series of model checkpoints during training.\n"
            "- After training, use the provided `tace-average` command to compute an averaged "
            "model from any set of models with the same architecture.\n\n"
            "This approach is especially useful in the late stage of training, "
            "particularly when increasing the energy loss weight, and can significantly "
            "improve the model's generalization performance."
        )

    # === Put checkpoint at the end ===
    other_cbs = [
        cb for cb in initialized_callbacks if not isinstance(cb, ModelCheckpoint)
    ]
    checkpoint_cbs = [
        cb for cb in initialized_callbacks if isinstance(cb, ModelCheckpoint)
    ]
    initialized_callbacks = other_cbs + checkpoint_cbs     
    if checkpoint_cbs:
        for cb in checkpoint_cbs:
            logging.info(f"Model checkpoints will be saved to: {cb.dirpath}")
            if not cb.monitor:
                logging.warning(f"Checkpoint in {cb.dirpath} has no monitor metric specified")
    else:
        raise RuntimeError(
            "No ModelCheckpoint callback configured. "
            "You must provide at least one ModelCheckpoint to save the model checkpoints."
        )
    
    try:
        trainer_cfg = cfg["trainer"]
        filter_trainer_cfg = {}
        for k, v in trainer_cfg.items():
            filter_keys = ["logger", "callbacks"]
            if k not in filter_keys:
                filter_trainer_cfg.update({k: v})
        num_batch = 2 if not cfg.get("dataset", {}).get("no_valid_set", False) else 0
        filter_trainer_cfg.update({"num_sanity_val_steps": num_batch})
        trainer = instantiate(
            filter_trainer_cfg,
            logger=logger_instance,
            callbacks=initialized_callbacks,
            # _convert_="partial" # none, partial, all
        )
    except Exception as e:
        error_detail = (
            "Trainer initialization failed\n"
            f"Configuration: {(cfg["trainer"])}\n"
            f"Error: {str(e)}"
        )
        raise RuntimeError(error_detail) from e
    
    logging.info(f"Callbacks: {list(cfg.get("callbacks", {}))}")

    return trainer


# === Training entrypoint ===
def train(
    cfg: Dict,
    statistics,
    model,
    datamodule,
    target_property,
    embedding_property: list[str] = []
):
    lit_model = LightningWrapperModel(
        cfg, 
        model,
        target_property, 
        embedding_property,
        statistics, 
    )

    # Trainer
    trainer = build_trainer(cfg)

    # TRAIN AND VALID
    resume_ckpt = cfg.get("resume_from_model", None)
    if resume_ckpt:
        ckpt_path = Path(resume_ckpt)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not exists: {ckpt_path}")
        logging.info(f"Resume Training from checkpoint: {resume_ckpt}")
        trainer.fit(
            lit_model,
            datamodule=datamodule,
            ckpt_path=resume_ckpt,
        )
    else:
        logging.debug(lit_model)
        log_parameters(model)
        trainer.fit(
            lit_model,
            datamodule=datamodule,
        )
    
    # TEST
    if cfg['dataset'].get('test_files', None):
        trainer.test(
            lit_model,
            datamodule=datamodule,
            ckpt_path="best",
            verbose=False,
        )
    logging.info("Training completed at %s", datetime.now().strftime("%Y-%m-%d %H:%M"))

