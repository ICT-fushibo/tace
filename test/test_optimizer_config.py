import copy

import torch

from tace.lightning.lit_model import LightningWrapperModel


def test_configure_optimizers_does_not_mutate_configuration():
    cfg = {
        "finetune": {},
        "loss": {"_target_": "torch.nn.MSELoss"},
        "dataset": {},
        "misc": {"LossSkipController": {"enable": False}},
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 1.0e-3,
            "weight_decay": 1.0e-8,
        },
        "scheduler": {
            "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.5,
            "extra": {
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        },
    }
    original_cfg = copy.deepcopy(cfg)
    module = LightningWrapperModel(
        cfg,
        torch.nn.Linear(1, 1),
        target_property=[],
        embedding_property=[],
        statistics=[],
    )

    first = module.configure_optimizers()
    second = module.configure_optimizers()

    assert module.cfg is not cfg
    assert module.cfg["optimizer"] is not cfg["optimizer"]
    assert module.hparams["cfg"] is module.cfg
    assert cfg == original_cfg
    assert module.cfg == original_cfg
    assert isinstance(first["optimizer"], torch.optim.AdamW)
    assert isinstance(second["optimizer"], torch.optim.AdamW)
    assert isinstance(
        first["lr_scheduler"]["scheduler"],
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    )
    assert isinstance(
        second["lr_scheduler"]["scheduler"],
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    )
