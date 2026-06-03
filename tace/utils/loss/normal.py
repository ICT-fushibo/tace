################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import List, Union


import torch
from omegaconf import ListConfig


from .mse_fn import LOSS_FN


class NormalLoss(torch.nn.Module):
    def __init__(
        self,
        loss_property: List[str],
        loss_function_name: List[str],
        loss_property_weights: Union[List[float], None],
        loss_huber_delta: Union[float, List[float], None] = 0.01,
        normalize: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(
            loss_property, (list, ListConfig)
        ), f"cfg.loss.loss_property should be a list, got {type(loss_property)}"
        assert isinstance(
            loss_function_name, (list, ListConfig)
        ), f"cfg.loss.loss_function_name should be a list, got {type(loss_function_name)}"
        assert isinstance(
            loss_property_weights, (list, ListConfig)
        ), f"cfg.loss.loss_property_weights should be a list, got {type(loss_property_weights)}"
        if isinstance(loss_huber_delta, float) or loss_huber_delta is None:
            loss_huber_delta = [loss_huber_delta] * len(loss_function_name)
        assert isinstance(
            loss_huber_delta, (list, ListConfig)
        ), f"cfg.loss.loss_huber_delta should be a list, got {type(loss_huber_delta)}"
        assert len(loss_function_name) == len(loss_property_weights) == len(loss_huber_delta)
        assert len(loss_property) <= len(loss_function_name)
        for fn in loss_function_name:
            assert (
                fn in LOSS_FN
            ), f"{fn} not in LOSS_FN, add by yourself, all available function name are {list(LOSS_FN)}"
        self.loss_property = loss_property
        self.loss_function_name = loss_function_name
        self.loss_huber_delta = loss_huber_delta
        if normalize:
            normalizer = sum(loss_property_weights)
            self.loss_property_weights = [w / normalizer for w in loss_property_weights]
        else:
            self.loss_property_weights = loss_property_weights

    def forward(self, pred, label):
        total_loss = 0.0
        for i, func_name in enumerate(self.loss_function_name):
            huber_delta = self.loss_huber_delta[i]
            loss = LOSS_FN[func_name](pred, label, huber_delta)
            total_loss += loss * self.loss_property_weights[i]
        return total_loss

    def __repr__(self):
        task_strs = [
            f"  - {p:<12} | "
            f"weight={w:>7.3f} | delta={d:.3f} | fn={fn}"
            for p, fn, w, d in zip(
                self.loss_property,
                self.loss_function_name,
                self.loss_property_weights,
                self.loss_huber_delta,
            )
        ]
        tasks_info = "\n".join(task_strs)
        return f"{self.__class__.__name__}(\n{tasks_info}\n)"


