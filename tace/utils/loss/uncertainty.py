################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Union

import torch
from omegaconf import ListConfig
from torch import nn

from .mse_fn import LOSS_FN
from .registry import validate_loss_function_names


class UncertaintyLoss(nn.Module):
    """
    Multi-task uncertainty weighted loss.
    Based on: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    (https://arxiv.org/abs/1705.07115)
    """

    def __init__(
        self,
        loss_property: list[str],
        loss_function_name: list[str],
        loss_property_weights: list[float],
        loss_huber_delta: Union[float, list[float]] = 0.01,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(loss_property, (list, ListConfig)), (
            f"cfg.loss.loss_property should be a list, got {type(loss_property)}"
        )
        assert isinstance(loss_function_name, (list, ListConfig)), (
            f"cfg.loss.loss_function_name should be a list, got {type(loss_function_name)}"
        )
        assert isinstance(loss_property_weights, (list, ListConfig)), (
            f"cfg.loss.loss_property_weights should be a list, got {type(loss_property_weights)}"
        )
        init_log_sigmas = [-math.log(2.0 * w) for w in loss_property_weights]
        if isinstance(loss_huber_delta, float) or loss_huber_delta is None:
            loss_huber_delta = [loss_huber_delta] * len(loss_function_name)
        assert isinstance(loss_huber_delta, (list, ListConfig)), (
            f"cfg.loss.loss_huber_delta should be a list, got {type(loss_huber_delta)}"
        )
        assert len(loss_function_name) == len(init_log_sigmas) == len(loss_huber_delta)
        assert len(loss_property) <= len(loss_function_name)
        validate_loss_function_names(loss_function_name)

        self.loss_property = loss_property
        self.loss_function_name = loss_function_name
        self.loss_huber_delta = loss_huber_delta
        self.log_sigmas = nn.ParameterDict()
        for p, val in zip(loss_property, init_log_sigmas):
            self.log_sigmas[p] = nn.Parameter(
                torch.tensor(val, dtype=torch.get_default_dtype())
            )

    def forward(self, pred, label):
        total_loss = 0.0
        for i, (p, fn_name) in enumerate(
            zip(self.loss_property, self.loss_function_name)
        ):
            huber_delta = self.loss_huber_delta[i]
            p_loss = LOSS_FN[fn_name](pred, label, huber_delta)
            log_sigma = self.log_sigmas[p]
            total_loss += 0.5 * torch.exp(-log_sigma) * p_loss + log_sigma
        return total_loss

    def __repr__(self):
        task_strs = [
            f"  - {p:<10} | log_sigma={v.item():>7.3f} | delta={d:.3f} | fn={fn}"
            for p, fn, v, d in zip(
                self.loss_property,
                self.loss_function_name,
                self.log_sigmas.values(),
                self.loss_huber_delta,
            )
        ]
        tasks_info = "\n".join(task_strs)
        return f"{self.__class__.__name__}(\n{tasks_info}\n)"
