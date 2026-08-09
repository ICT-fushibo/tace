################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import importlib
import re
from collections import defaultdict
from typing import Iterable

from .mse_fn import LOSS_FN

LOSS_MODULES = (
    "mse_fn",
    "mae_fn",
    "huber_fn",
    "l2mae_fn",
    "dens",
    "special_fn",
)
LOSS_NAME_PREFIXES = (
    "l2mae_",
    "huber_",
    "mse_",
    "mae_",
)


def ensure_loss_functions_registered() -> None:
    for module_name in LOSS_MODULES:
        importlib.import_module(f".{module_name}", package=__package__)


def _natural_sort_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def _loss_property_name(loss_name: str) -> str:
    for prefix in LOSS_NAME_PREFIXES:
        if loss_name.startswith(prefix):
            return loss_name[len(prefix) :]
    return loss_name


def _is_special_loss(loss_fn) -> bool:
    return loss_fn.__module__.endswith(".special_fn")


def available_losses_by_property(
    *,
    include_special: bool = False,
) -> dict[str, list[str]]:
    ensure_loss_functions_registered()
    losses_by_property: dict[str, list[str]] = defaultdict(list)
    for loss_name, loss_fn in LOSS_FN.items():
        if not include_special and _is_special_loss(loss_fn):
            continue
        losses_by_property[_loss_property_name(loss_name)].append(loss_name)

    return {
        property_name: sorted(loss_names, key=_natural_sort_key)
        for property_name, loss_names in sorted(
            losses_by_property.items(),
            key=lambda item: _natural_sort_key(item[0]),
        )
    }


def format_available_losses_by_property() -> str:
    lines = ["Available loss functions by property:"]
    for property_name, loss_names in available_losses_by_property().items():
        lines.append(f"{property_name}:")
        lines.extend(f"  - {loss_name}" for loss_name in loss_names)
    return "\n".join(lines)


def format_unknown_loss_error(unknown_loss_names: Iterable[str]) -> str:
    unknown = sorted(set(unknown_loss_names), key=_natural_sort_key)
    unknown_lines = "\n".join(f"  - {loss_name}" for loss_name in unknown)
    return (
        "Unknown loss function(s):\n"
        f"{unknown_lines}\n\n"
        f"{format_available_losses_by_property()}"
    )


def validate_loss_function_names(loss_function_names: Iterable[str]) -> None:
    ensure_loss_functions_registered()
    unknown = [
        loss_name for loss_name in loss_function_names if loss_name not in LOSS_FN
    ]
    if unknown:
        raise ValueError(format_unknown_loss_error(unknown))
