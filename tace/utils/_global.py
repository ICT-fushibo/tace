################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import packaging
from string import ascii_letters
from pathlib import Path

import torch

CACHE_DIR = Path.home() / ".cache" / "tace"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


BOOL = {
    0: False,
    1: True,
    "f": False,
    "t": True,
    "false": False,
    "true": True,
    "False": False,
    "True": True,
}


DTYPE = {
    None: None,
    16: torch.float16,
    32: torch.float32,
    64: torch.float64,
    "half": torch.float16,
    "single": torch.float32,
    "double": torch.float64,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    torch.float16: torch.float16,
    torch.float32: torch.float32,
    torch.float64: torch.float64,
}


num_gpus = 32
DEVICE = {
    None: None,
    "cpu": torch.device("cpu"),
    "cuda": torch.device("cuda"),
    torch.device("cpu"): torch.device("cpu"),
    torch.device("cuda"): torch.device("cuda"),
    **{i: torch.device(f"cuda:{i}") for i in range(num_gpus)},
    **{f"cuda:{i}": torch.device(f"cuda:{i}") for i in range(num_gpus)},
    **{torch.device(f"cuda:{i}"): torch.device(f"cuda:{i}") for i in range(num_gpus)},
}

LETTERS = list(ascii_letters)[3:]


# _TORCH_VERSION = packaging.version.parse(torch.__version__)
# _TORCH_GE_2_9 = _TORCH_VERSION >= packaging.version.parse("2.9")
# _GLOBAL_STATE_INITIALIZED = False

