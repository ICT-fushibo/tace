################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from pathlib import Path
from string import ascii_letters

import packaging
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


################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from pathlib import Path
from string import ascii_letters

import packaging
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
    "16": torch.float16,
    "32": torch.float32,
    "64": torch.float64,
    "half": torch.float16,
    "single": torch.float32,
    "double": torch.float64,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    # Lightning true precision modes: model parameters use this dtype.
    "16-true": torch.float16,
    "bf16-true": torch.bfloat16,
    "32-true": torch.float32,
    "64-true": torch.float64,
    # Lightning mixed precision keeps model parameters in float32.
    "16-mixed": torch.float32,
    "bf16-mixed": torch.float32,
    # Transformer Engine weight/fallback dtypes used by Lightning.
    "transformer-engine": torch.bfloat16,
    "transformer-engine-float16": torch.float16,
    torch.float16: torch.float16,
    torch.bfloat16: torch.bfloat16,
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
