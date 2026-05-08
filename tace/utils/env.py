################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import os
from typing import Dict

def set_env(cfg: Dict):
    env = cfg.get("misc", {}).get("env", {})
    for k, v in env.items():
        os.environ[k] = v


def get_tace_use_oeq():
    return os.environ.get("TACE_USE_OEQ", "0")


def get_tace_use_cue():
    return os.environ.get("TACE_USE_CUE", "0")


def get_tace_use_eqt():
    return os.environ.get("TACE_USE_EQT", "0")


def get_tace_apply_u_shift():
    return os.environ.get("TACE_APPLY_U_SHIFT", "0")


def get_tace_use_dens():
    return os.environ.get("TACE_USE_DENS", "0")