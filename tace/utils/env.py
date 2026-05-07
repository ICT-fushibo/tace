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

TACE_USE_OEQ = os.environ.get('TACE_USE_OEQ', '0')
TACE_USE_CUE = os.environ.get('TACE_USE_CUE', '0')
TACE_USE_EQT = os.environ.get('TACE_USE_EQT', '0')
TACE_APPLY_U_SHIFT = os.environ.get('TACE_APPLY_U_SHIFT', '0') 
TACE_USE_DENS = os.environ.get('TACE_USE_DENS', '0')