################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import os

TACE_USE_OEQ = os.environ.get('TACE_USE_OEQ', '0')
TACE_USE_CUEQ = os.environ.get('TACE_USE_CUEQ', '0')
TACE_WEIGHT_INIT = os.environ.get('TACE_WEIGHT_INIT', 'randn') 