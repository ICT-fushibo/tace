from .v1.tace import TACEV1
from .v1.wrapper import WrapModelV1

__all__ = [
    "TACEV1",
    "WrapModelV1",
]

try:
    from .v2.tace import TACEV2
    from .v2.wrapper import WrapModelV2

    __all__.extend([
        "TACEV2",
        "WrapModelV2",
    ])
except ImportError:
    pass
