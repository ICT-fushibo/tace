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

# try:
#     from .v1_sph.tace import SphTACEV1
#     from .v1_sph.wrapper import WrapSphModelV1
#     __all__.extend([
#         "SphTACEV1",
#         "WrapSphModelV1",
#     ])
# except ImportError:
#     pass


from .v1_sph.tace import SphTACEV1
from .v1_sph.wrapper import WrapSphModelV1
__all__.extend([
    "SphTACEV1",
    "WrapSphModelV1",
])
