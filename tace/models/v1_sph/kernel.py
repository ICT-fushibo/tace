import dataclasses

try:
    import cuequivariance as cueq
    import cuequivariance_torch as cueqt
    CUET_AVAILABLE = True
except Exception:
    CUET_AVAILABLE = False

try:
    import openequivariance as opeq
    OPEQ_AVAILABLE = True
except Exception:
    OPEQ_AVAILABLE = False


@dataclasses.dataclass
class CuEquivarianceConfig:
    pass


@dataclasses.dataclass
class OpenEquivarianceConfig:
    pass

