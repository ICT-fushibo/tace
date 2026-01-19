# import tomli
# import importlib.resources

# def get_version():
#     with importlib.resources.open_text("tace", "pyproject.toml") as f:
#         pyproject = tomli.load(f)
#     return pyproject["project"]["version"]

# __version__ = get_version()

from torch.serialization import add_safe_globals
from tace.dataset.statistics import Statistics
add_safe_globals([Statistics])

__version__ = "0.0.9"

