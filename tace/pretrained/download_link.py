################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
# TODO, The model has not been released yet. We are just writing the script for now

import logging
import urllib.request
from pathlib import Path
from collections.abc import Mapping

from ..utils._global import CACHE_DIR

OAM_SERIES = {
    'TACE-v1-OMat24-M': 'https://github.com/xvzemin/tace',
    'TACE-v1-OAM-M': 'https://github.com/xvzemin/tace',
}

REICO_SERIES = {
    'TACE-v1-PdAgCHO': 'https://github.com/xvzemin/tace'
}



class CachedModelRegistry(Mapping):
    def __init__(self, registry: dict[str, str], cache_dir=None):
        self._registry = registry
 
    def __getitem__(self, key: str) -> Path:
        if key not in self._registry:
            logging.error(f"Unknown pretrained model: {key}")
            self.print_models()
            raise KeyError(key)

        url = self._registry[key]
        model_path = self.cache_dir / key

        if not model_path.exists():
            logging.info(f"Downloading {key} from {url}")
            self._download(url, model_path)

        return model_path

    def __iter__(self):
        return iter(self._registry)

    def __len__(self):
        return len(self._registry)

    def _download(self, url: str, target: Path):
        urllib.request.urlretrieve(url, target)

    def list_models(self) -> list[str]:
        return sorted(self._registry.keys())

    def print_models(self):
        logging.info("Available pretrained models:")
        for name in self.list_models():
            logging.info(f"  - {name}")

uMLIPs = CachedModelRegistry(
    OAM_SERIES 
    | REICO_SERIES
)

