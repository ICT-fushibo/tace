################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import logging
from pathlib import Path


import hydra
from omegaconf import DictConfig


from .train import build

@hydra.main(version_base="1.3", config_path=str(Path.cwd()), config_name="tace")
def main(cfg: DictConfig):
    _, _, _, _, datamodule = build(cfg)
    datamodule.prepare_data()
    datamodule.setup('fit')
    logging.info(f"Finished building lmdb dataset")

if __name__ == "__main__":
    main()


