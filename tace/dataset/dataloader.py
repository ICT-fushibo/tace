################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import gc
import logging
from typing import Dict, List


from tqdm import tqdm
from hydra.utils import instantiate
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from .element import build_element_lookup, TorchElement
from .read import tace_read_all_files
from .graph import from_atoms
from .statistics import compute_atomic_energy, _compute_statistics, Statistics
from .quantity import KeySpecification


@rank_zero_only
def create_graphs_for_main_rank(atomsList, element, for_dataset, stage):
    dataset = []
    # for atoms in tqdm(atomsList, desc=f"Building graphs for {stage}"):
    #     dataset.append(from_atoms(element, atoms, **for_dataset))
    for atoms in atomsList:
        dataset.append(from_atoms(element, atoms, **for_dataset))
    return dataset


@rank_zero_only
def build_atomsList(
    cfg: dict,
    target_property: List[str],
    embedding_property: List[str],
    keyspec: KeySpecification,
    num_levels: int,
):
    threeAtomsList = tace_read_all_files(cfg, target_property, embedding_property, keyspec)

    # ==== read atomic_numbers and atomic_energy from dataset and cfg ===
    try:
        atomsList = (
            (threeAtomsList[0])
            if threeAtomsList[1] is None
            else threeAtomsList[0] + threeAtomsList[1]
        )
        atomic_numbers_from_dataset = set(
            int(atomic_number)
            for atoms in atomsList
            for atomic_number in atoms.get_atomic_numbers()
        )
    except Exception as e:
        raise RuntimeError(f"Failed to extract atomic numbers from dataset: {e}")

    atomic_numbers = cfg['model']['config'].get("atomic_numbers", None)
    if  atomic_numbers is not None:
        atomic_numbers_from_cfg = set(atomic_numbers)
        assert atomic_numbers_from_dataset.issubset(atomic_numbers_from_cfg), (
            f"cfg.model.config.atomic_numbers must include all atomic numbers present in the dataset, "
            f"but is missing: {atomic_numbers_from_dataset - atomic_numbers_from_cfg}"
        )
        atomic_numbers_from_dataset = atomic_numbers_from_cfg

    # === multi-level atomic_energy ===
    atomic_energies= []
    if "energy" in target_property:

        num_levels = cfg['model']['config'].get("num_levels", 1)
        atomic_energies_cfg = cfg['model']['config'].get("atomic_energies", None)
        if isinstance(atomic_energies_cfg, Dict): 
            atomic_energies_cfg = [atomic_energies_cfg]
        assert atomic_energies_cfg is None or isinstance(atomic_energies_cfg, List)

        if atomic_energies_cfg is not None:
            assert num_levels == len(atomic_energies_cfg)
            for v in atomic_energies_cfg:
                assert isinstance(v, Dict), "If you want to use multi-fidelity or multi-head training, "
                "you must provide each level's atomic energy or set null"
            atomic_numbers_from_energy = set(z for IAE_dict in atomic_energies_cfg for z in IAE_dict.keys())
            atomic_numbers_from_dataset = atomic_numbers_from_dataset | atomic_numbers_from_energy

        element = build_element_lookup(atomic_numbers_from_dataset)

        if atomic_energies_cfg is None:
            logging.info("Computing Isolated Atomic Energies (IAE) automatically for each level")
            atomic_energies = compute_atomic_energy(
                threeAtomsList[0], 
                element, 
                keyspec,
                num_levels
            ) 
        else:
            for idx, energy_cfg in enumerate(atomic_energies_cfg):
                atomic_energy = {int(k): float(v) for k, v in energy_cfg.items()}
                atomic_energy_keys = set(atomic_energy.keys())
                assert atomic_energy_keys.issubset(atomic_numbers_from_dataset), (
                    f"Level {idx}: Keys in atomic_energy must be subset of dataset atomic numbers. "
                    f"Unexpected: {atomic_energy_keys - atomic_numbers_from_dataset}"
                )
                for z in atomic_numbers_from_dataset:
                    if z not in atomic_energy:
                        atomic_energy[z] = 0.0
                        logging.warning(
                            f"Level {idx}: No isolated atomic energy provided for Z={z}, using 0.0 as default."
                        )
                atomic_energies.append(atomic_energy)

        logging.info("Isolated Atomic Energies per computational level:")
        for idx, energy in enumerate(atomic_energies):
            logging.info(f"  Level {idx}: {energy}")
    else:
        element = build_element_lookup(atomic_numbers_from_dataset)
    return element, threeAtomsList, atomic_energies


@rank_zero_only
def compute_statistics(
    cfg: Dict,
    target_property: List[str],
    embedding_property: List[str],
    keyspec: KeySpecification,
    num_levels: int,
    element: TorchElement,
    threeAtomsList,
    atomic_energies: List[Dict[int, float]],
    dataloader_train = None,
):
  
    if dataloader_train is None:
        for_dataset = {
            "cutoff": float(cfg['model']['config'].get("cutoff", 6.0)),
            "max_neighbors": cfg['model']['config'].get("max_neighbors", None),
            "keyspec": keyspec,
            "target_property": target_property,
            "embedding_property": embedding_property,
            "universal_embedding": cfg.get("model", {})
            .get("config", {})
            .get("universal_embedding", None),
            "neighborlist_backend": cfg.get("dataset", {}).get("neighborlist_backend", "matscipy"),
        }
        dataset_train = create_graphs_for_main_rank(threeAtomsList[0], element, for_dataset, 'train')
        dataloader_train = instantiate(
            cfg["dataset"]["train_dataloader"],
            dataset=dataset_train
        )

    statistics = _compute_statistics(
        dataloader_train,
        sorted(element.atomic_numbers),
        atomic_energies,
        target_property=target_property,
        device=cfg.get("misc", {}).get("device", "cpu"),
        num_levels=num_levels,
    )

    del dataloader_train
    gc.collect()
    
    return statistics
