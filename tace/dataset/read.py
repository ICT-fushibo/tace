################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
from typing import List, Dict
from pathlib import Path
import multiprocessing
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_properties


from .split import random_split
from .quantity import KeySpecification


class DatasetsSplit:
    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test


class ThreeDataset:
    def __init__(self, train, valid, test=None):
        self.data = DatasetsSplit(train=train, valid=valid, test=test or [])

    def __getitem__(self, idx):
        return [self.data.train, self.data.valid, self.data.test][idx]

    def __setitem__(self, idx, value):
        if idx == 0:
            self.data.train = value
        elif idx == 1:
            self.data.valid = value
        elif idx == 2:
            self.data.test = value
        else:
            raise IndexError("Index out of range for ThreeDataset")

    def __len__(self):
        return 3

    @property
    def train(self):
        return self.data.train

    @property
    def valid(self):
        return self.data.valid

    @property
    def test(self):
        return self.data.test


def check_keys(
    atomsList: List[Atoms],
    target_property: List[str],
    keyspec: KeySpecification,
    embedding_property: List[str] = [],
    check: bool = True,
):
    if not check:
        return atomsList
    
    for atoms in atomsList:
        if atoms.calc is not None:
            for p in (target_property + embedding_property):
                found = False
                if p in keyspec.info_keys:
                    key = keyspec.info_keys[p]
                    debug_key = f"{key}s"
                    if key in atoms.info.keys():
                        continue
                    if key in all_properties:
                        atoms.info[key] = atoms.calc.results[key]
                        found = True
                    if not found:
                        if debug_key in all_properties:
                            atoms.info[key] = atoms.calc.results[debug_key]
                if p in keyspec.arrays_keys:
                    key = keyspec.arrays_keys[p]
                    debug_key = f"{key}s"
                    if key in atoms.arrays.keys():
                        continue
                    if key in all_properties:
                        atoms.arrays[key] = atoms.calc.results[key]
                        found = True
                    if not found:
                        if debug_key in all_properties:
                            atoms.arrays[key] = atoms.calc.results[debug_key]
    return atomsList


def ase_io_read(filename: str):
    from ase.io import read
    return read(filename, index=":")


def ase_db_connect(filename: str):
    from ase.db import connect
    return [row.toatoms() for row in connect(filename).select()]


def aqcat25_aselmdb(filename: str):
    from ase.db import connect
    atomsList = []
    with connect(filename) as db:
        for idx in range(1, len(db) + 1):
            row = db.get(idx)
            if not row:
                continue
            atoms = row.toatoms()
            atoms.info = row.data 
            atoms.info["level"] = atoms.info.get("level", 1) # default to second level
            # move_mask
            atoms.arrays["move_mask"] = np.array(
                [0 if tag == 0 else 1 for tag in atoms.get_tags()], dtype=np.int64
            )
            # spin_on
            is_spin_off = atoms.info.get("is_spin_off", None)
            if is_spin_off is False:
                atoms.info["spin_on"] = 1
            elif is_spin_off is True: 
                atoms.info["spin_on"] = 0
            else:
                raise ValueError(f"is_spin_off not found or invalid in {filename}, idx={idx}")
            atomsList.append(atoms)
    return atomsList


def fair_aselmdb(filename: str):
    raise NotImplementedError("fair_lmdb is not yet implemented")


def torchsim_h5(filename: str):
    raise NotImplementedError("torchsim_h5 is not yet implemented")


RGLOB = {
    "ase": ["*.xyz", "*.extxyz", "*.traj"],
    "ase_db": ["*.db"],
    "aqcat25_aselmdb": ["*.aselmdb"],
    "fair_aselmdb": ["*.aselmdb"],
    "torchsim_h5": ["*.h5"],
}


HOW_TO_READ = {
    "ase": ase_io_read,
    "ase_db": ase_db_connect,
    "aqcat25_aselmdb": aqcat25_aselmdb,
    "fair_aselmdb": fair_aselmdb,
    "torchsim_h5": torchsim_h5,
}


def read_single_file(fpath: str, target_property, keyspec, embedding_property, backend="ase"):
    atomsList = HOW_TO_READ[backend](fpath)

    try:
        return check_keys(atomsList, target_property, keyspec, embedding_property)
    except Exception as e:
        logging.warning(f"Failed to read {fpath}: {e}, pass")
        return []
    

def read_all_files(
    filename: str | List[str],
    target_property: List[str],
    keyspec,
    embedding_property: List[str],
    num_workers: int = None,
    backend="ase",
):
    """
    Behavior
    --------
    - filename can be:
        * a single file path
        * a single directory path
        * a list of files and/or directories (mixed)
    - Directories are searched recursively for possible files.
    - All discovered files are read and aggregated.

    Parameters
    ----------
    filename : str or List[str]
        File path(s) or directory path(s).
    target_property : List[str]
        List of target properties to check.
    keyspec :
        Specification of property keys.
    embedding_property : List[str]
        List of embedding-related properties.

    Returns
    -------
    list
        Aggregated structures passed to check_keys().
    """

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 4)

    if isinstance(filename, (str, Path)):
        paths = [Path(filename)]
    else:
        paths = [Path(f) for f in filename]

    all_files: List[Path] = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if path.is_file():
            all_files.append(path)

        elif path.is_dir():
            all_files.extend(
                f
                for pattern in RGLOB[backend]
                for f in path.rglob(pattern)
            )

        else:
            raise ValueError(f"Unsupported path type: {path}")

    all_files = sorted(set(all_files))

    if not all_files:
        raise FileNotFoundError("No dataset files found in the provided paths")

    logging.info(f"Found {len(all_files)} files in total")
    logging.info(f"Using {num_workers} processes for parallel reading")

    all_structures = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                read_single_file,
                str(f),
                target_property,
                keyspec,
                embedding_property,
                backend,
            ): f
            for f in all_files
        }

        for future in as_completed(futures):
            all_structures.extend(future.result())

    logging.info(
        f"Successfully read {len(all_structures)} structures from {len(all_files)} files"
    )

    return all_structures


def tace_read_all_files(
    cfg: Dict,
    target_property: List[str],
    embedding_property: List[str],
    keyspec: KeyboardInterrupt,
    in_datamodule: bool = False,
) -> ThreeDataset:
    
    file_type = cfg["dataset"].get("type", "ase")
    no_valid_set = cfg["dataset"].get("no_valid_set", False)
    num_workers = max(1, multiprocessing.cpu_count() // 4)
    num_workers = cfg["dataset"].get("num_workers", num_workers)

    train_file = cfg["dataset"]["train_file"]
    assert train_file, "No valid training dataset provided. Please check cfg.dataset.train_file"
    valid_file = cfg.get("dataset", {}).get("valid_file", None)
    test_files = cfg.get("dataset", {}).get("test_files", None)

    try:
        tmp_train_atoms_list = read_all_files(
            train_file, target_property, keyspec, embedding_property, num_workers, file_type
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load training file from cfg.dataset.train_file: {e}"
        )
    try:
        tmp_valid_atoms_list = (
            read_all_files(
                valid_file, target_property, keyspec, embedding_property, num_workers, file_type
            )
            if valid_file is not None
            else None
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load validation file from cfg.dataset.valid_file: {e}"
        )
    try:
        if test_files is not None:
            if isinstance(test_files, str):
                test_atoms_list = [
                    read_all_files(
                        test_files, target_property, keyspec, embedding_property, num_workers, file_type
                    )
                ]
            elif isinstance(test_files, list):
                test_atoms_list = [
                    read_all_files(
                        f, target_property, keyspec, embedding_property, num_workers, file_type
                    )
                    for f in test_files
                ]
            else:
                test_atoms_list = None
        else:
            test_atoms_list = None
    except Exception as e:
        raise RuntimeError(f"Failed to load test file from cfg.dataset.test_files: {e}")
    
    if not in_datamodule:
        if test_atoms_list is None:
            logging.info("No test file is given")

    if tmp_valid_atoms_list is not None:
        train_atoms_list = tmp_train_atoms_list
        valid_atoms_list = tmp_valid_atoms_list
        if not in_datamodule:
            logging.info(
                f"Using training set from: {train_file}",
            )
            logging.info(
                f"Using validation set from: {valid_file}",
            )
            if test_atoms_list is not None:
                logging.info(
                    f"Using test set from: {test_files}",
                )
    elif cfg.get("dataset", {}).get("valid_from_index", False):
        # In the earlier version, the order of the training set was not taken 
        # into account, and therefore only the valid indices were saved.
        train_index_path = Path(".") / "train.index"
        valid_index_path = Path(".") / "valid.index"
        assert (
            valid_index_path.is_file()
        ), f"File does not exist or is not a regular file: {valid_index_path}"
        with valid_index_path.open("r", encoding="utf-8") as f:
            valid_indices = [int(line.strip()) for line in f if line.strip()]
        valid_atoms_list = [tmp_train_atoms_list[i] for i in valid_indices]
        if train_index_path.exists():
            assert (
                train_index_path.is_file()
            ), f"File does not exist or is not a regular file: {train_index_path}"
            with train_index_path.open("r", encoding="utf-8") as f:
                train_indices = [int(line.strip()) for line in f if line.strip()]
            train_atoms_list = [tmp_train_atoms_list[i] for i in train_indices]
        else:
            train_atoms_list = [
                item
                for idx, item in enumerate(tmp_train_atoms_list)
                if idx not in valid_indices
            ]
        if not in_datamodule:
            logging.info(
                f"Using training set from: {train_file}",
            )
            logging.info(f"Using valid set index from: {str(valid_index_path)}")
            if test_atoms_list is not None:
                logging.info(
                    f"Using test set from: {test_files}",
                )
    elif no_valid_set:
        train_atoms_list = tmp_train_atoms_list
        valid_atoms_list = None
        if not in_datamodule:
            logging.info(
                f"Using training set from: {train_file}",
            )
            logging.warning(
                f"This training has no validation set, you must use lr_scheduler not depending on validation set",
            )
            if test_atoms_list is not None:
                logging.info(
                    f"Using test set from: {test_files}",
                )
    else:
        try:
            ratio = cfg["dataset"]["valid_ratio"]
        except Exception as e:
            raise RuntimeError(
                "Valid_ratio must be provided if no validation file is given."
            ) from e
        assert isinstance(
            ratio, float
        ), "Valid_ratio must be provided if no validation file is given"
        assert 0.0 < ratio < 1.0, "Valid_ratio must be in the range (0, 1)."
        if not in_datamodule:
            logging.info(
                f"Using training set from: {train_file}",
            )
            logging.info(
                "Using random %s%% of training set for validation",
                100 * ratio,
            )
            if test_atoms_list is not None:
                logging.info(
                    f"Using test set from: {test_files}",
                )
        train_atoms_list, valid_atoms_list = random_split(
            tmp_train_atoms_list,
            ratio,
            cfg.get("dataset", {}).get("split_seed", 1),
        )

    assert len(train_atoms_list) > 0, "Training set is empty !"
    if not no_valid_set:
        assert len(valid_atoms_list) > 0, "Validation set is empty !"

    return ThreeDataset(train_atoms_list, valid_atoms_list, test_atoms_list)
