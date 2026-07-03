from pathlib import Path
import numpy as np

def fair_aselmdb(filename: str):  # [only test energy, forces, stress]
    from ase.db import connect
    atoms_list = []
    with connect(filename) as db:
        idx = 0
        for idx, row in enumerate(db.select()):
            atoms = row.toatoms()
            atoms.info = row.data
            atoms.arrays["move_mask"] = np.array([0 if tag == 0 else 1 for tag in atoms.get_tags()], dtype=np.int64)
            is_spin_off = atoms.info.get("is_spin_off", None)

            if is_spin_off is False:
                atoms.info["fidelity_idx"] = 0
            elif is_spin_off is True: 
                atoms.info["fidelity_idx"] = 1
            else:
                raise

            del atoms.info["is_rattled"]
            del atoms.info["is_md"]
            del atoms.info["is_rerun"]
            del atoms.info["fid"]
            del atoms.info["sid"]
            del atoms.arrays["tags"]
            atoms_list.append(atoms)
            idx += 1
            if idx > 10:
                break
    return atoms_list

from ase.io import write

for source_dir in ["is_slabs", "test_id_v2", "train_id_v2", "val_id_v2"]:
    source_path = Path(source_dir)
    output_dir = Path(f"{source_dir}_xyz")
    for input_file in source_path.rglob("*.aselmdb"):
        output_file = output_dir / input_file.relative_to(source_path).with_suffix(".xyz")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        atoms_list = fair_aselmdb(str(input_file))
        write(str(output_file), atoms_list, format="extxyz")
        print(f"{input_file} -> {output_file}")