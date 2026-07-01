from __future__ import annotations
import sys
import csv
import gzip
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers, chemical_symbols

from matbench_discovery.diatomics import calc_diatomic_curve, homo_nuc
from matbench_discovery.metrics import diatomics
from matbench_discovery.metrics.diatomics import DiatomicCurves


# ======
# CONFIG
# ======

ELEMENTS = ["H", "Mo"]  # symbols or atomic numbers, e.g. ["H", 8, "Fe"]
NAME = "external_calc"

MIN_DIST = 1.0
MAX_DIST = 6.0
N_POINTS = 119

OUT_JSON = Path("external-diatomics.json.gz")
OUT_CSV = Path("diatomic-summary.csv")
PLOT_DIR = Path("diatomic-plots")
REF_FUNCTIONAL = "PBE"  # "PBE" or "r2SCAN"


def get_calculator() -> Calculator:
    from tace.interface.ase import TACEAseCalc
    return TACEAseCalc(sys.argv[1])























def normalize_elements(tokens: list[str | int]) -> list[str]:
    elems: list[str] = []
    for token in tokens:
        if isinstance(token, int) or str(token).isdigit():
            z_num = int(token)
            if not 1 <= z_num < len(chemical_symbols):
                raise ValueError(f"Invalid atomic number: {token}")
            elem = chemical_symbols[z_num]
        else:
            elem = str(token).capitalize()
            if elem not in atomic_numbers:
                raise ValueError(f"Invalid element symbol: {token}")
        if elem not in elems:
            elems.append(elem)
    return elems


def is_valid_curve(curve: dict[str, Any]) -> bool:
    return bool(
        curve.get("energies")
        and curve.get("forces")
        and np.isfinite(curve["energies"]).all()
        and np.isfinite(curve["forces"]).all()
    )


def save_json(curves: dict[str, Any], distances: np.ndarray, metadata: dict[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    results = {
        homo_nuc: curves,
        "hetero-nuclear": {},
        "distances": distances.tolist(),
        "run_metadata": metadata,
    }
    with gzip.open(OUT_JSON, mode="wt") as file:
        json.dump(results, file, allow_nan=False, default=lambda arr: arr.tolist())


def calc_metrics(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    pred_curves = DiatomicCurves.from_dict(data)
    ref_curves = diatomics.load_dft_reference_curves(REF_FUNCTIONAL)
    return diatomics.calc_diatomic_metrics(
        ref_curves=ref_curves,
        pred_curves=pred_curves,
        interpolate=200,
    )


def save_summary(
    rows: list[dict[str, Any]],
    metrics: dict[str, dict[str, float]],
) -> None:
    for row in rows:
        row.update(metrics.get(row["element"], {}))

    metric_keys = sorted({key for vals in metrics.values() for key in vals})
    fieldnames = [
        "Z",
        "element",
        "formula",
        "status",
        "n_energy",
        "n_force",
        *metric_keys,
    ]

    with OUT_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(data: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    pred_curves = DiatomicCurves.from_dict(data)
    ref_curves = diatomics.load_dft_reference_curves(REF_FUNCTIONAL)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for elem, pred in pred_curves.homo_nuclear.items():
        z_num = atomic_numbers[elem]
        ref = ref_curves.homo_nuclear.get(elem)

        pred_energy = np.asarray(pred.energies, dtype=float)
        pred_energy = pred_energy - pred_energy[-1]
        pred_force = np.asarray(pred.forces, dtype=float)[:, 0, 0]

        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        axes[0].plot(pred.distances, pred_energy, label=NAME, lw=2)
        axes[1].plot(pred.distances, pred_force, label=NAME, lw=2)

        if ref is not None:
            ref_energy = np.asarray(ref.energies, dtype=float)
            ref_energy = ref_energy - ref_energy[-1]
            axes[0].plot(ref.distances, ref_energy, "--", label=REF_FUNCTIONAL, lw=2)

            if len(ref.forces):
                ref_force = np.asarray(ref.forces, dtype=float)[:, 0, 0]
                axes[1].plot(ref.distances, ref_force, "--", label=REF_FUNCTIONAL, lw=2)

        axes[0].set_title(f"{elem}-{elem}  Z={z_num}")
        axes[0].set_ylabel("energy - E_far (eV)")
        axes[1].set_ylabel("force atom0 x (eV/Angstrom)")
        axes[1].set_xlabel("distance (Angstrom)")
        axes[0].legend()
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"Z{z_num:03d}-{elem}.png", dpi=180)
        plt.close(fig)


def print_worst(rows: list[dict[str, Any]]) -> None:
    for key in ("pbe_energy_mae", "pbe_force_mae", "energy_jump", "force_jump"):
        ranked = []
        for row in rows:
            val = row.get(key)
            if val is None:
                continue
            val = float(val)
            if np.isfinite(val):
                ranked.append((val, row))
        if not ranked:
            continue
        ranked.sort(key=lambda item: item[0], reverse=True)
        print(f"\nWorst by {key}:")
        for val, row in ranked[:10]:
            print(f"  {row['element']:>2} Z={int(row['Z']):03d}: {val:.6g}")


def main() -> None:
    elems = normalize_elements(ELEMENTS)
    pairs = [(elem, elem) for elem in elems]
    formulas = [f"{elem}-{elem}" for elem in elems]
    distances = np.geomspace(MIN_DIST, MAX_DIST, N_POINTS)

    calc = get_calculator()
    if not isinstance(calc, Calculator):
        raise TypeError(f"get_calculator() returned {type(calc).__name__}, not Calculator")

    start = time.perf_counter()
    raw_curves: dict[str, Any] = {}
    calc_diatomic_curve(
        pairs=pairs,
        calculator=calc,
        model_name=NAME,
        distances=distances,
        results=raw_curves,
    )
    run_time_sec = round(time.perf_counter() - start, 2)

    curves = {formula: curve for formula, curve in raw_curves.items() if is_valid_curve(curve)}
    rows = []
    for elem, formula in zip(elems, formulas, strict=True):
        curve = raw_curves.get(formula, {})
        rows.append(
            {
                "Z": atomic_numbers[elem],
                "element": elem,
                "formula": formula,
                "status": "ok" if formula in curves else "empty_or_failed",
                "n_energy": len(curve.get("energies", [])),
                "n_force": len(curve.get("forces", [])),
            }
        )

    data = {homo_nuc: curves, "hetero-nuclear": {}, "distances": distances.tolist()}
    metadata = {
        "name": NAME,
        "elements": elems,
        "run_time_sec": run_time_sec,
    }

    save_json(curves, distances, metadata)
    metrics = calc_metrics(data) if curves else {}
    save_summary(rows, metrics)
    if curves:
        plot_curves(data)

    print(f"\nWrote {len(curves)}/{len(pairs)} curves to {OUT_JSON}")
    print(f"Wrote summary to {OUT_CSV}")
    if curves:
        print(f"Wrote plots to {PLOT_DIR}")

    failed = [row for row in rows if row["status"] != "ok"]
    if failed:
        print("\nEmpty/failed curves:")
        print(", ".join(f"{row['element']}(Z={row['Z']})" for row in failed))

    for row in rows:
        row.update(metrics.get(row["element"], {}))
    print_worst(rows)


if __name__ == "__main__":
    main()
