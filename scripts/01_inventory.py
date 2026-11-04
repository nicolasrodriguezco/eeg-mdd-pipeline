"""Paso 1: inventario de los datos crudos (no modifica nada)."""

import argparse
from pathlib import Path

import _common  # noqa: F401

from eeg_mdd.config import ensure_dirs, load_config
from eeg_mdd.inventory import build_inventory, summarize_inventory


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    ds = cfg["dataset"]

    inv = build_inventory(cfg["paths"]["raw_dir"], ds["subject_id_regex"], ds["eeg_extensions"])
    out = Path(cfg["paths"]["interim_dir"]) / "inventory.csv"
    inv.to_csv(out, index=False)

    print(f"Archivos inventariados: {len(inv)}  ->  {out}")
    print(summarize_inventory(inv).to_string(index=False))
    eeg = inv[inv["is_eeg"]]
    print(f"\nArchivos de EEG: {len(eeg)}")
    print(f"Sin ID detectado: {eeg['subject_id_guess'].isna().sum()} (revísalos y ajusta subject_id_regex)")
    print(f"IDs repetidos: {eeg['subject_id_guess'].dropna().duplicated().sum()}")


if __name__ == "__main__":
    main()
