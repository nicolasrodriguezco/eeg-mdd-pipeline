"""Paso 5: comparar el dataset viejo (tabla ancha) con el nuevo."""

import argparse
from pathlib import Path

import _common  # noqa: F401
import pandas as pd

from eeg_mdd.compare import compare_old_new, counts_from_old
from eeg_mdd.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--old", required=True, help="Ruta al dataset viejo (.parquet o .csv)")
    ap.add_argument("--subject-col", default="Subject")
    ap.add_argument("--group-col", default="Group")
    ap.add_argument("--condition-col", default="Condition")
    args = ap.parse_args()

    cfg = load_config(args.config)
    interim, reports = Path(cfg["paths"]["interim_dir"]), Path(cfg["paths"]["reports_dir"])

    cols = [args.subject_col, args.group_col, args.condition_col]
    old_path = Path(args.old)
    # Solo se leen 3 columnas: la tabla vieja tiene ~25.000.
    old = pd.read_parquet(old_path, columns=cols) if old_path.suffix == ".parquet" else pd.read_csv(old_path, usecols=cols)

    old_counts = counts_from_old(old, *cols)
    new_log = pd.read_csv(reports / "preprocessing_log.csv", dtype={"subject_id": str})
    subjects = pd.read_csv(interim / "subjects.csv", dtype={"eeg_id": str})

    result = compare_old_new(old_counts, new_log, subjects)
    out = reports / "old_vs_new.csv"
    result.to_csv(out, index=False)

    flagged = result[result["issues"] != ""]
    print(f"Sujetos comparados: {len(result)}  |  con diferencias: {len(flagged)}  ->  {out}")
    if len(flagged):
        print(flagged[["subject_id", "old_group", "new_group", "issues"]].to_string(index=False))


if __name__ == "__main__":
    main()
