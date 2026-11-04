"""Paso 3: cruce de EEG con datos clínicos, sin arreglos automáticos."""

import argparse
from pathlib import Path

import _common  # noqa: F401
import pandas as pd

from eeg_mdd.checks import check_labels
from eeg_mdd.config import ensure_dirs, load_config
from eeg_mdd.matching import load_clinical, load_manual_map, match_subjects, usable_subjects


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    clin_cfg = cfg["dataset"]["clinical"]
    interim = Path(cfg["paths"]["interim_dir"])

    audit = pd.read_csv(interim / "audit.csv", dtype={"subject_id": str, "error": str}, keep_default_na=False)
    eeg_ids = audit.loc[audit["error"] == "", "subject_id"].replace("", None).dropna().tolist()

    clinical = load_clinical(clin_cfg["file"], clin_cfg["id_column"], clin_cfg["group_column"], clin_cfg["group_map"])
    manual = load_manual_map(cfg["paths"]["manual_id_map"])
    matches = match_subjects(eeg_ids, clinical, manual)
    matches.to_csv(interim / "id_matching.csv", index=False)

    print("Resultado del cruce:")
    print(matches["status"].value_counts().reindex(matches["status"].cat.categories, fill_value=0).to_string())

    pending = matches[~matches["status"].isin(["matched", "matched_manual"])]
    if len(pending):
        print(f"\n{len(pending)} caso(s) para revisar a mano (ver id_matching.csv):")
        print(pending.to_string(index=False))

    subjects = usable_subjects(matches)
    problems = check_labels(subjects)
    if problems:
        print("\nProblemas de etiquetas:")
        for p in problems:
            print(f"  - {p}")
    subjects.to_csv(interim / "subjects.csv", index=False)
    print(f"\nSujetos usables: {len(subjects)}  (MDD={int((subjects['group'] == 'MDD').sum())}, "
          f"HC={int((subjects['group'] == 'HC').sum())})  ->  {interim / 'subjects.csv'}")


if __name__ == "__main__":
    main()
