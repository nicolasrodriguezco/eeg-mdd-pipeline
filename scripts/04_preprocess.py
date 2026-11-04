"""Paso 4: preprocesamiento por sujeto y guardado de épocas en .fif."""

import argparse
from pathlib import Path

import _common  # noqa: F401
import pandas as pd

from eeg_mdd.checks import check_labels, raise_if_problems
from eeg_mdd.config import ensure_dirs, load_config
from eeg_mdd.io import load_raw
from eeg_mdd.preprocess import load_ica_exclusions, preprocess_raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--subject", default=None, help="Procesar solo este sujeto (útil para revisar)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    paths, pp = cfg["paths"], cfg["preprocessing"]
    interim, processed, reports = Path(paths["interim_dir"]), Path(paths["processed_dir"]), Path(paths["reports_dir"])

    subjects = pd.read_csv(interim / "subjects.csv", dtype={"eeg_id": str})
    raise_if_problems(check_labels(subjects), "etiquetas")
    audit = pd.read_csv(interim / "audit.csv", dtype={"subject_id": str})
    file_by_id = dict(zip(audit["subject_id"], audit["file"]))
    ica_excl = load_ica_exclusions(paths["ica_exclusions"])

    if args.subject:
        subjects = subjects[subjects["eeg_id"] == args.subject]

    logs = []
    for _, s in subjects.iterrows():
        sid = s["eeg_id"]
        print(f"Procesando {sid} ({s['group']})...")
        try:
            raw = load_raw(file_by_id[sid])
            epochs, log = preprocess_raw(raw, pp, sid, ica_excl.get(sid), ica_dir=processed / "ica")
            epochs.save(processed / f"sub-{sid}-epo.fif", overwrite=True, verbose="error")
            log["group"] = s["group"]
            log["error"] = ""
        except Exception as exc:  # noqa: BLE001
            log = {"subject_id": sid, "group": s["group"], "error": f"{type(exc).__name__}: {exc}"}
        logs.append(log)

    out = reports / "preprocessing_log.csv"
    pd.DataFrame(logs).to_csv(out, index=False)
    print(f"\nRegistro guardado en {out}")


if __name__ == "__main__":
    main()
