"""Paso 2: auditoría por registro (canales, frecuencia, eventos, canales planos)."""

import argparse
import json
from collections import Counter
from pathlib import Path

import _common  # noqa: F401
import pandas as pd

from eeg_mdd.audit import audit_all
from eeg_mdd.checks import check_audit
from eeg_mdd.config import ensure_dirs, load_config
from eeg_mdd.io import find_eeg_files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    ds, pp = cfg["dataset"], cfg["preprocessing"]

    files = find_eeg_files(cfg["paths"]["raw_dir"], ds["eeg_extensions"])
    print(f"Auditando {len(files)} registros...")
    audit = audit_all(files, ds["subject_id_regex"], pp.get("stim_channel"))
    out = Path(cfg["paths"]["interim_dir"]) / "audit.csv"
    audit.to_csv(out, index=False)
    print(f"Auditoría guardada en {out}")

    # Códigos de evento en todo el dataset: base para llenar event_id.
    totals: Counter = Counter()
    for raw_counts in audit.get("event_counts", []).dropna():
        totals.update({int(k): v for k, v in json.loads(raw_counts).items()})
    print("\nCódigos de evento encontrados (código: total):")
    for code, n in sorted(totals.items()):
        print(f"  {code}: {n}")

    egi_totals: Counter = Counter()
    for raw_counts in audit.get("egi_event_counts", pd.Series(dtype=str)).dropna():
        egi_totals.update(json.loads(raw_counts))
    if egi_totals:
        print("\nEventos de EGI por nombre (nombre: total en todos los archivos):")
        for name, n in sorted(egi_totals.items()):
            print(f"  {name}: {n}")

    problems = check_audit(audit, ds.get("expected_n_eeg_channels"), ds.get("expected_sfreq"),
                           allowed_flat=ds.get("allowed_flat_channels", []))
    print(f"\nProblemas detectados: {len(problems)}")
    for p in problems:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
