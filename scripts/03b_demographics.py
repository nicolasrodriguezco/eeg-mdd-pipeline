"""Paso 3b: tabla demográfica por grupo y posibles confusores de registro.

Salidas:
  reports/tables/demographics.csv     edad, sexo, escolaridad (+ cuestionarios, solo descriptivos)
  reports/tables/recording_dates.csv  fecha de registro por sujeto (nombre de archivo y cabecera)
"""

import argparse
import json
from pathlib import Path

import _common  # noqa: F401
import pandas as pd

from eeg_mdd.config import load_config
from eeg_mdd.demographics import demographic_table, recording_datetime_from_name
from eeg_mdd.io import load_raw
from eeg_mdd.matching import load_clinical

DEMOGRAPHIC = {"age": "age", "education": "education（years）"}
QUESTIONNAIRES = ["PHQ-9", "GAD-7", "CTQ-SF", "LES", "SSRS", "PSQI"]   # SOLO descriptivos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    clin_cfg = cfg["dataset"]["clinical"]
    interim = Path(cfg["paths"]["interim_dir"])
    tables = Path(cfg["paths"]["reports_dir"]) / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    subjects = pd.read_csv(interim / "subjects.csv", dtype={"eeg_id": str, "clinical_id": str})
    clinical = load_clinical(clin_cfg["file"], clin_cfg["id_column"], clin_cfg["group_column"], clin_cfg["group_map"])
    clinical = clinical.drop(columns="group")
    df = subjects.merge(clinical, on="clinical_id", how="left", validate="one_to_one")
    df = df.rename(columns={v: k for k, v in DEMOGRAPHIC.items()})

    demo = demographic_table(df, ["age", "education"] + QUESTIONNAIRES, ["gender"])
    demo["role"] = demo["variable"].map(lambda v: "questionnaire (descriptive only)" if v in QUESTIONNAIRES
                                        else "demographic (confounder analysis)")
    demo.to_csv(tables / "demographics.csv", index=False)
    cols = ["variable", "MDD_n", "MDD_mean_sd", "MDD_counts", "HC_n", "HC_mean_sd", "HC_counts", "test", "p_value"]
    print(demo[[c for c in cols if c in demo.columns]].to_string(index=False))

    # Fecha/hora de registro: el grupo podría estar confundido con la época de grabación.
    audit = pd.read_csv(interim / "audit.csv", dtype={"subject_id": str})
    rows = []
    for _, r in df.iterrows():
        f = audit.loc[audit["subject_id"] == r["eeg_id"], "file"].item()
        date, hhmm = recording_datetime_from_name(f)
        raw = load_raw(f)
        counts = json.loads(audit.loc[audit["subject_id"] == r["eeg_id"], "egi_event_counts"].item())
        rows.append({
            "subject_id": r["eeg_id"], "group": r["group"], "file": Path(f).name,
            "date_from_name": date.date() if date is not None else None, "time_from_name": hhmm,
            "meas_date_header": raw.info["meas_date"],
            "n_resp": counts.get("resp", 0),
            "extra_event_channels": ",".join(sorted(k for k in counts if k[:2] in ("DI", "D1", "D2", "Ca"))),
        })
    rec = pd.DataFrame(rows)
    rec.to_csv(tables / "recording_dates.csv", index=False)
    rec["month"] = pd.to_datetime(rec["date_from_name"]).dt.to_period("M")
    print("\nSujetos por mes de registro y grupo (fecha del nombre de archivo):")
    print(pd.crosstab(rec["month"], rec["group"]).to_string())
    print(f"\nTablas en {tables}")


if __name__ == "__main__":
    main()
