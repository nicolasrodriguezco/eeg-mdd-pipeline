"""Cruce entre los registros de EEG y los datos clínicos.

Principio: NO hay coincidencias aproximadas ni "normalizaciones" creativas.
Un ID solo se une si es idéntico (después de quitar espacios) o si aparece en
el mapa manual con una justificación escrita.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

STATUS_ORDER = ["matched", "matched_manual", "only_eeg", "only_clinical", "duplicated", "unlabeled"]


def normalize_id(value) -> str:
    """Única normalización permitida: convertir a texto y quitar espacios."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    # Excel suele convertir IDs numéricos a float (por ejemplo 2010002.0).
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def load_clinical(path: str | Path, id_column: str, group_column: str, group_map: dict) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str)
    missing = {id_column, group_column} - set(df.columns)
    if missing:
        raise KeyError(f"Columnas no encontradas en el archivo clínico: {missing}. Hay: {list(df.columns)}")
    out = df.copy()
    out["clinical_id"] = out[id_column].map(normalize_id)
    out["group"] = out[group_column].map(lambda v: group_map.get(str(v).strip()) if pd.notna(v) else None)
    return out


def load_manual_map(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}
    mm = pd.read_csv(path, dtype=str).fillna("")
    if mm.empty:
        return {}
    bad = mm[mm["justification"].str.strip() == ""]
    if not bad.empty:
        raise ValueError(
            "Hay cruces manuales sin justificación. Cada fila de manual_id_map.csv "
            f"necesita una explicación:\n{bad.to_string(index=False)}"
        )
    return {normalize_id(e): normalize_id(c) for e, c in zip(mm["eeg_id"], mm["clinical_id"])}


def match_subjects(eeg_ids: list[str], clinical: pd.DataFrame, manual_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Una fila por ID con su estado de cruce y su etiqueta.

    Estados:
      matched / matched_manual  -> unido y con etiqueta válida
      only_eeg                  -> hay EEG pero no datos clínicos
      only_clinical             -> hay datos clínicos pero no EEG
      duplicated                -> el ID aparece más de una vez (EEG o clínico)
      unlabeled                 -> unido pero sin etiqueta MDD/HC válida
    """
    manual_map = manual_map or {}
    eeg_norm = [normalize_id(i) for i in eeg_ids if normalize_id(i)]
    eeg_counts = pd.Series(eeg_norm, dtype=str).value_counts()
    clin_counts = clinical["clinical_id"].value_counts()
    clin_by_id = clinical.drop_duplicates("clinical_id").set_index("clinical_id")

    rows = []
    used_clinical: set[str] = set()
    for eid in sorted(set(eeg_norm)):
        target, how = (manual_map[eid], "matched_manual") if eid in manual_map else (eid, "matched")
        row = {"eeg_id": eid, "clinical_id": target, "group": None}
        if eeg_counts.get(eid, 0) > 1 or clin_counts.get(target, 0) > 1:
            row["status"] = "duplicated"
        elif target not in clin_by_id.index:
            row["status"] = "only_eeg"
            row["clinical_id"] = None
        else:
            group = clin_by_id.loc[target, "group"]
            row["group"] = group if isinstance(group, str) else None
            row["status"] = how if row["group"] in ("MDD", "HC") else "unlabeled"
        if row["clinical_id"]:
            used_clinical.add(row["clinical_id"])
        rows.append(row)

    for cid in sorted(set(clinical["clinical_id"]) - used_clinical - {""}):
        rows.append({"eeg_id": None, "clinical_id": cid, "group": None, "status": "only_clinical"})

    out = pd.DataFrame(rows, columns=["eeg_id", "clinical_id", "group", "status"])
    out["status"] = pd.Categorical(out["status"], categories=STATUS_ORDER, ordered=True)
    return out.sort_values(["status", "eeg_id"]).reset_index(drop=True)


def usable_subjects(matches: pd.DataFrame) -> pd.DataFrame:
    """Solo los sujetos que pueden pasar al preprocesamiento."""
    return matches[matches["status"].isin(["matched", "matched_manual"])].copy()
