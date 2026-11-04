"""Validaciones que deben pasar antes de avanzar de fase.

Cada función devuelve una lista de problemas en texto. Lista vacía = todo bien.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

VALID_GROUPS = {"MDD", "HC"}


def _text(value) -> str:
    """Texto vacío para NaN/None (al leer un CSV, una celda vacía llega como NaN)."""
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


def check_audit(audit: pd.DataFrame, expected_n_eeg: int | None, expected_sfreq: float | None,
                allowed_flat: Iterable[str] = ()) -> list[str]:
    """``allowed_flat``: canales que se espera que estén planos (p. ej. la referencia
    de registro de EGI, E129/VREF, que vale cero hasta re-referenciar)."""
    problems: list[str] = []
    allowed = set(allowed_flat)
    for _, r in audit.iterrows():
        sid = _text(r.get("subject_id")) or r["file"]
        if _text(r.get("error")):
            problems.append(f"{sid}: no se pudo leer ({r['error']})")
            continue
        if not r.get("subject_id"):
            problems.append(f"{r['file']}: no se pudo extraer el ID del sujeto del nombre")
        if expected_n_eeg and r.get("n_eeg_channels") != expected_n_eeg:
            problems.append(f"{sid}: {r.get('n_eeg_channels')} canales EEG (se esperaban {expected_n_eeg})")
        if expected_sfreq and float(r.get("sfreq", 0)) != float(expected_sfreq):
            problems.append(f"{sid}: frecuencia {r.get('sfreq')} Hz (se esperaba {expected_sfreq})")
        if r.get("n_events", 0) == 0:
            problems.append(f"{sid}: sin eventos")
        flat = [c for c in _text(r.get("flat_channels_first60s")).split(",") if c and c not in allowed]
        if flat:
            problems.append(f"{sid}: {len(flat)} canales planos ({','.join(flat)})")
    ids = audit["subject_id"].dropna()
    for dup in sorted(ids[ids.duplicated()].unique()):
        problems.append(f"{dup}: aparece en más de un archivo")
    return problems


def check_labels(subjects: pd.DataFrame) -> list[str]:
    """Cada sujeto usable tiene exactamente una etiqueta válida."""
    problems: list[str] = []
    if subjects["eeg_id"].duplicated().any():
        dups = subjects.loc[subjects["eeg_id"].duplicated(), "eeg_id"].tolist()
        problems.append(f"Sujetos repetidos: {dups}")
    bad = subjects[~subjects["group"].isin(VALID_GROUPS)]
    for _, r in bad.iterrows():
        problems.append(f"{r['eeg_id']}: etiqueta inválida ({r['group']!r})")
    return problems


def check_no_subject_overlap(train_ids: Iterable[str], test_ids: Iterable[str]) -> list[str]:
    """Ningún sujeto puede estar en entrenamiento y prueba a la vez."""
    overlap = sorted(set(train_ids) & set(test_ids))
    return [f"Sujetos en entrenamiento y prueba a la vez: {overlap}"] if overlap else []


def raise_if_problems(problems: list[str], step: str) -> None:
    if problems:
        detail = "\n  - ".join(problems)
        raise RuntimeError(f"[{step}] {len(problems)} problema(s):\n  - {detail}")
