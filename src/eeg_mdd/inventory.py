"""Inventario de los datos crudos, sin modificar nada."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .io import extract_subject_id


def _size_mb(p: Path) -> float:
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
    return p.stat().st_size / 1e6


def build_inventory(raw_dir: str | Path, subject_id_regex: str, eeg_extensions: list[str]) -> pd.DataFrame:
    """Una fila por archivo (o carpeta .mff) con tamaño, extensión e ID detectado."""
    raw_dir = Path(raw_dir)
    eeg_exts = {e.lower() for e in eeg_extensions}
    rows = []
    for p in sorted(raw_dir.rglob("*")):
        if any(parent.suffix.lower() == ".mff" for parent in p.parents):
            continue  # el contenido de una carpeta .mff se cuenta como un solo registro
        if p.is_dir() and p.suffix.lower() != ".mff":
            continue
        ext = p.suffix.lower()
        sid = extract_subject_id(p, subject_id_regex)
        # Lo que queda del nombre sin el ID: sirve para ver variantes como
        # "erp", "erp_new", "erpnew" o nombres sin sufijo (posibles re-grabaciones).
        rest = p.stem.replace(sid, "", 1).strip(" _-") if sid else p.stem
        rows.append(
            {
                "relative_path": str(p.relative_to(raw_dir)),
                "extension": ext or "(sin extensión)",
                "is_eeg": ext in eeg_exts,
                "size_mb": round(_size_mb(p), 3),
                "subject_id_guess": sid,
                "name_rest": rest,
                "possible_rerecording": "new" in rest.lower(),
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return pd.DataFrame(rows)


def summarize_inventory(inv: pd.DataFrame) -> pd.DataFrame:
    """Resumen por extensión: cuántos archivos y cuántos MB."""
    if inv.empty:
        return pd.DataFrame(columns=["extension", "n_files", "total_mb"])
    return (
        inv.groupby("extension")
        .agg(n_files=("relative_path", "count"), total_mb=("size_mb", "sum"))
        .reset_index()
        .sort_values("total_mb", ascending=False)
    )
