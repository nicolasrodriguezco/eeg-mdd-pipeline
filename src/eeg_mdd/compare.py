"""Comparación entre el dataset viejo (tabla ancha) y el nuevo (registro por sujeto)."""

from __future__ import annotations

import pandas as pd


def counts_from_old(old: pd.DataFrame, subject_col: str = "Subject", group_col: str = "Group",
                    condition_col: str = "Condition") -> pd.DataFrame:
    """Ensayos por sujeto y condición en la tabla vieja (una fila por ensayo)."""
    counts = (
        old.groupby([subject_col, condition_col]).size().unstack(fill_value=0)
        .add_prefix("old_n_").reset_index().rename(columns={subject_col: "subject_id"})
    )
    groups = old.groupby(subject_col)[group_col].agg(lambda s: ",".join(sorted(set(map(str, s)))))
    counts["old_group"] = counts["subject_id"].map(groups)
    counts["subject_id"] = counts["subject_id"].astype(str).str.strip()
    return counts


def compare_old_new(old_counts: pd.DataFrame, new_log: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    """Une ambas versiones y marca cada diferencia."""
    new = new_log.copy()
    new["subject_id"] = new["subject_id"].astype(str).str.strip()
    labels = subjects[["eeg_id", "group"]].rename(columns={"eeg_id": "subject_id", "group": "new_group"})
    new = new.merge(labels, on="subject_id", how="left")

    merged = old_counts.merge(new, on="subject_id", how="outer", indicator=True)
    merged["in_old"] = merged["_merge"].isin(["both", "left_only"])
    merged["in_new"] = merged["_merge"].isin(["both", "right_only"])
    merged = merged.drop(columns="_merge")

    def flag(r) -> str:
        issues = []
        if not r["in_old"]:
            issues.append("solo en el nuevo")
        if not r["in_new"]:
            issues.append("solo en el viejo")
        if r["in_old"] and r["in_new"] and str(r.get("old_group")) != str(r.get("new_group")):
            issues.append(f"ETIQUETA DISTINTA: viejo={r.get('old_group')} nuevo={r.get('new_group')}")
        return "; ".join(issues)

    merged["issues"] = merged.apply(flag, axis=1)
    return merged.sort_values(["issues", "subject_id"], ascending=[False, True]).reset_index(drop=True)
