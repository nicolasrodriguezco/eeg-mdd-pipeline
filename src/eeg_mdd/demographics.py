"""Tabla demográfica por grupo y pruebas de diferencias (análisis de confusores).

Estas variables NUNCA entran como predictores del modelo de EEG; solo se usan
para describir la muestra, para el análisis de confusores y para la línea base
demográfica (ver docs/preregistration.md).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

GROUPS = ("MDD", "HC")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def continuous_row(df: pd.DataFrame, col: str, group_col: str = "group") -> dict:
    """Media ± DE, mediana [RIC] por grupo y prueba de Mann-Whitney (y t de Welch)."""
    out = {"variable": col}
    vals = {g: _num(df.loc[df[group_col] == g, col]).dropna() for g in GROUPS}
    for g, v in vals.items():
        out[f"{g}_n"] = int(len(v))
        out[f"{g}_mean_sd"] = f"{v.mean():.1f} ± {v.std(ddof=1):.1f}" if len(v) > 1 else ""
        q1, med, q3 = (v.quantile([0.25, 0.5, 0.75]).tolist() if len(v) else [np.nan] * 3)
        out[f"{g}_median_iqr"] = f"{med:.1f} [{q1:.1f}–{q3:.1f}]" if len(v) else ""
    a, b = vals["MDD"], vals["HC"]
    if len(a) > 1 and len(b) > 1:
        out["test"] = "Mann-Whitney U"
        out["statistic"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").statistic)
        out["p_value"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        out["p_welch_t"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    return out


def categorical_row(df: pd.DataFrame, col: str, group_col: str = "group") -> dict:
    """Conteos por grupo y prueba exacta de Fisher (2x2) o chi² (si hay más niveles)."""
    tab = pd.crosstab(df[col].astype(str).str.strip(), df[group_col]).reindex(columns=list(GROUPS), fill_value=0)
    out = {"variable": col}
    for g in GROUPS:
        out[f"{g}_n"] = int(tab[g].sum())
        out[f"{g}_counts"] = ", ".join(f"{lvl}={int(n)}" for lvl, n in tab[g].items())
    if tab.shape[0] == 2:
        out["test"] = "Fisher exact"
        res = stats.fisher_exact(tab.values)
        out["statistic"], out["p_value"] = float(res.statistic), float(res.pvalue)
    elif tab.shape[0] > 2:
        out["test"] = "Chi-squared"
        res = stats.chi2_contingency(tab.values)
        out["statistic"], out["p_value"] = float(res.statistic), float(res.pvalue)
    return out


def demographic_table(df: pd.DataFrame, continuous: list[str], categorical: list[str],
                      group_col: str = "group") -> pd.DataFrame:
    rows = [continuous_row(df, c, group_col) for c in continuous if c in df.columns]
    rows += [categorical_row(df, c, group_col) for c in categorical if c in df.columns]
    return pd.DataFrame(rows)


_FILE_DT = re.compile(r"(20\d{6})\s+(\d{2,4})")


def recording_datetime_from_name(path: str | Path) -> tuple[pd.Timestamp | None, str]:
    """Fecha (y hora en texto, tal cual) escritas en el nombre del archivo.

    La hora a veces viene truncada (p. ej. "14" o "110"), así que se devuelve
    como texto y la fecha como Timestamp. No se adivina nada.
    """
    m = _FILE_DT.search(Path(path).name)
    if not m:
        return None, ""
    return pd.Timestamp(m.group(1)), m.group(2)
