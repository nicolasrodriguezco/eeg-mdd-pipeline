"""Métricas por sujeto e intervalos por bootstrap sobre sujetos."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score

METRICS = ["balanced_accuracy", "auc", "sensitivity", "specificity", "f1", "brier"]


def aggregate_to_subject(prob: np.ndarray, y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """Media de probabilidades por sujeto (la etiqueta es constante dentro del sujeto)."""
    df = pd.DataFrame({"subject_id": groups, "y": y, "prob": prob})
    agg = df.groupby("subject_id").agg(y=("y", "first"), y_nuniq=("y", "nunique"), prob=("prob", "mean"))
    if (agg["y_nuniq"] > 1).any():
        raise ValueError("Un sujeto tiene ensayos con etiquetas distintas")
    return agg.drop(columns="y_nuniq").reset_index()


def subject_metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Positivo = MDD (1). Umbral fijo de 0,5 (no se ajusta con los datos de prueba)."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    both = len(np.unique(y)) == 2
    return {
        "balanced_accuracy": balanced_accuracy_score(y, pred) if both else np.nan,
        "auc": roc_auc_score(y, prob) if both else np.nan,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "brier": brier_score_loss(y, prob),
    }


def bootstrap_ci(y: np.ndarray, prob: np.ndarray, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> pd.DataFrame:
    """IC percentil por bootstrap estratificado sobre sujetos (una fila por sujeto)."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rows = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        rows.append(subject_metrics(y[idx], prob[idx]))
    boot = pd.DataFrame(rows)
    point = subject_metrics(y, prob)
    return pd.DataFrame({
        "estimate": pd.Series(point),
        "ci_low": boot.quantile(alpha / 2),
        "ci_high": boot.quantile(1 - alpha / 2),
    })
