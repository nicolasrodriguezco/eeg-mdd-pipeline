"""Particiones por sujeto con verificación explícita de que no hay solapamiento."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class SubjectLeakError(AssertionError):
    """Un sujeto aparece en entrenamiento y en prueba a la vez."""


def assert_disjoint(groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    overlap = set(np.asarray(groups)[train_idx]) & set(np.asarray(groups)[test_idx])
    if overlap:
        raise SubjectLeakError(f"Sujetos en entrenamiento y prueba a la vez: {sorted(overlap)[:5]}")


def subject_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[set, set]]:
    """Partición estratificada de SUJETOS (no de filas). Depende solo del conjunto de
    sujetos y sus etiquetas, así que es idéntica para datos por sujeto y por ensayo."""
    subj = (pd.DataFrame({"g": np.asarray(groups), "y": np.asarray(y)})
            .drop_duplicates("g").sort_values("g").reset_index(drop=True))
    if subj["g"].duplicated().any():
        raise ValueError("Un sujeto tiene más de una etiqueta")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(set(subj["g"].iloc[tr]), set(subj["g"].iloc[te])) for tr, te in cv.split(subj["g"], subj["y"])]


def outer_splits(y: np.ndarray, groups: np.ndarray, n_splits: int = 5, n_repeats: int = 10,
                 seed0: int = 0) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """(repetición, pliegue, train_idx, test_idx) sobre filas, con sujetos disjuntos."""
    groups = np.asarray(groups)
    for r in range(n_repeats):
        for k, (tr_s, te_s) in enumerate(subject_folds(y, groups, n_splits, seed0 + r)):
            tr = np.flatnonzero(np.isin(groups, list(tr_s)))
            te = np.flatnonzero(np.isin(groups, list(te_s)))
            assert_disjoint(groups, tr, te)
            yield r, k, tr, te
