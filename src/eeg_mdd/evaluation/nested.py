"""Validación cruzada anidada, agrupada por sujeto, para modelos por sujeto o por ensayo.

- Bucle externo: ``StratifiedGroupKFold`` repetido (``outer_splits``).
- Bucle interno: ``GridSearchCV`` con ``StratifiedGroupKFold`` sobre los sujetos de
  entrenamiento (hiperparámetros, y si se pide, la familia de modelo).
- Todo lo que aprende (imputación, escalado, selección, covarianzas de referencia)
  está dentro del estimador/Pipeline, así que se ajusta solo con entrenamiento.
- Las probabilidades por ensayo se promedian por sujeto antes de calcular métricas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from .cv import assert_disjoint, outer_splits
from .metrics import METRICS, aggregate_to_subject, bootstrap_ci, subject_metrics


@dataclass
class CVResult:
    name: str
    oof: pd.DataFrame                    # subject_id, y, prob, repeat, fold
    fold_scores: pd.DataFrame            # métricas por (repeat, fold)
    repeat_scores: pd.DataFrame          # métricas por repetición (predicciones OOF agrupadas)
    best_params: list = field(default_factory=list)

    def summary(self) -> pd.Series:
        return self.repeat_scores[METRICS].mean()

    def mean_prob(self) -> pd.DataFrame:
        """Probabilidad OOF media por sujeto sobre las repeticiones."""
        return self.oof.groupby("subject_id").agg(y=("y", "first"), prob=("prob", "mean")).reset_index()

    def ci(self, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
        """IC 95 % por bootstrap sobre sujetos, promediando los extremos entre repeticiones."""
        per_rep = []
        for r, d in self.oof.groupby("repeat"):
            per_rep.append(bootstrap_ci(d["y"].to_numpy(), d["prob"].to_numpy(), n_boot, seed + int(r)))
        stacked = pd.concat(per_rep, keys=range(len(per_rep)))
        out = stacked.groupby(level=1).mean()
        out["estimate"] = self.summary()
        return out.loc[METRICS]


def _inner_cv(n_inner: int, seed: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)


def _fit_predict(estimator, param_grid, X, y, groups, tr, te, n_inner, seed, scoring):
    assert_disjoint(groups, tr, te)
    Xtr = X[tr] if not isinstance(X, pd.DataFrame) else X.iloc[tr]
    Xte = X[te] if not isinstance(X, pd.DataFrame) else X.iloc[te]
    if param_grid:
        inner = _inner_cv(n_inner, seed)
        # El bucle interno también separa por sujeto: groups solo de entrenamiento.
        splits = list(inner.split(np.zeros(len(tr)), y[tr], groups[tr]))
        for itr, ite in splits:
            assert_disjoint(groups[tr], itr, ite)
        gs = GridSearchCV(clone(estimator), param_grid, scoring=scoring, cv=splits, n_jobs=1, refit=True,
                          error_score="raise")
        gs.fit(Xtr, y[tr])
        model, best = gs.best_estimator_, gs.best_params_
    else:
        model, best = clone(estimator).fit(Xtr, y[tr]), {}
    prob = model.predict_proba(Xte)[:, list(model.classes_).index(1)]
    return te, prob, best


def nested_cv(name: str, estimator, param_grid, X, y, groups, n_splits: int = 5, n_repeats: int = 10,
              n_inner: int = 5, seed0: int = 0, n_jobs: int = 1, scoring: str = "roc_auc") -> CVResult:
    """CV anidada por sujeto. ``y`` y ``groups`` tienen una entrada por fila de ``X``
    (fila = sujeto o ensayo). Devuelve predicciones OOF por sujeto y métricas."""
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    jobs = list(outer_splits(y, groups, n_splits, n_repeats, seed0))
    outs = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict)(estimator, param_grid, X, y, groups, tr, te, n_inner, seed0 + 1000 * r + k, scoring)
        for r, k, tr, te in jobs)

    oof_rows, fold_rows, params = [], [], []
    for (r, k, _, _), (te, prob, best) in zip(jobs, outs):
        subj = aggregate_to_subject(prob, y[te], groups[te]).assign(repeat=r, fold=k)
        oof_rows.append(subj)
        fold_rows.append({"repeat": r, "fold": k, "n_test_subjects": len(subj),
                          **subject_metrics(subj["y"], subj["prob"])})
        params.append({"repeat": r, "fold": k, **{str(p): str(v) for p, v in best.items()}})
    oof = pd.concat(oof_rows, ignore_index=True)
    rep = pd.DataFrame([{"repeat": r, **subject_metrics(d["y"], d["prob"])} for r, d in oof.groupby("repeat")])
    return CVResult(name, oof, pd.DataFrame(fold_rows), rep, params)


def permutation_test(estimator, param_grid, X, y, groups, observed_auc: float, n_perm: int = 1000,
                     n_splits: int = 5, n_inner: int = 5, seed0: int = 0, n_jobs: int = 1) -> dict:
    """Test de permutación a nivel sujeto: se barajan las etiquetas ENTRE sujetos
    (cada ensayo hereda la etiqueta permutada de su sujeto) y se repite toda la
    CV anidada (1 repetición). p = (1 + #{AUC_perm >= AUC_obs}) / (1 + n_perm)."""
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    subj = pd.DataFrame({"g": groups, "y": y}).drop_duplicates("g")

    def one(i):
        rng = np.random.default_rng(10_000 + i)
        perm = dict(zip(subj["g"], rng.permutation(subj["y"].to_numpy())))
        yp = np.array([perm[g] for g in groups])
        res = nested_cv("perm", estimator, param_grid, X, yp, groups, n_splits, 1, n_inner, seed0, n_jobs=1)
        return float(res.repeat_scores["auc"].iloc[0])

    null = np.array(Parallel(n_jobs=n_jobs)(delayed(one)(i) for i in range(n_perm)))
    p = (1 + np.sum(null >= observed_auc)) / (1 + n_perm)
    return {"observed_auc": observed_auc, "n_perm": n_perm, "p_value": float(p),
            "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)), "null": null}


def corrected_ttest(diff_per_fold: np.ndarray, n_train: int, n_test: int) -> dict:
    """t-test de remuestreo corregido (Nadeau y Bengio, 2003; Bouckaert y Frank, 2004)
    para comparar dos modelos evaluados en los mismos pliegues."""
    from scipy import stats

    d = np.asarray(diff_per_fold, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / n + n_test / n_train) * var) if var > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return {"mean_diff": float(d.mean()), "t": float(t), "p_value": float(p), "n_folds": n}
