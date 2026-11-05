"""Catálogo de modelos: cada entrada devuelve (Pipeline, rejilla de hiperparámetros).

Todos los pasos que aprenden de los datos (imputación, escalado, covarianzas de
referencia, prototipos de ERP) están dentro del Pipeline, así que en la CV
anidada se ajustan solo con los sujetos de entrenamiento.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
C_GRID = np.logspace(-3, 1, 9)


def _lr(penalty="l2", **kw):
    # scikit-learn >= 1.8: la penalización se elige con l1_ratio (0 = L2, 1 = L1).
    if penalty == "l1":
        return LogisticRegression(l1_ratio=1.0, solver="liblinear", class_weight="balanced", max_iter=5000,
                                  random_state=SEED, **kw)
    return LogisticRegression(l1_ratio=0.0, solver="lbfgs", class_weight="balanced", max_iter=5000,
                              random_state=SEED, **kw)


def _tab(clf) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", clf)])


def lgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(max_depth=3, num_leaves=7, learning_rate=0.05, min_child_samples=5, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.5, class_weight="balanced", random_state=SEED,
                          n_jobs=1, verbose=-1)


def tabular_models() -> dict[str, tuple[Pipeline, list | dict]]:
    """Modelos para tablas por sujeto (características o demografía)."""
    return {
        "LR-L2": (_tab(_lr("l2")), {"clf__C": C_GRID}),
        "LR-L1": (_tab(_lr("l1")), {"clf__C": np.logspace(-2, 1, 7)}),
        "SVM-linear": (_tab(SVC(kernel="linear", probability=True, class_weight="balanced", random_state=SEED)),
                       {"clf__C": np.logspace(-4, 0, 5)}),
        "SVM-RBF": (_tab(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)),
                    {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 1e-3, 1e-4]}),
        "RandomForest": (_tab(RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=SEED,
                                                     n_jobs=1)),
                         {"clf__max_depth": [2, 4, None], "clf__min_samples_leaf": [1, 3]}),
        "LightGBM": (_tab(lgbm()), {"clf__n_estimators": [100, 300]}),
    }


def model_selection_grid() -> tuple[Pipeline, list[dict]]:
    """Una sola búsqueda interna que elige familia de modelo E hiperparámetros.

    Sirve para estimar sin sesgo el desempeño de "quedarse con el mejor modelo".
    """
    grids = []
    for _, (pipe, grid) in tabular_models().items():
        g = {"clf": [pipe.named_steps["clf"]]}
        g.update(grid)
        grids.append(g)
    return _tab(_lr("l2")), grids


def riemann_models(precomputed_cov: bool = False) -> dict[str, tuple[Pipeline, dict]]:
    """Modelos por ensayo sobre señales regionales (n_trials, n_regions, n_times).

    ``precomputed_cov=True``: TS-LR y MDM reciben ya las covarianzas OAS por ensayo
    (n_trials, n_regions, n_regions). Es seguro: cada covarianza usa solo su propio
    ensayo y no aprende nada de otros datos; lo que sí aprende (media de referencia
    del espacio tangente, escalado, LR, centroides de MDM) sigue dentro del Pipeline.
    ERPCov-TS-LR siempre recibe la señal: sus prototipos de ERP usan etiquetas y se
    ajustan dentro del pliegue. Por coste de cómputo su C queda fijo a priori (0.1),
    sin rejilla interna (desviación D5).
    """
    from pyriemann.classification import MDM
    from pyriemann.estimation import Covariances, ERPCovariances
    from pyriemann.tangentspace import TangentSpace

    cov = [] if precomputed_cov else [("cov", Covariances(estimator="oas"))]
    return {
        "TS-LR": (Pipeline(cov + [("ts", TangentSpace(metric="riemann")), ("scale", StandardScaler()),
                                  ("clf", _lr("l2"))]), {"clf__C": np.logspace(-3, 1, 5)}),
        "MDM": (Pipeline(cov + [("clf", MDM(metric="riemann"))]), {}),
        "ERPCov-TS-LR": (Pipeline([("cov", ERPCovariances(classes=[0, 1], estimator="oas")),
                                   ("ts", TangentSpace(metric="riemann")), ("scale", StandardScaler()),
                                   ("clf", _lr("l2", C=0.1))]), {}),
    }
