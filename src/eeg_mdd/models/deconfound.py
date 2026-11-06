"""Desconfusión dentro del pliegue (D6, exploratorio).

``ConfoundRegressor`` recibe una matriz cuyas últimas ``n_confounds`` columnas son los confusores (por ejemplo edad,
sexo y escolaridad):

- En ``fit`` estima por mínimos cuadrados, solo con los sujetos de entrenamiento, la parte de cada característica
  que explican los confusores.
- En ``transform`` resta esa parte con los coeficientes de entrenamiento y descarta los confusores.

Así el clasificador no ve información lineal de los confusores, y ningún dato de prueba interviene en el ajuste.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ConfoundRegressor(TransformerMixin, BaseEstimator):
    def __init__(self, n_confounds: int = 3):
        self.n_confounds = n_confounds

    def _split(self, X):
        X = np.asarray(X, dtype=float)
        k = self.n_confounds
        return X[:, :-k], X[:, -k:]

    def fit(self, X, y=None):
        F, C = self._split(X)
        self.c_mean_ = np.nanmean(C, axis=0)
        A = np.c_[np.ones(len(C)), np.nan_to_num(C - self.c_mean_)]
        self.coef_, *_ = np.linalg.lstsq(A, F, rcond=None)
        return self

    def transform(self, X):
        F, C = self._split(X)
        A = np.c_[np.ones(len(C)), np.nan_to_num(C - self.c_mean_)]
        return F - A[:, 1:] @ self.coef_[1:]
