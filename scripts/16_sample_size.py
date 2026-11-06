"""Fase C (D6, exploratorio): tamaño de muestra mínimo para detectar un AUC dado.

Cota optimista: una puntuación ya entrenada (sin costo de aprendizaje) con distribución binormal de varianzas
iguales, proporción MDD = 18/45 como en este estudio, prueba unilateral alfa = 0.05 con la nula exacta de
Mann-Whitney (aproximación normal) y semilla fija. Un pipeline que aprende de la misma muestra necesita más
(ver power_semisynthetic.csv). Salida: reports/results/audit/sample_size.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parents[1] / "reports" / "results" / "audit"
P_MDD, ALPHA, SIMS = 18 / 45, 0.05, 4000
rng = np.random.default_rng(0)


def power(n, auc):
    n1 = max(2, round(n * P_MDD)); n0 = n - n1
    d = np.sqrt(2) * stats.norm.ppf(auc)
    thr = 0.5 + stats.norm.ppf(1 - ALPHA) * np.sqrt((n0 + n1 + 1) / (12 * n0 * n1))
    a = rng.normal(d, 1, (SIMS, n1)); b = rng.normal(0, 1, (SIMS, n0))
    r = stats.rankdata(np.concatenate([a, b], 1), axis=1)[:, :n1].sum(1)
    est = (r - n1 * (n1 + 1) / 2) / (n1 * n0)
    hw = 1.96 * np.std(est)
    return float((est >= thr).mean()), float(hw)


rows = []
for auc in (0.60, 0.65, 0.70, 0.75, 0.80):
    for n in (45, 60, 80, 100, 150, 200, 300, 400, 600, 800):
        p, hw = power(n, auc)
        rows.append({"true_auc": auc, "n_total": n, "power": p, "auc_ci_halfwidth": hw})
df = pd.DataFrame(rows)
need = (df[df.power >= 0.8].groupby("true_auc").n_total.min()
        .reindex(df.true_auc.unique()).rename("n_for_80pct_power"))
df.to_csv(OUT / "sample_size.csv", index=False)
need.reset_index().to_csv(OUT / "sample_size_needed.csv", index=False)
print(df.pivot(index="n_total", columns="true_auc", values="power").round(2).to_string())
print(need.to_string())
