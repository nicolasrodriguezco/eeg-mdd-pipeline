"""Paso 14 (D6, exploratorio): test de permutación del RF de amplitudes crudas, separado por sujeto.

El control de inflación (``07_models.py --only control``) dio un AUC por sujeto de 0.68 / 0.75 / 0.66 en tres
repeticiones, pero sin test de permutación. Aquí se usa exactamente la misma configuración que la repetición 0:

- pliegues de ``outer_splits(seed0=0)``;
- RF con 300 árboles, ``max_features="sqrt"``, ``min_samples_leaf=3``, ``class_weight="balanced"`` y
  ``random_state=0``;
- 1 de cada 2 muestras de la señal regional.

Las etiquetas se permutan entre sujetos. p = (1 + #{AUC_perm ≥ AUC_obs}) / (1 + n_perm).
"""

import argparse
import json
import time
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from eeg_mdd.config import load_config
from eeg_mdd.evaluation.cv import outer_splits


def run_once(Xf, y, g, n_jobs, seed=0):
    prob = np.zeros(len(y))
    for _, _, tr, te in outer_splits(y, g, 5, 1, seed0=seed):
        m = RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=3,
                                   class_weight="balanced", n_jobs=n_jobs, random_state=seed).fit(Xf[tr], y[tr])
        prob[te] = m.predict_proba(Xf[te])[:, 1]
    s = pd.DataFrame({"g": g, "y": y, "p": prob}).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
    return roc_auc_score(s["y"], s["p"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=24)
    args = ap.parse_args()
    cfg = load_config(args.config)
    feat_dir = Path(cfg["paths"]["processed_dir"]) / "features"
    d = np.load(feat_dir / "trial_regions.npz", allow_pickle=True)
    X = np.load(feat_dir / "trial_regions_X.npy", mmap_mode="r")
    Xf = np.asarray(X).reshape(len(X), -1)[:, ::2].astype(np.float32)
    y = (d["group"] == "MDD").astype(int)
    g = d["subject_id"].astype(str)
    out = Path(cfg["paths"]["reports_dir"]) / "results" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    obs = run_once(Xf, y, g, args.n_jobs)
    print(f"observado (rep 0): {obs:.3f}  [{time.time() - t0:.0f} s]", flush=True)
    subj = pd.DataFrame({"g": g, "y": y}).drop_duplicates("g")
    null = []
    for i in range(args.n_perm):
        rng = np.random.default_rng(10_000 + i)
        mp = dict(zip(subj["g"], rng.permutation(subj["y"].to_numpy())))
        null.append(run_once(Xf, np.array([mp[s] for s in g]), g, args.n_jobs))
        if i % 10 == 9:
            print(f"  perm {i + 1}/{args.n_perm}  [{time.time() - t0:.0f} s]", flush=True)
    null = np.array(null)
    np.save(out / "rawrf_perm_null.npy", null)
    r = {"observed_auc_rep0": obs, "n_perm": args.n_perm,
         "p_value": float((1 + (null >= obs).sum()) / (1 + args.n_perm)),
         "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
         "seconds": round(time.time() - t0, 1)}
    (out / "rawrf_permutation.json").write_text(json.dumps(r, indent=2))
    print(r)


if __name__ == "__main__":
    main()
