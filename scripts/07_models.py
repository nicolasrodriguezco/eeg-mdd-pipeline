"""Paso 7: modelos clásicos, demográficos y riemannianos con CV anidada por sujeto.

Protocolo (docs/preregistration.md):
  externo  5 pliegues estratificados por sujeto x 10 repeticiones (semillas 0-9)
  interno  5 pliegues agrupados por sujeto (hiperparámetros)
  métrica  por sujeto (probabilidades de ensayo promediadas), IC por bootstrap, permutaciones

Salidas: reports/results/models/<modelo>/{oof,fold_scores,repeat_scores,ci,params}.csv y perm.json
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
from sklearn.model_selection import StratifiedKFold

from eeg_mdd.config import load_config
from eeg_mdd.evaluation import nested_cv, permutation_test
from eeg_mdd.features import family_of
from eeg_mdd.matching import load_clinical
from eeg_mdd.models import model_selection_grid, riemann_models, tabular_models

N_SPLITS, N_REPEATS, N_INNER = 5, 10, 5


def save(res, out: Path, perm: dict | None = None, extra: dict | None = None) -> pd.Series:
    out.mkdir(parents=True, exist_ok=True)
    res.oof.to_csv(out / "oof.csv", index=False)
    res.fold_scores.to_csv(out / "fold_scores.csv", index=False)
    res.repeat_scores.to_csv(out / "repeat_scores.csv", index=False)
    pd.DataFrame(res.best_params).to_csv(out / "params.csv", index=False)
    ci = res.ci()
    ci.to_csv(out / "ci.csv")
    meta = dict(extra or {})
    if perm:
        np.save(out / "perm_null.npy", perm.pop("null"))
        meta.update(perm)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    s = ci["estimate"]
    print(f"  {res.name:<28} AUC={s['auc']:.3f} [{ci.loc['auc', 'ci_low']:.3f}, {ci.loc['auc', 'ci_high']:.3f}]  "
          f"BalAcc={s['balanced_accuracy']:.3f}" + (f"  p_perm={meta['p_value']:.4f}" if "p_value" in meta else ""),
          flush=True)
    return s


def run(name, est, grid, X, y, g, out_dir, n_perm, n_jobs, extra=None):
    t0 = time.time()
    res = nested_cv(name, est, grid, X, y, g, N_SPLITS, N_REPEATS, N_INNER, n_jobs=n_jobs)
    perm = None
    if n_perm:
        obs = float(res.repeat_scores["auc"].iloc[0])           # repetición 0, igual que en las permutaciones
        perm = permutation_test(est, grid, X, y, g, obs, n_perm=n_perm, n_splits=N_SPLITS, n_inner=N_INNER,
                                n_jobs=n_jobs)
    meta = {"seconds": round(time.time() - t0, 1), "n_rows": int(len(y)), "n_subjects": int(len(set(g))),
            **(extra or {})}
    return save(res, out_dir / name, perm, meta)


def trialwise_control(X, y, g, out: Path) -> pd.DataFrame:
    """INVÁLIDO A PROPÓSITO: separación por ensayo (como la versión vieja), solo para medir la inflación."""
    rows = []
    models = {"TS-LR": riemann_models()["TS-LR"][0].set_params(clf__C=0.1),
              "RawAmplitude-RF (old approach)": None}
    Xflat = X.reshape(len(X), -1)[:, ::2]                 # señal regional cruda, 1 de cada 2 muestras
    for name, est in models.items():
        for split in ("trial-wise (INVALID)", "subject-wise"):
            for r in range(3):
                if split.startswith("trial"):
                    folds = StratifiedKFold(5, shuffle=True, random_state=r).split(X, y)
                else:
                    from eeg_mdd.evaluation.cv import outer_splits
                    folds = [(tr, te) for _, _, tr, te in outer_splits(y, g, 5, 1, seed0=r)]
                prob = np.zeros(len(y))
                for tr, te in folds:
                    if est is None:
                        m = RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=3,
                                                   class_weight="balanced", n_jobs=-1, random_state=r)
                        m.fit(Xflat[tr], y[tr])
                        prob[te] = m.predict_proba(Xflat[te])[:, 1]
                    else:
                        from sklearn.base import clone
                        m = clone(est).fit(X[tr], y[tr])
                        prob[te] = m.predict_proba(X[te])[:, 1]
                trial_auc = roc_auc_score(y, prob)
                subj = pd.DataFrame({"g": g, "y": y, "p": prob}).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
                rows.append({"model": name, "split": split, "repeat": r, "trial_auc": trial_auc,
                             "subject_auc": roc_auc_score(subj["y"], subj["p"])})
                print(f"  control {name:<32} {split:<22} rep{r} trial AUC={trial_auc:.3f} "
                      f"subject AUC={rows[-1]['subject_auc']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "inflation_control.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--only", nargs="*", default=None, help="demo primary secondary families selection riemann control")
    ap.add_argument("--perm-fast", type=int, default=1000)
    ap.add_argument("--perm-slow", type=int, default=200)
    ap.add_argument("--all-subjects", action="store_true",
                    help="sensibilidad: incluye también a los sujetos excluidos por los criterios")
    args = ap.parse_args()
    todo = set(args.only or ["demo", "primary", "secondary", "families", "selection", "riemann", "control"])

    cfg = load_config(args.config)
    feat_dir = Path(cfg["paths"]["processed_dir"]) / "features"
    out = Path(cfg["paths"]["reports_dir"]) / "results" / "models"
    feats = pd.read_parquet(feat_dir / "subject_features.parquet").sort_values("subject_id").reset_index(drop=True)
    if args.all_subjects:
        out = out.parent / "models_all53"
    else:
        feats = feats[feats["included"]].reset_index(drop=True)
    y = (feats["group"] == "MDD").astype(int).to_numpy()
    g = feats["subject_id"].to_numpy()
    cols = [c for c in feats.columns if family_of(c)]
    X = feats[cols]
    print(f"Sujetos: {len(feats)} (MDD={y.sum()}, HC={len(y) - y.sum()}), características: {len(cols)}")

    tab = tabular_models()
    if "demo" in todo:
        cc = cfg["dataset"]["clinical"]
        clin = load_clinical(cc["file"], cc["id_column"], cc["group_column"], cc["group_map"])
        demo = pd.DataFrame({
            "subject_id": clin["clinical_id"],
            "age": pd.to_numeric(clin["age"], errors="coerce"),
            "male": (clin["gender"].str.strip() == "M").astype(float),
            "education": pd.to_numeric(clin["education（years）"], errors="coerce"),
        }).set_index("subject_id").loc[g].reset_index(drop=True)
        print("Línea base demográfica (edad, sexo, escolaridad):")
        run("Demographics-LR", *tab["LR-L2"], demo, y, g, out, args.perm_fast, args.n_jobs,
            {"features": "age, sex, education"})

    if "primary" in todo:
        print("Modelo primario (fijado a priori): LR-L2 sobre todas las características")
        run("PRIMARY_LR-L2_all", *tab["LR-L2"], X, y, g, out, args.perm_fast, args.n_jobs,
            {"features": "all", "n_features": len(cols)})

    if "secondary" in todo:
        print("Modelos secundarios sobre todas las características:")
        for name in ("LR-L1", "SVM-linear", "SVM-RBF", "RandomForest", "LightGBM"):
            n_perm = args.perm_fast if name in ("LR-L1", "SVM-linear") else args.perm_slow
            run(f"{name}_all", *tab[name], X, y, g, out, n_perm, args.n_jobs, {"features": "all"})

    if "families" in todo:
        print("LR-L2 por familia de características:")
        for fam in ("spectral", "erp", "connectivity", "complexity"):
            fc = [c for c in cols if family_of(c) == fam]
            run(f"LR-L2_{fam}", *tab["LR-L2"], feats[fc], y, g, out, args.perm_fast, args.n_jobs,
                {"features": fam, "n_features": len(fc)})

    if "selection" in todo:
        print("Selección de familia de modelo DENTRO del bucle interno (estimación sin sesgo de 'el mejor'):")
        pipe, grids = model_selection_grid()
        run("NestedModelSelection_all", pipe, grids, X, y, g, out, args.perm_slow, args.n_jobs, {"features": "all"})

    if "riemann" in todo or "control" in todo:
        d = np.load(feat_dir / "trial_regions.npz", allow_pickle=True)
        Xt = d["X"].astype(np.float64)
        yt = (d["group"] == "MDD").astype(int)
        gt = d["subject_id"].astype(str)
        print(f"Ensayos: {len(yt)} de {len(set(gt))} sujetos; forma {Xt.shape}")
        if "riemann" in todo:
            for name, (est, grid) in riemann_models().items():
                run(name, est, grid, Xt, yt, gt, out, args.perm_slow, args.n_jobs, {"input": "trial region signals"})
        if "control" in todo:
            trialwise_control(Xt, yt, gt, out.parent / "control")


if __name__ == "__main__":
    main()
