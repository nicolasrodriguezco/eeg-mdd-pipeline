"""Paso 13 (D6, exploratorio y post hoc): análisis de la auditoría.

Partes (``--only``):
  confounds  EEG → escolaridad; calidad, EMG y conducta por grupo en los 53; EMG frente a complejidad y beta/gamma
  models     valor incremental (demografía + EEG), EEG desconfundido, PCA, conjunto compacto, conducta, Riemann por
             sujeto; Holm entre ellos y comparaciones con la línea base demográfica
  power      poder: puntuación binormal ideal y simulación semisintética con el pipeline primario
  curve      curva de aprendizaje del primario y de la demografía
  ident      identificación de sujeto a partir de covarianzas por ensayo (por qué la división por ensayo infla)

Salida: ``reports/results/audit/``. Mismos pliegues que los modelos prerregistrados.
"""

import argparse
import json
import time
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eeg_mdd.config import load_config
from eeg_mdd.evaluation import nested_cv, permutation_test
from eeg_mdd.evaluation.metrics import subject_metrics
from eeg_mdd.evaluation.nested import _fit_predict, corrected_ttest
from eeg_mdd.evaluation.cv import outer_splits
from eeg_mdd.features import family_of
from eeg_mdd.models import C_GRID, _lr, _tab
from eeg_mdd.models.deconfound import ConfoundRegressor
from eeg_mdd.resources import free_gb, limit_threads, safe_n_jobs

N_SPLITS, N_REPEATS, N_INNER = 5, 10, 5
DEMO = ["age", "male", "education"]


def save(res, out: Path, perm: dict | None, extra: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    res.oof.to_csv(out / "oof.csv", index=False)
    res.fold_scores.to_csv(out / "fold_scores.csv", index=False)
    res.repeat_scores.to_csv(out / "repeat_scores.csv", index=False)
    ci = res.ci()
    ci.to_csv(out / "ci.csv")
    meta = dict(extra)
    if perm:
        np.save(out / "perm_null.npy", perm.pop("null"))
        meta.update(perm)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    row = {"model": res.name, "auc": ci.loc["auc", "estimate"], "auc_lo": ci.loc["auc", "ci_low"],
           "auc_hi": ci.loc["auc", "ci_high"], "balanced_accuracy": ci.loc["balanced_accuracy", "estimate"],
           "brier": ci.loc["brier", "estimate"], "p_perm": meta.get("p_value"), "n_perm": meta.get("n_perm"),
           **{k: v for k, v in extra.items() if k in ("n_features", "test")}}
    (out / "row.json").write_text(json.dumps(row, default=float))
    print(f"  {res.name:<34} AUC={row['auc']:.3f} [{row['auc_lo']:.3f}, {row['auc_hi']:.3f}]"
          + (f" p={row['p_perm']:.4f}" if row["p_perm"] is not None else ""), flush=True)
    return row


def block_permutation(est, grid, X: np.ndarray, y, g, eeg_idx: np.ndarray, observed: float, n_perm: int,
                      n_jobs: int) -> dict:
    """Nula de "el EEG no añade nada": se barajan entre sujetos las FILAS del bloque de EEG.

    Las etiquetas y la demografía se dejan fijas, así que se conserva la relación demografía–diagnóstico y solo se
    rompe la del EEG con ambas.
    """
    def one(i):
        rng = np.random.default_rng(20_000 + i)
        Xp = X.copy()
        Xp[:, eeg_idx] = X[rng.permutation(len(X))][:, eeg_idx]
        r = nested_cv("perm", est, grid, Xp, y, g, N_SPLITS, 1, N_INNER, n_jobs=1)
        return float(r.repeat_scores["auc"].iloc[0])

    null = np.array(Parallel(n_jobs=n_jobs)(delayed(one)(i) for i in range(n_perm)))
    return {"observed_auc": observed, "n_perm": n_perm, "p_value": float((1 + (null >= observed).sum()) / (1 + n_perm)),
            "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)), "null": null}


def load(cfg):
    feat_dir = Path(cfg["paths"]["processed_dir"]) / "features"
    feats = pd.read_parquet(feat_dir / "subject_features.parquet").sort_values("subject_id").reset_index(drop=True)
    aud = pd.read_parquet(feat_dir / "audit_features.parquet").drop(columns=["group", "included", "faa_F3F4", "faa_F7F8"])
    meta = pd.read_csv(Path(cfg["paths"]["reports_dir"]) / "results" / "confounds" / "subject_meta.csv",
                       dtype={"subject_id": str})
    allf = feats.merge(aud, on="subject_id").merge(meta[["subject_id"] + DEMO], on="subject_id", how="left")
    compact = json.loads((feat_dir / "audit_compact_cols.json").read_text())
    return allf, compact, feat_dir


# ----------------------------------------------------------------------------------------------- confusores
def part_confounds(cfg, allf, out, n_jobs, n_perm):
    reports = Path(cfg["paths"]["reports_dir"])
    log = pd.read_csv(reports / "preprocessing_log.csv", dtype={"subject_id": str})
    a53 = allf.merge(log[["subject_id", "n_trials_kept_total", "n_bad_channels", "ica_n_excluded",
                          "ica_n_muscle_artifact", "n_segments_kept"]], on="subject_id")
    # El Excel tiene la demografía de los 53 y subject_meta solo de los 45
    from eeg_mdd.matching import load_clinical
    cc = cfg["dataset"]["clinical"]
    clin = load_clinical(cc["file"], cc["id_column"], cc["group_column"], cc["group_map"]).set_index("clinical_id")
    a53["education"] = pd.to_numeric(clin.loc[a53["subject_id"], "education（years）"], errors="coerce").to_numpy()
    rows = []
    tests = ["n_trials_kept_total", "n_bad_channels", "ica_n_excluded", "ica_n_muscle_artifact", "n_segments_kept",
             "emg_rel_global", "emg_logpow_global", "beh_rt_median", "beh_rt_sd", "beh_rt_cv", "beh_miss_rate",
             "beh_rt_sad_minus_happy", "ap_exponent_global", "alpha_peak_freq_post", "fmtheta_rel",
             "fmtheta_periodic"]
    for sample, d in (("all 53", a53), ("included 45", a53[a53["included"]])):
        for v in tests:
            m, h = d.loc[d["group"] == "MDD", v].dropna(), d.loc[d["group"] == "HC", v].dropna()
            p = stats.mannwhitneyu(m, h).pvalue
            sd = np.sqrt((m.var(ddof=1) + h.var(ddof=1)) / 2)
            rows.append({"sample": sample, "variable": v, "mdd_median": m.median(), "hc_median": h.median(),
                         "cohen_d": (m.mean() - h.mean()) / sd if sd > 0 else np.nan, "p_mwu": p})
    q = pd.DataFrame(rows)
    q.to_csv(out / "group_tests_quality_emg_behaviour.csv", index=False)
    print(q.round(3).to_string())

    inc = a53[a53["included"]]
    corr = []
    for f in ("cplxglob_sampen", "cplxglob_lzc", "cplxglob_hfd", "relglob_beta", "relglob_lowgamma",
              "ap_exponent_global", "wpli_theta_central_L__temporal_L"):
        r, p = stats.spearmanr(inc["emg_rel_global"], inc[f])
        corr.append({"feature": f, "spearman_r_with_emg": r, "p": p})
    # ¿El EMG predice el grupo? (univariado, como pista de confusión)
    c = pd.DataFrame(corr)
    c.to_csv(out / "emg_vs_features.csv", index=False)
    print(c.round(3).to_string())

    # Correlación de las 288 variables con la escolaridad entre los 45
    cols = [x for x in allf.columns if family_of(x)]
    pe = np.array([stats.spearmanr(inc[x], inc["education"]).pvalue for x in cols])
    # EEG → escolaridad por encima de la mediana, con el pipeline primario
    y_ed = (inc["education"] > inc["education"].median()).astype(int).to_numpy()
    g = inc["subject_id"].to_numpy()
    pipe, grid = _tab(_lr("l2")), {"clf__C": C_GRID}
    res = nested_cv("EEG -> education > median", pipe, grid, inc[cols], y_ed, g, N_SPLITS, N_REPEATS, N_INNER,
                    n_jobs=n_jobs)
    perm = permutation_test(pipe, grid, inc[cols], y_ed, g, float(res.repeat_scores["auc"].iloc[0]), n_perm=n_perm,
                            n_jobs=n_jobs)
    row = save(res, out / "eeg_to_education", perm, {"n_features": len(cols)})
    row.update({"n_pos": int(y_ed.sum()), "n": len(y_ed), "n_feat_p05_education": int((pe < 0.05).sum()),
                "n_feat_fdr_education": int((stats.false_discovery_control(pe) < 0.05).sum())})
    pd.DataFrame([row]).to_csv(out / "eeg_to_education.csv", index=False)

    # Demografía de los 45 incluidos
    demo = []
    for v in ("age", "education"):
        m, h = inc.loc[inc["group"] == "MDD", v], inc.loc[inc["group"] == "HC", v]
        demo.append({"variable": v, "mdd": f"{m.mean():.1f} ± {m.std():.1f}", "hc": f"{h.mean():.1f} ± {h.std():.1f}",
                     "p": stats.mannwhitneyu(m, h).pvalue})
    ct = pd.crosstab(inc["group"], inc["male"])
    demo.append({"variable": "sex F/M", "mdd": f"{ct.loc['MDD', 0]}/{ct.loc['MDD', 1]}",
                 "hc": f"{ct.loc['HC', 0]}/{ct.loc['HC', 1]}", "p": stats.fisher_exact(ct.to_numpy())[1]})
    pd.DataFrame(demo).to_csv(out / "demographics_included45.csv", index=False)
    print(pd.DataFrame(demo).to_string())


# ----------------------------------------------------------------------------------------------- modelos
def part_models(cfg, allf, compact, feat_dir, out, n_jobs, n_perm):
    inc = allf[allf["included"]].reset_index(drop=True)
    y = (inc["group"] == "MDD").astype(int).to_numpy()
    g = inc["subject_id"].to_numpy()
    cols = [c for c in allf.columns if family_of(c)]
    beh = ["beh_rt_median", "beh_rt_sd", "beh_rt_cv", "beh_miss_rate", "beh_rt_median_fear", "beh_rt_median_happy",
           "beh_rt_median_sad", "beh_rt_sad_minus_happy"]
    lr = {"clf__C": C_GRID}
    rows = []

    def done(name):
        f = out / name / "row.json"
        if f.exists():
            rows.append(json.loads(f.read_text()))
            print(f"  {name}: ya calculado", flush=True)
            return True
        return False

    def std(name, pipe, grid, X, extra):
        if done(name):
            return
        t0 = time.time()
        res = nested_cv(name, pipe, grid, X, y, g, N_SPLITS, N_REPEATS, N_INNER, n_jobs=n_jobs)
        perm = permutation_test(pipe, grid, X, y, g, float(res.repeat_scores["auc"].iloc[0]), n_perm=n_perm,
                                n_jobs=n_jobs)
        rows.append(save(res, out / name, perm, {**extra, "seconds": round(time.time() - t0, 1)}))

    # 1. Valor incremental: demografía + EEG, con la nula que baraja el bloque de EEG
    for name, ec in (("Demo+EEG288", cols), ("Demo+Compact", compact)):
        if done(name):
            continue
        X = inc[ec + DEMO].to_numpy(float)
        pipe = _tab(_lr("l2"))
        res = nested_cv(name, pipe, lr, X, y, g, N_SPLITS, N_REPEATS, N_INNER, n_jobs=n_jobs)
        perm = block_permutation(pipe, lr, X, y, g, np.arange(len(ec)), float(res.repeat_scores["auc"].iloc[0]),
                                 n_perm, n_jobs)
        rows.append(save(res, out / name, perm, {"n_features": len(ec) + 3,
                                                 "test": "EEG-block permutation (EEG adds nothing)"}))
    # 2. EEG desconfundido de edad, sexo y escolaridad dentro del pliegue
    deconf = Pipeline([("impute", SimpleImputer(strategy="median")), ("deconf", ConfoundRegressor(3)),
                       ("scale", StandardScaler()), ("clf", _lr("l2"))])
    std("EEG288-deconfounded", deconf, lr, inc[cols + DEMO].to_numpy(float),
        {"n_features": len(cols), "test": "label permutation"})
    # 3. PCA dentro del pliegue
    pca = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                    ("pca", PCA(random_state=0)), ("clf", _lr("l2"))])
    std("EEG288-PCA-LR", pca, {"pca__n_components": [5, 10, 20], "clf__C": C_GRID}, inc[cols].to_numpy(float),
        {"n_features": len(cols), "test": "label permutation"})
    # 4. Conjunto compacto con base teórica
    std("Compact-LR", _tab(_lr("l2")), lr, inc[compact].to_numpy(float),
        {"n_features": len(compact), "test": "label permutation"})
    # 5. Conducta (TR): sin EEG ni cuestionarios
    std("Behaviour-RT-LR", _tab(_lr("l2")), lr, inc[beh].to_numpy(float),
        {"n_features": len(beh), "test": "label permutation"})
    # 6. Riemann por sujeto: covarianza media por sujeto (sin etiquetas) → espacio tangente dentro del pliegue
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.utils.mean import mean_covariance
    covs = np.load(feat_dir / "trial_regions_cov.npy", mmap_mode="r")
    gt = np.load(feat_dir / "trial_regions.npz", allow_pickle=True)["subject_id"].astype(str)
    S = np.stack([mean_covariance(np.asarray(covs[gt == s]), metric="riemann") for s in g])
    riem = Pipeline([("ts", TangentSpace(metric="riemann")), ("scale", StandardScaler()), ("clf", _lr("l2"))])
    std("Subject-Riemann-TS-LR", riem, lr, S, {"n_features": 55, "test": "label permutation"})

    tab = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    tab["p_holm"] = multipletests(tab["p_perm"], method="holm")[1]
    # Comparación con la demografía y con el primario en los mismos pliegues (Nadeau–Bengio)
    mres = Path(cfg["paths"]["reports_dir"]) / "results" / "models"
    ref = {k: pd.read_csv(mres / k / "fold_scores.csv").set_index(["repeat", "fold"])["auc"]
           for k in ("Demographics-LR", "PRIMARY_LR-L2_all")}
    for i, r in tab.iterrows():
        fs = pd.read_csv(out / r["model"] / "fold_scores.csv").set_index(["repeat", "fold"])["auc"]
        for k, lab in (("Demographics-LR", "demo"), ("PRIMARY_LR-L2_all", "primary")):
            t = corrected_ttest((fs - ref[k]).to_numpy(), 36, 9)
            tab.loc[i, f"dAUC_vs_{lab}"], tab.loc[i, f"p_vs_{lab}"] = t["mean_diff"], t["p_value"]
    tab.to_csv(out / "exploratory_models.csv", index=False)
    print(tab.round(3).to_string())


# ----------------------------------------------------------------------------------------------- poder
def part_power(cfg, allf, out, n_jobs, n_sims):
    rng = np.random.default_rng(0)
    null = np.load(Path(cfg["paths"]["reports_dir"]) / "results" / "models" / "PRIMARY_LR-L2_all" / "perm_null.npy")
    thr = float(np.quantile(null, 0.95))
    n1, n0 = 18, 27
    # (a) Puntuación ideal binormal: el mejor caso, sin costo de aprendizaje
    ideal = []
    for auc in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        d = np.sqrt(2) * stats.norm.ppf(auc)
        hits = 0
        for _ in range(20000):
            a, b = rng.normal(d, 1, n1), rng.normal(0, 1, n0)
            hits += (a[:, None] > b[None, :]).mean() >= thr
        ideal.append({"true_auc": auc, "power_ideal_score": hits / 20000})
    # (b) Semisintético: las 288 variables reales con las filas barajadas (sin señal real) más un desplazamiento de
    #     d DE en k = 10 variables al azar; se corre el pipeline primario completo (1 repetición).
    inc = allf[allf["included"]].reset_index(drop=True)
    cols = [c for c in allf.columns if family_of(c)]
    X0 = inc[cols].to_numpy(float)
    y = (inc["group"] == "MDD").astype(int).to_numpy()
    g = inc["subject_id"].to_numpy()
    pipe, grid = _tab(_lr("l2")), {"clf__C": C_GRID}

    def sim(d, i):
        r = np.random.default_rng(1000 * int(d * 100) + i)
        X = X0[r.permutation(len(X0))].copy()
        S = r.choice(X.shape[1], 10, replace=False)
        sd = X[:, S].std(0)
        X[:, S] += d * sd * y[:, None]
        z = (X[:, S] - X[:, S].mean(0)) / sd
        from sklearn.metrics import roc_auc_score
        oracle = roc_auc_score(y, z.sum(1))
        res = nested_cv("sim", pipe, grid, X, y, g, N_SPLITS, 1, N_INNER, seed0=i, n_jobs=1)
        return d, oracle, float(res.repeat_scores["auc"].iloc[0])

    t0 = time.time()
    sims = Parallel(n_jobs=n_jobs)(delayed(sim)(d, i) for d in (0.0, 0.3, 0.5, 0.8, 1.0) for i in range(n_sims))
    s = pd.DataFrame(sims, columns=["d_per_feature", "oracle_auc", "cv_auc"])
    semi = s.groupby("d_per_feature").agg(oracle_auc=("oracle_auc", "mean"), cv_auc_mean=("cv_auc", "mean"),
                                          power=("cv_auc", lambda v: float((v >= thr).mean()))).reset_index()
    pd.DataFrame(ideal).to_csv(out / "power_ideal.csv", index=False)
    semi.to_csv(out / "power_semisynthetic.csv", index=False)
    (out / "power_meta.json").write_text(json.dumps({"threshold_null_q95": thr, "n_sims": n_sims, "k": 10,
                                                     "seconds": round(time.time() - t0, 1)}))
    print(f"umbral {thr:.3f}\n", pd.DataFrame(ideal).to_string(), "\n", semi.round(3).to_string())


# ----------------------------------------------------------------------------------------------- curva
def part_curve(cfg, allf, out, n_jobs):
    inc = allf[allf["included"]].reset_index(drop=True)
    cols = [c for c in allf.columns if family_of(c)]
    y = (inc["group"] == "MDD").astype(int).to_numpy()
    g = inc["subject_id"].to_numpy()
    sets = {"Primary LR-L2 (EEG 288)": inc[cols].to_numpy(float), "Demographics LR": inc[DEMO].to_numpy(float)}
    pipe, grid = _tab(_lr("l2")), {"clf__C": C_GRID}
    jobs = []
    for r, k, tr, te in outer_splits(y, g, N_SPLITS, N_REPEATS):
        for n_tr in (12, 18, 24, 30, 36):
            rng = np.random.default_rng(100 * r + 10 * k + n_tr)
            pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
            n_pos = max(3, int(round(n_tr * len(pos) / len(tr))))
            sub = np.r_[rng.choice(pos, min(n_pos, len(pos)), replace=False),
                        rng.choice(neg, min(n_tr - n_pos, len(neg)), replace=False)]
            for name in sets:
                jobs.append((name, r, k, n_tr, sub, te))

    def one(name, r, k, n_tr, sub, te):
        n_inner = int(min(N_INNER, y[sub].sum(), (1 - y[sub]).sum()))
        idx = np.r_[sub, te]
        _, prob, _ = _fit_predict(pipe, grid, sets[name][idx], y[idx], g[idx], np.arange(len(sub)),
                                  np.arange(len(sub), len(idx)), n_inner, 1000 * r + k, "roc_auc")
        return name, r, n_tr, g[te], y[te], prob

    res = Parallel(n_jobs=n_jobs)(delayed(one)(*j) for j in jobs)
    rows = []
    df = pd.DataFrame([(n, r, t, gi, yi, pi) for n, r, t, G, Y, P in res for gi, yi, pi in zip(G, Y, P)],
                      columns=["model", "repeat", "n_train", "subject_id", "y", "prob"])
    for (n, t, r), d in df.groupby(["model", "n_train", "repeat"]):
        rows.append({"model": n, "n_train": t, "repeat": r, "auc": subject_metrics(d["y"], d["prob"])["auc"]})
    lc = pd.DataFrame(rows).groupby(["model", "n_train"])["auc"].agg(["mean", "std"]).reset_index()
    lc.to_csv(out / "learning_curve.csv", index=False)
    print(lc.round(3).to_string())


# ----------------------------------------------------------------------------------------------- identificación
def part_ident(cfg, feat_dir, out, n_jobs):
    from pyriemann.tangentspace import TangentSpace
    from sklearn.model_selection import StratifiedKFold
    covs = np.load(feat_dir / "trial_regions_cov.npy", mmap_mode="r")
    z = np.load(feat_dir / "trial_regions.npz", allow_pickle=True)
    gt = z["subject_id"].astype(str)
    C = np.asarray(covs)
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(C, gt):
        ts = TangentSpace(metric="riemann").fit(C[tr])
        sc = StandardScaler().fit(ts.transform(C[tr]))
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(ts.transform(C[tr])), gt[tr])
        pred = clf.predict(sc.transform(ts.transform(C[te])))
        accs.append(float((pred == gt[te]).mean()))
        print(f"  identificación de sujeto, pliegue: acc={accs[-1]:.3f}", flush=True)
    r = {"task": "identify participant (45 classes) from single-trial regional covariances, trial-wise 5-fold",
         "accuracy_mean": float(np.mean(accs)), "accuracy_folds": accs, "chance": 1 / len(set(gt)),
         "n_trials": int(len(gt))}
    (out / "subject_identification.json").write_text(json.dumps(r, indent=2))
    print(r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="*", default=["confounds", "models", "power", "curve", "ident"])
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-sims", type=int, default=200)
    args = ap.parse_args()
    limit_threads()
    n_jobs = safe_n_jobs(0.4, max_jobs=args.n_jobs)
    print(f"Procesos: {n_jobs} (RAM libre {free_gb():.1f} GB)", flush=True)
    cfg = load_config(args.config)
    out = Path(cfg["paths"]["reports_dir"]) / "results" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    allf, compact, feat_dir = load(cfg)
    for part in args.only:
        t0 = time.time()
        print(f"=== {part}", flush=True)
        if part == "confounds":
            part_confounds(cfg, allf, out, n_jobs, args.n_perm)
        elif part == "models":
            part_models(cfg, allf, compact, feat_dir, out, n_jobs, args.n_perm)
        elif part == "power":
            part_power(cfg, allf, out, n_jobs, args.n_sims)
        elif part == "curve":
            part_curve(cfg, allf, out, n_jobs)
        elif part == "ident":
            part_ident(cfg, feat_dir, out, n_jobs)
        print(f"=== {part}: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
