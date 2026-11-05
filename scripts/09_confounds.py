"""Paso 9: confusores y sesgos.

1. ¿Las características de EEG predicen sexo, edad (corte en la mediana) o período de registro?
2. Desempeño del modelo primario por sexo.
3. Sensibilidad: modelo primario solo con los sujetos registrados en jun.-ago. 2015.
4. Calidad de señal por grupo (canales malos, componentes ICA quitados, ensayos y segmentos válidos).
5. ¿La probabilidad del modelo primario se asocia con la escolaridad o la edad dentro de cada grupo?
"""

import argparse
import json
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats

from eeg_mdd.config import load_config
from eeg_mdd.resources import free_gb, limit_threads, safe_n_jobs
from eeg_mdd.demographics import recording_datetime_from_name
from eeg_mdd.evaluation import nested_cv, permutation_test, subject_metrics
from eeg_mdd.features import family_of
from eeg_mdd.matching import load_clinical
from eeg_mdd.models import tabular_models


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args()
    limit_threads()
    args.n_jobs = safe_n_jobs(0.5, max_jobs=args.n_jobs)
    print(f"Procesos: {args.n_jobs} (RAM libre {free_gb():.1f} GB)", flush=True)

    cfg = load_config(args.config)
    feat_dir = Path(cfg["paths"]["processed_dir"]) / "features"
    reports = Path(cfg["paths"]["reports_dir"])
    out = reports / "results" / "confounds"
    out.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(feat_dir / "subject_features.parquet").sort_values("subject_id").reset_index(drop=True)
    feats = feats[feats["included"]].reset_index(drop=True)
    cols = [c for c in feats.columns if family_of(c)]
    g = feats["subject_id"].to_numpy()
    cc = cfg["dataset"]["clinical"]
    clin = load_clinical(cc["file"], cc["id_column"], cc["group_column"], cc["group_map"]).set_index("clinical_id")
    audit = pd.read_csv(Path(cfg["paths"]["interim_dir"]) / "audit.csv", dtype={"subject_id": str}).set_index("subject_id")

    meta = pd.DataFrame({"subject_id": g, "group": feats["group"]})
    meta["age"] = pd.to_numeric(clin.loc[g, "age"], errors="coerce").to_numpy()
    meta["male"] = (clin.loc[g, "gender"].str.strip() == "M").astype(int).to_numpy()
    meta["education"] = pd.to_numeric(clin.loc[g, "education（years）"], errors="coerce").to_numpy()
    meta["rec_date"] = [recording_datetime_from_name(audit.loc[s, "file"])[0] for s in g]
    meta["late_period"] = (meta["rec_date"] >= pd.Timestamp("2015-09-01")).astype(int)
    meta.to_csv(out / "subject_meta.csv", index=False)

    pipe, grid = tabular_models()["LR-L2"]
    rows = []
    targets = {
        "sex (male)": meta["male"].to_numpy(),
        "age > median": (meta["age"] > meta["age"].median()).astype(int).to_numpy(),
        "recording period (>= Sep 2015)": meta["late_period"].to_numpy(),
    }
    for tname, yt in targets.items():
        res = nested_cv(tname, pipe, grid, feats[cols], yt, g, 5, 10, 5, n_jobs=args.n_jobs)
        ci = res.ci()
        perm = permutation_test(pipe, grid, feats[cols], yt, g, float(res.repeat_scores["auc"].iloc[0]),
                                n_perm=args.n_perm, n_jobs=args.n_jobs)
        rows.append({"target": tname, "n_pos": int(yt.sum()), "n": len(yt), "auc": ci.loc["auc", "estimate"],
                     "auc_ci_low": ci.loc["auc", "ci_low"], "auc_ci_high": ci.loc["auc", "ci_high"],
                     "balanced_accuracy": ci.loc["balanced_accuracy", "estimate"], "p_perm": perm["p_value"],
                     "n_perm": args.n_perm})
        print(f"EEG -> {tname}: AUC={rows[-1]['auc']:.3f} [{rows[-1]['auc_ci_low']:.3f}, "
              f"{rows[-1]['auc_ci_high']:.3f}] p={perm['p_value']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(out / "eeg_predicts_confounds.csv", index=False)

    # Crosstab grupo x período
    pd.crosstab(meta["late_period"], meta["group"]).to_csv(out / "group_by_period.csv")

    # 2. Desempeño por sexo (predicciones OOF del modelo primario)
    oof = pd.read_csv(reports / "results" / "models" / "PRIMARY_LR-L2_all" / "oof.csv", dtype={"subject_id": str})
    oof = oof.merge(meta[["subject_id", "male", "age", "education", "late_period"]], on="subject_id")
    by_sex = []
    for sex, d in oof.groupby("male"):
        reps = pd.DataFrame([subject_metrics(dd["y"], dd["prob"]) for _, dd in d.groupby("repeat")])
        by_sex.append({"sex": "male" if sex else "female", "n_subjects": d["subject_id"].nunique(),
                       "n_mdd": int(d.drop_duplicates("subject_id")["y"].sum()), **reps.mean().to_dict()})
    pd.DataFrame(by_sex).to_csv(out / "primary_by_sex.csv", index=False)
    by_period = []
    for per, d in oof.groupby("late_period"):
        reps = pd.DataFrame([subject_metrics(dd["y"], dd["prob"]) for _, dd in d.groupby("repeat")])
        by_period.append({"period": ">= Sep 2015" if per else "< Sep 2015", "n_subjects": d["subject_id"].nunique(),
                          "n_mdd": int(d.drop_duplicates("subject_id")["y"].sum()), **reps.mean().to_dict()})
    pd.DataFrame(by_period).to_csv(out / "primary_by_period.csv", index=False)

    # 5. Asociación de la probabilidad con escolaridad / edad dentro de cada grupo
    mp = oof.groupby("subject_id").agg(prob=("prob", "mean"), y=("y", "first"), education=("education", "first"),
                                       age=("age", "first")).reset_index()
    assoc = []
    for grp, d in mp.groupby("y"):
        for v in ("education", "age"):
            r, p = stats.spearmanr(d["prob"], d[v], nan_policy="omit")
            assoc.append({"group": "MDD" if grp else "HC", "variable": v, "spearman_r": r, "p": p, "n": len(d)})
    pd.DataFrame(assoc).to_csv(out / "prob_vs_demographics.csv", index=False)

    # 3. Sensibilidad: solo jun.-ago. 2015
    sub = meta["rec_date"].between(pd.Timestamp("2015-06-01"), pd.Timestamp("2015-08-31")).to_numpy()
    ys = (feats.loc[sub, "group"] == "MDD").astype(int).to_numpy()
    res = nested_cv("Primary Jun-Aug 2015", pipe, grid, feats.loc[sub, cols], ys, g[sub], 5, 10, 5, n_jobs=args.n_jobs)
    ci = res.ci()
    perm = permutation_test(pipe, grid, feats.loc[sub, cols], ys, g[sub], float(res.repeat_scores["auc"].iloc[0]),
                            n_perm=args.n_perm, n_jobs=args.n_jobs)
    sens = {"subset": "recorded Jun-Aug 2015", "n": int(sub.sum()), "n_mdd": int(ys.sum()),
            **{f"{m}": float(ci.loc[m, "estimate"]) for m in ci.index},
            "auc_ci_low": float(ci.loc["auc", "ci_low"]), "auc_ci_high": float(ci.loc["auc", "ci_high"]),
            "p_perm": perm["p_value"], "n_perm": args.n_perm}
    (out / "sensitivity_jun_aug_2015.json").write_text(json.dumps(sens, indent=2))
    print(f"Sensibilidad jun-ago 2015 (n={sens['n']}, MDD={sens['n_mdd']}): AUC={sens['auc']:.3f} "
          f"[{sens['auc_ci_low']:.3f}, {sens['auc_ci_high']:.3f}] p={sens['p_perm']:.4f}")

    # 4. Calidad de señal por grupo (todos los sujetos preprocesados, incluidos los excluidos)
    log = pd.read_csv(reports / "preprocessing_log.csv", dtype={"subject_id": str})
    q = []
    for v in ("n_bad_channels", "ica_n_excluded", "n_trials_kept_total", "n_segments_kept"):
        a = log.loc[log["group"] == "MDD", v].dropna()
        b = log.loc[log["group"] == "HC", v].dropna()
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        q.append({"variable": v, "MDD_median": a.median(), "MDD_iqr": f"{a.quantile(.25):.0f}-{a.quantile(.75):.0f}",
                  "HC_median": b.median(), "HC_iqr": f"{b.quantile(.25):.0f}-{b.quantile(.75):.0f}",
                  "U": u.statistic, "p": u.pvalue})
    pd.DataFrame(q).to_csv(out / "quality_by_group.csv", index=False)
    print(pd.DataFrame(q).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
