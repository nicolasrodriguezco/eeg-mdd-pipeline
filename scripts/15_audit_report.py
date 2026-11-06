"""Paso 15 (D6): macros, tabla y figuras de la auditoría para el artículo.

Salidas:
- ``docs/paper/generated/audit_numbers.tex``: macros ``\\Aud...``;
- ``docs/paper/tables/audit_models.tex``;
- ``docs/paper/figures/fig_power_curve.pdf`` y ``fig_emg.pdf``.
"""

import json
from pathlib import Path

import _common  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "reports" / "results" / "audit"
PAPER = ROOT / "docs" / "paper"

LABELS = {"Demo+EEG288": "Demographics + all 288 EEG features",
          "Demo+Compact": "Demographics + compact EEG set",
          "EEG288-deconfounded": "EEG 288, age/sex/education regressed out in-fold",
          "EEG288-PCA-LR": "EEG 288, PCA (in-fold) + LR",
          "Compact-LR": "Compact theory-driven EEG set (36)",
          "Behaviour-RT-LR": "Reaction times only (8)",
          "Subject-Riemann-TS-LR": "Subject-mean covariance, tangent space + LR"}


def f2(x):
    return f"{x:.2f}"


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main() -> None:
    m: dict[str, str] = {}
    # --- confusores
    ed = pd.read_csv(A / "eeg_to_education.csv").iloc[0]
    m.update(AudEduAuc=f2(ed["auc"]), AudEduLo=f2(ed["auc_lo"]), AudEduHi=f2(ed["auc_hi"]), AudEduP=fp(ed["p_perm"]),
             AudEduNsig=str(int(ed["n_feat_p05_education"])), AudEduNfdr=str(int(ed["n_feat_fdr_education"])))
    q = pd.read_csv(A / "group_tests_quality_emg_behaviour.csv")
    key = {"n_trials_kept_total": "Trials", "n_bad_channels": "BadCh", "ica_n_excluded": "Ica",
           "ica_n_muscle_artifact": "IcaMuscle", "emg_rel_global": "EmgRel", "emg_logpow_global": "EmgPow",
           "beh_rt_median": "Rt", "beh_rt_sd": "RtSd", "beh_miss_rate": "Miss", "beh_rt_sad_minus_happy": "RtSadHappy",
           "ap_exponent_global": "ApExp", "alpha_peak_freq_post": "Apf", "fmtheta_periodic": "FmTheta"}
    for _, r in q.iterrows():
        if r["variable"] in key:
            s = "All" if r["sample"].startswith("all") else "Inc"
            m[f"AudQ{key[r['variable']]}{s}P"] = fp(r["p_mwu"])
            m[f"AudQ{key[r['variable']]}{s}D"] = f2(r["cohen_d"])
    rt = q[(q["variable"] == "beh_rt_median") & (q["sample"] == "included 45")].iloc[0]
    m.update(AudRtMdd=f"{rt['mdd_median']:.0f}", AudRtHc=f"{rt['hc_median']:.0f}")
    e = pd.read_csv(A / "emg_vs_features.csv").set_index("feature")
    for f, k in (("cplxglob_sampen", "Sampen"), ("cplxglob_lzc", "Lzc"), ("relglob_beta", "Beta"),
                 ("relglob_lowgamma", "Gamma"), ("ap_exponent_global", "ApExp"),
                 ("wpli_theta_central_L__temporal_L", "Wpli")):
        m[f"AudEmgR{k}"] = f2(e.loc[f, "spearman_r_with_emg"])
        m[f"AudEmgP{k}"] = fp(e.loc[f, "p"])
    d45 = pd.read_csv(A / "demographics_included45.csv").set_index("variable")
    m.update(AudDemoEduMdd=d45.loc["education", "mdd"].replace("±", "$\\pm$"),
             AudDemoEduHc=d45.loc["education", "hc"].replace("±", "$\\pm$"), AudDemoEduP=fp(d45.loc["education", "p"]),
             AudDemoAgeMdd=d45.loc["age", "mdd"].replace("±", "$\\pm$"),
             AudDemoAgeHc=d45.loc["age", "hc"].replace("±", "$\\pm$"), AudDemoAgeP=fp(d45.loc["age", "p"]),
             AudDemoSexMdd=d45.loc["sex F/M", "mdd"], AudDemoSexHc=d45.loc["sex F/M", "hc"],
             AudDemoSexP=fp(d45.loc["sex F/M", "p"]))
    # --- modelos
    t = pd.read_csv(A / "exploratory_models.csv")
    rows = []
    for _, r in t.iterrows():
        k = r["model"].replace("+", "Plus").replace("-", "").replace("288", "All")
        m[f"Aud{k}Auc"], m[f"Aud{k}Lo"], m[f"Aud{k}Hi"] = f2(r["auc"]), f2(r["auc_lo"]), f2(r["auc_hi"])
        m[f"Aud{k}P"], m[f"Aud{k}Pholm"] = fp(r["p_perm"]), fp(r["p_holm"])
        m[f"Aud{k}DDemo"], m[f"Aud{k}PDemo"] = f2(r["dAUC_vs_demo"]), fp(r["p_vs_demo"])
        rows.append(f"{LABELS[r['model']]} & {f2(r['auc'])} [{f2(r['auc_lo'])}, {f2(r['auc_hi'])}] & "
                    f"{f2(r['balanced_accuracy'])} & {fp(r['p_perm'])} & {fp(r['p_holm'])} & "
                    f"{r['dAUC_vs_demo']:+.2f} ({fp(r['p_vs_demo'])}) \\\\")
    tab = ("\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
           "Model (exploratory, post hoc) & AUC [95\\% CI] & BalAcc & $p_\\text{perm}$ & $p_\\text{Holm}$ & "
           "$\\Delta$AUC vs demo ($p_\\text{corr}$) \\\\\n\\midrule\n" + "\n".join(rows)
           + "\n\\bottomrule\n\\end{tabular}")
    (PAPER / "tables" / "audit_models.tex").write_text(tab, encoding="utf-8")
    m["AudMinP"] = fp(t["p_perm"].min())
    m["AudMinPholm"] = fp(t["p_holm"].min())
    # --- poder
    pm = json.loads((A / "power_meta.json").read_text())
    ideal = pd.read_csv(A / "power_ideal.csv").set_index("true_auc")["power_ideal_score"]
    semi = pd.read_csv(A / "power_semisynthetic.csv")
    m["AudPowThr"] = f2(pm["threshold_null_q95"])
    for a, w in ((0.65, "SixtyFive"), (0.7, "Seventy"), (0.75, "SeventyFive"), (0.8, "Eighty")):
        m[f"AudPowIdeal{w}"] = f"{100 * ideal.loc[a]:.0f}"
    for _, r in semi.iterrows():
        k = {0.0: "Zero", 0.3: "Three", 0.5: "Five", 0.8: "Eight", 1.0: "Ten"}[r["d_per_feature"]]
        m[f"AudSemi{k}Oracle"], m[f"AudSemi{k}Cv"] = f2(r["oracle_auc"]), f2(r["cv_auc_mean"])
        m[f"AudSemi{k}Power"] = f"{100 * r['power']:.0f}"
    m["AudSemiNsims"] = str(pm["n_sims"])
    lc = pd.read_csv(A / "learning_curve.csv")
    for name, k in (("Primary LR-L2 (EEG 288)", "Prim"), ("Demographics LR", "Demo")):
        d = lc[lc["model"] == name].set_index("n_train")["mean"]
        m[f"AudLc{k}Min"], m[f"AudLc{k}Max"] = f2(d.loc[12]), f2(d.loc[36])
    ident = json.loads((A / "subject_identification.json").read_text())
    m["AudIdentAcc"] = f"{100 * ident['accuracy_mean']:.1f}"
    m["AudIdentChance"] = f"{100 * ident['chance']:.1f}"
    ss = pd.read_csv(A / "sample_size_needed.csv").set_index("true_auc")["n_for_80pct_power"]
    for a, w in ((0.6, "Sixty"), (0.65, "SixtyFive"), (0.7, "Seventy")):
        m[f"AudNNeed{w}"] = str(int(ss.loc[a]))
    rf = A / "rawrf_permutation.json"
    if rf.exists():
        r = json.loads(rf.read_text())
        m.update(AudRawRfObs=f2(r["observed_auc_rep0"]), AudRawRfP=fp(r["p_value"]), AudRawRfN=str(r["n_perm"]),
                 AudRawRfQ=f2(r["null_q95"]))
        # Holm sobre la familia exploratoria (7 modelos de la tabla + este RF)
        from scipy.stats import beta as _beta
        from statsmodels.stats.multitest import multipletests
        ps = list(t["p_perm"]) + [r["p_value"]]
        m["AudRawRfPholm"] = fp(multipletests(ps, method="holm")[1][-1])
        m["AudRawRfFamily"] = str(len(ps))
        k = round(r["p_value"] * (r["n_perm"] + 1) - 1)  # permutaciones con AUC >= observado
        m["AudRawRfExceed"] = str(k)
        m["AudRawRfPciHi"] = f2(_beta.ppf(0.975, k + 1, r["n_perm"] - k))
    (PAPER / "generated" / "audit_numbers.tex").write_text(
        "% Generado por scripts/15_audit_report.py: no editar a mano.\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())) + "\n", encoding="utf-8")
    print(f"{len(m)} macros")

    # --- figura: poder y curva de aprendizaje
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    ax[0].plot(ideal.index, 100 * ideal.values, "o-", color="#2a6f97", label="ideal score (no learning)")
    ax[0].plot(semi["oracle_auc"], 100 * semi["power"], "s--", color="#c44e52",
               label="primary pipeline (semi-synthetic)")
    ax[0].axhline(80, color="grey", lw=0.6, ls=":")
    ax[0].set(xlabel="true (oracle) AUC", ylabel="power (%)", ylim=(0, 102),
              title=f"Detection at null 95th pct ({pm['threshold_null_q95']:.2f})")
    ax[0].legend(frameon=False, fontsize=7)
    for name, c in (("Primary LR-L2 (EEG 288)", "#c44e52"), ("Demographics LR", "#2a6f97")):
        d = lc[lc["model"] == name]
        ax[1].errorbar(d["n_train"], d["mean"], yerr=d["std"], marker="o", capsize=2, color=c, label=name)
    ax[1].axhline(0.5, color="grey", lw=0.6, ls="--")
    ax[1].set(xlabel="training participants per fold", ylabel="participant-level AUC", title="Learning curve")
    ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "fig_power_curve.pdf")
    fig.savefig(A.parent / "figures" / "fig_power_curve.png", dpi=200)

    # --- figura: EMG frente a las variables
    feats = pd.read_parquet(ROOT / "data" / "processed" / "features" / "audit_features.parquet")
    feats = feats[feats["included"]]
    fig, ax = plt.subplots(1, 3, figsize=(7, 2.3))
    for a_, (f, lab) in zip(ax, (("cplxglob_sampen", "mean sample entropy"), ("relglob_lowgamma",
                                                                              "relative 30–40 Hz power"),
                                 ("ap_exponent_global", "aperiodic exponent"))):
        for grp, c in (("HC", "#2a6f97"), ("MDD", "#c44e52")):
            d = feats[feats["group"] == grp]
            a_.scatter(d["emg_rel_global"], d[f], s=10, color=c, label=grp, alpha=0.8)
        a_.set(xlabel="EMG index (log 55–95 / 1–40 Hz)", ylabel=lab)
        a_.set_title(f"$\\rho$ = {e.loc[f, 'spearman_r_with_emg']:.2f}", fontsize=8)
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "fig_emg.pdf")
    fig.savefig(A.parent / "figures" / "fig_emg.png", dpi=200)
    print("figuras listas")


if __name__ == "__main__":
    main()
