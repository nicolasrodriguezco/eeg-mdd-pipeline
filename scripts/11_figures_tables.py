"""Paso 11: figuras y tablas del artículo, leídas SOLO de las salidas de los pasos anteriores.

Salidas: reports/results/figures/*.pdf|png, reports/results/tables/*.tex|csv,
y docs/paper/generated/numbers.tex (macros con cada número citado en el texto).
"""

import argparse
import json
from pathlib import Path

import _common  # noqa: F401
import mne
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from eeg_mdd import plotting as P
from eeg_mdd.config import load_config
from eeg_mdd.evaluation import METRICS
from eeg_mdd.features.regions import region_table

DISPLAY = {
    "PRIMARY_LR-L2_all": ("LR-L2, all features (primary)", "Handcrafted features"),
    "LR-L1_all": ("LR-L1, all features", "Handcrafted features"),
    "SVM-linear_all": ("Linear SVM, all features", "Handcrafted features"),
    "SVM-RBF_all": ("RBF SVM, all features", "Handcrafted features"),
    "RandomForest_all": ("Random forest, all features", "Handcrafted features"),
    "LightGBM_all": ("LightGBM, all features", "Handcrafted features"),
    "NestedModelSelection_all": ("Nested model selection, all features", "Handcrafted features"),
    "LR-L2_spectral": ("LR-L2, spectral + FAA", "Feature families"),
    "LR-L2_erp": ("LR-L2, ERP", "Feature families"),
    "LR-L2_connectivity": ("LR-L2, wPLI connectivity", "Feature families"),
    "LR-L2_complexity": ("LR-L2, complexity", "Feature families"),
    "TS-LR": ("Tangent space + LR", "Riemannian (trial level)"),
    "MDM": ("MDM", "Riemannian (trial level)"),
    "ERPCov-TS-LR": ("ERP covariances + TS + LR", "Riemannian (trial level)"),
    "DL-EEGNet": ("EEGNet", "Deep learning (trial level)"),
    "DL-ShallowFBCSPNet": ("ShallowFBCSPNet", "Deep learning (trial level)"),
    "DL-TCN": ("TCN", "Deep learning (trial level)"),
    "Demographics-LR": ("Age + sex + education (LR)", "Demographic baseline"),
}
CATEGORY_ORDER = ["Demographic baseline", "Handcrafted features", "Feature families", "Riemannian (trial level)",
                  "Deep learning (trial level)"]


def load_models(models_dir: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(models_dir.iterdir()):
        if not (d / "ci.csv").exists():
            continue
        ci = pd.read_csv(d / "ci.csv", index_col=0)
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        rep = pd.read_csv(d / "repeat_scores.csv")
        name, cat = DISPLAY.get(d.name, (d.name, "Other"))
        row = {"key": d.name, "model": name, "category": cat, "n_repeats": rep["repeat"].nunique(),
               "p_perm": meta.get("p_value", np.nan), "n_perm": meta.get("n_perm", 0),
               "null_q95": meta.get("null_q95", np.nan), "auc_sd_repeats": rep["auc"].std()}
        for m in METRICS:
            row[m] = ci.loc[m, "estimate"]
            row[f"{m}_lo"] = ci.loc[m, "ci_low"]
            row[f"{m}_hi"] = ci.loc[m, "ci_high"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["cat_rank"] = df["category"].map({c: i for i, c in enumerate(CATEGORY_ORDER)}).fillna(99)
    return df.sort_values(["cat_rank", "auc"], ascending=[True, False]).reset_index(drop=True)


def fmt_ci(v, lo, hi, nd=2):
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def fmt_p(p, n):
    if not np.isfinite(p) or not n:
        return "--"
    floor = 1 / (n + 1)
    return f"$<${floor:.3f}" if p <= floor + 1e-12 else f"{p:.3f}"


def latex_main_table(df: pd.DataFrame, path: Path) -> None:
    lines = [r"\begin{tabular}{@{}llllllr@{}}", r"\toprule",
             r"Model & AUC [95\% CI] & Bal.\ acc.\ [95\% CI] & Sens. & Spec. & Brier & $p_\text{perm}$ \\",
             r"\midrule"]
    for cat in CATEGORY_ORDER:
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{cat}}}}} \\")
        for _, r in sub.iterrows():
            name = r["model"].replace("&", r"\&")
            if "primary" in name:
                name = r"\textbf{" + name + "}"
            perm = fmt_p(r["p_perm"], r["n_perm"]) + (f" ({int(r['n_perm'])})" if r["n_perm"] else "")
            lines.append(f"\\quad {name} & {fmt_ci(r['auc'], r['auc_lo'], r['auc_hi'])} & "
                         f"{fmt_ci(r['balanced_accuracy'], r['balanced_accuracy_lo'], r['balanced_accuracy_hi'])} & "
                         f"{r['sensitivity']:.2f} & {r['specificity']:.2f} & {r['brier']:.3f} & {perm} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def fig_model_comparison(df: pd.DataFrame, path: Path) -> None:
    fig, axes = P.plt.subplots(1, 2, figsize=(7.2, 0.24 * len(df) + 1.0), sharey=True)
    ypos = np.arange(len(df))[::-1]
    cat_color = {c: P.SERIES[i] for i, c in enumerate(CATEGORY_ORDER)}
    for ax, m, lab in ((axes[0], "auc", "Subject-level ROC AUC"),
                       (axes[1], "balanced_accuracy", "Subject-level balanced accuracy")):
        ax.axvline(0.5, color=P.INK2, lw=0.9, ls="--")
        for yv, (_, r) in zip(ypos, df.iterrows()):
            c = cat_color.get(r["category"], P.INK2)
            ax.plot([r[f"{m}_lo"], r[f"{m}_hi"]], [yv, yv], color=c, lw=2, solid_capstyle="round")
            ax.plot(r[m], yv, "o", ms=6 if "primary" in r["model"] else 4.5, color=c, mec="white", mew=1.2)
        ax.set_xlabel(lab)
        ax.set_xlim(0.15, 1.0)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, df["model"], fontsize=7)
    for lbl in axes[0].get_yticklabels():
        if "primary" in lbl.get_text():
            lbl.set_fontweight("bold")
    handles = [P.plt.Line2D([], [], color=cat_color[c], marker="o", lw=2, label=c) for c in CATEGORY_ORDER
               if c in set(df["category"])]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.55, -0.06))
    axes[0].text(0.5, ypos.max() + 0.9, "chance", fontsize=6.5, color=P.INK2, ha="center")
    P.save(fig, path)


def fig_roc(models_dir: Path, keys: list[str], path: Path) -> None:
    fig, ax = P.plt.subplots(figsize=(3.4, 3.2))
    grid = np.linspace(0, 1, 201)
    for i, key in enumerate(keys):
        d = models_dir / key
        if not (d / "oof.csv").exists():
            continue
        oof = pd.read_csv(d / "oof.csv")
        tprs = []
        for _, rr in oof.groupby("repeat"):
            fpr, tpr, _ = roc_curve(rr["y"], rr["prob"])
            tprs.append(np.interp(grid, fpr, tpr))
            tprs[-1][0] = 0
        tprs = np.array(tprs)
        ci = pd.read_csv(d / "ci.csv", index_col=0)
        col = P.SERIES[[0, 1, 2, 6][i % 4]]
        ax.plot(grid, tprs.mean(0), color=col, lw=2,
                label=f"{DISPLAY.get(key, (key,))[0].replace(', all features', '')} (AUC {ci.loc['auc', 'estimate']:.2f})")
        if len(tprs) > 1:
            ax.fill_between(grid, np.percentile(tprs, 2.5, 0), np.percentile(tprs, 97.5, 0), color=col, alpha=0.15, lw=0)
    ax.plot([0, 1], [0, 1], color=P.INK2, lw=0.8, ls="--")
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_aspect("equal")
    ax.legend(fontsize=6, loc="lower right")
    ax.set_title("Out-of-fold subject-level ROC")
    P.save(fig, path)


def fig_permutation(models_dir: Path, keys: list[str], path: Path) -> None:
    keys = [k for k in keys if (models_dir / k / "perm_null.npy").exists()]
    fig, axes = P.plt.subplots(1, len(keys), figsize=(2.3 * len(keys), 2.2), sharey=True, squeeze=False,
                               gridspec_kw=dict(wspace=0.12))
    for ax, key in zip(axes[0], keys):
        null = np.load(models_dir / key / "perm_null.npy")
        meta = json.loads((models_dir / key / "meta.json").read_text())
        ax.hist(null, bins=np.linspace(0, 1, 41), color=P.SERIES[0], alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.axvline(meta["observed_auc"], color=P.SERIES[1], lw=2)
        ax.set_title(DISPLAY.get(key, (key,))[0].replace(", all features", "").replace(" (LR)", ""), fontsize=7.5)
        ax.text(0.03, 0.95, f"observed {meta['observed_auc']:.2f}\np = {fmt_p(meta['p_value'], meta['n_perm'])}\n"
                f"{meta['n_perm']} perms", transform=ax.transAxes, fontsize=6.5, color=P.INK, va="top")
        ax.set_xlim(0, 1)
        ax.set_xticks([0.25, 0.5, 0.75])
        ax.set_xlabel("AUC")
    axes[0][0].set_ylabel("Count (permuted labels)")
    P.save(fig, path)


def fig_inflation(ctrl: pd.DataFrame, path: Path) -> pd.DataFrame:
    s = ctrl.groupby(["model", "split"])[["trial_auc", "subject_auc"]].agg(["mean", "std"])
    fig, axes = P.plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    models = list(dict.fromkeys(ctrl["model"]))
    w = 0.36
    for ax, level, title in ((axes[0], "trial_auc", "Trial level"), (axes[1], "subject_auc", "Participant level")):
        for i, (split, col) in enumerate((("subject-wise", P.SERIES[0]), ("trial-wise (INVALID)", P.SERIES[1]))):
            vals = [s.loc[(m, split), (level, "mean")] for m in models]
            err = [s.loc[(m, split), (level, "std")] for m in models]
            x = np.arange(len(models)) + (i - 0.5) * w
            ax.bar(x, vals, w * 0.92, color=col, yerr=err, error_kw=dict(lw=0.8, ecolor=P.INK2),
                   label="subject-wise split" if i == 0 else "trial-wise split (invalid)")
            for xx, v in zip(x, vals):
                ax.text(xx, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5, color=P.INK)
        ax.axhline(0.5, color=P.INK2, lw=0.8, ls="--")
        ax.set_xticks(np.arange(len(models)),
                      [m.replace(" (old approach)", "\n(previous approach)") for m in models], fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0.3, 1.1)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("ROC AUC")
    axes[0].legend(fontsize=6.5, loc="upper left")
    P.save(fig, path)
    return s


def fig_regions(path: Path) -> None:
    t = region_table()
    mont = mne.channels.make_standard_montage("GSN-HydroCel-129")
    info = mne.create_info(mont.ch_names, 250.0, "eeg")
    info.set_montage(mont)
    lay = mne.channels.make_eeg_layout(info)
    pos2d = lay.pos[:, :2] + lay.pos[:, 2:] / 2          # centro de cada caja
    pos2d = pos2d - pos2d.mean(axis=0)
    order = {n: i for i, n in enumerate(lay.names)}
    pos2d = pos2d[[order[c] for c in mont.ch_names]]
    lobes = ["frontal", "central", "temporal", "parietal", "occipital"]
    fig, ax = P.plt.subplots(figsize=(3.3, 3.3))
    for i, ch in enumerate(mont.ch_names):
        reg = t.loc[ch, "region"]
        x, yv = pos2d[i]
        if reg in ("midline", "excluded"):
            ax.plot(x, yv, "x" if reg == "excluded" else "o", ms=3.5, color="#9a9994", mfc="none" if reg == "midline" else None)
            continue
        lobe, side = reg.split("_")
        c = P.SERIES[lobes.index(lobe)]
        ax.plot(x, yv, "o", ms=5, color=c, mfc=c if side == "L" else "white", mew=1.4)
    circ = P.plt.Circle((0, 0), np.abs(pos2d).max() * 1.05, fill=False, color=P.INK2, lw=0.8)
    ax.add_patch(circ)
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [P.plt.Line2D([], [], marker="o", ls="", color=P.SERIES[i], label=lobe) for i, lobe in enumerate(lobes)]
    handles += [P.plt.Line2D([], [], marker="o", ls="", color=P.INK2, mfc="white", label="right (open)"),
                P.plt.Line2D([], [], marker="o", ls="", color="#9a9994", mfc="none", label="midline (unused)"),
                P.plt.Line2D([], [], marker="x", ls="", color="#9a9994", label="face/neck (unused)")]
    ax.legend(handles=handles, fontsize=6, loc="upper left", bbox_to_anchor=(0.92, 1.0))
    ax.set_title("HydroCel-129 regions (filled = left, open = right)", fontsize=7.5)
    P.save(fig, path)


def fig_quality(log: pd.DataFrame, path: Path, crit: dict) -> None:
    fig, axes = P.plt.subplots(1, 3, figsize=(7.2, 2.2))
    for ax, (v, lab, thr) in zip(axes, (("n_trials_kept_total", "Retained face-locked trials (of 480)",
                                          crit["min_trials_total"]),
                                         ("n_bad_channels", "Interpolated channels (of 129)",
                                          crit["max_bad_channel_frac"] * 129),
                                         ("ica_n_excluded", "ICA components removed (of 30)", None))):
        for i, grp in enumerate(("HC", "MDD")):
            vals = log.loc[log["group"] == grp, v].to_numpy()
            inc = log.loc[log["group"] == grp, "included"].to_numpy()
            jit = np.random.default_rng(i).uniform(-0.15, 0.15, len(vals))
            ax.scatter(i + jit[inc], vals[inc], s=12, color=P.GROUP_COLORS[grp], lw=0)
            ax.scatter(i + jit[~inc], vals[~inc], s=16, facecolor="white", edgecolor=P.GROUP_COLORS[grp], lw=1.2)
            ax.plot([i - 0.25, i + 0.25], [np.median(vals)] * 2, color=P.INK, lw=1.5)
        if thr:
            ax.axhline(thr, color=P.INK2, ls="--", lw=0.8)
        ax.set_xticks([0, 1], ["HC", "MDD"])
        ax.set_title(lab, fontsize=7.5)
        ax.grid(axis="x", visible=False)
    axes[0].scatter([], [], s=12, color=P.INK2, label="included")
    axes[0].scatter([], [], s=16, facecolor="white", edgecolor=P.INK2, label="excluded")
    axes[0].legend(fontsize=6.5, loc="lower left")
    P.save(fig, path)


def _comparisons(models_dir: Path, df: pd.DataFrame, macros: dict, key) -> list[str]:
    """ΔAUC por pliegue frente al primario y a la línea base demográfica (t corregido) + Holm sobre secundarios."""
    from statsmodels.stats.multitest import multipletests

    from eeg_mdd.evaluation.nested import corrected_ttest

    def folds(k):
        f = models_dir / k / "fold_scores.csv"
        return pd.read_csv(f).set_index(["repeat", "fold"])["auc"] if f.exists() else None

    ref = {"primary": folds("PRIMARY_LR-L2_all"), "demo": folds("Demographics-LR")}
    if any(v is None for v in ref.values()):
        return []
    n_test, n_train = 9, 36                     # 45 participantes, 5 pliegues externos
    sec = df[~df["key"].isin(["PRIMARY_LR-L2_all", "Demographics-LR"]) & (df["n_perm"] > 0)].copy()
    sec["p_holm"] = multipletests(sec["p_perm"].to_numpy(), method="holm")[1] if len(sec) else []
    holm = dict(zip(sec["key"], sec["p_holm"]))
    rows = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
            r" & \multicolumn{2}{c}{vs.\ primary} & \multicolumn{2}{c}{vs.\ demographics} & & \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"Model & $\Delta$AUC & $p_\text{corr}$ & $\Delta$AUC & $p_\text{corr}$ & $p_\text{perm}$ & $p_\text{Holm}$ \\",
            r"\midrule"]
    for _, r in df[df["key"] != "Demographics-LR"].iterrows():
        a = folds(r["key"])
        cells = []
        for nm, b in ref.items():
            if r["key"] == "PRIMARY_LR-L2_all" and nm == "primary":
                cells += ["--", "--"]
                continue
            j = pd.concat([a, b], axis=1, join="inner").dropna()
            t = corrected_ttest((j.iloc[:, 0] - j.iloc[:, 1]).to_numpy(), n_train, n_test)
            cells += [f"{t['mean_diff']:+.2f}", f"{t['p_value']:.3f}"]
            macros[f"Cmp{key(r['key'])}Vs{nm.title()}Diff"] = f"{t['mean_diff']:+.2f}"
            macros[f"Cmp{key(r['key'])}Vs{nm.title()}P"] = f"{t['p_value']:.3f}"
        ph = holm.get(r["key"], np.nan)
        if np.isfinite(ph):
            macros[f"{key(r['key'])}Pholm"] = f"{ph:.3f}"
        name = r["model"].replace("&", r"\&")
        rows.append(f"{name} & " + " & ".join(cells) + f" & {fmt_p(r['p_perm'], r['n_perm'])} & "
                    f"{(f'{ph:.3f}' if np.isfinite(ph) else '--')} \\\\")
    macros["NSecondaryHolm"] = str(len(sec))
    rows += [r"\bottomrule", r"\end{tabular}"]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    P.setup()
    cfg = load_config(args.config)
    reports = Path(cfg["paths"]["reports_dir"])
    res_dir = reports / "results"
    models_dir = res_dir / "models"
    figs, tabs = res_dir / "figures", res_dir / "tables"
    tabs.mkdir(parents=True, exist_ok=True)
    macros = {}

    df = load_models(models_dir)
    df.to_csv(tabs / "main_results.csv", index=False)
    latex_main_table(df, tabs / "main_results.tex")
    fig_model_comparison(df, figs / "fig_model_comparison")
    best = df[df["category"] != "Demographic baseline"].sort_values("auc", ascending=False).iloc[0]
    roc_keys = ["PRIMARY_LR-L2_all", "Demographics-LR"] + ([best["key"]] if best["key"] != "PRIMARY_LR-L2_all" else [])
    fig_roc(models_dir, roc_keys, figs / "fig_roc")
    fig_permutation(models_dir, ["PRIMARY_LR-L2_all", "Demographics-LR", best["key"]] if best["key"] !=
                    "PRIMARY_LR-L2_all" else ["PRIMARY_LR-L2_all", "Demographics-LR"], figs / "fig_permutation")
    print(df[["model", "auc", "auc_lo", "auc_hi", "balanced_accuracy", "p_perm", "n_perm", "n_repeats"]]
          .round(3).to_string(index=False))

    def key(name):
        return "".join(ch for ch in name.title() if ch.isalpha())

    for _, r in df.iterrows():
        k = key(r["key"])
        for m in ("auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "brier"):
            mk = key(m)
            macros[f"{k}{mk}"] = f"{r[m]:.2f}" if m != "brier" else f"{r[m]:.3f}"
            macros[f"{k}{mk}Lo"] = f"{r[f'{m}_lo']:.2f}" if m != "brier" else f"{r[f'{m}_lo']:.3f}"
            macros[f"{k}{mk}Hi"] = f"{r[f'{m}_hi']:.2f}" if m != "brier" else f"{r[f'{m}_hi']:.3f}"
        macros[f"{k}Pperm"] = fmt_p(r["p_perm"], r["n_perm"])
        macros[f"{k}Nperm"] = str(int(r["n_perm"]))
        macros[f"{k}AucSd"] = f"{r['auc_sd_repeats']:.2f}"
        if np.isfinite(r["null_q95"]):
            macros[f"{k}NullQ"] = f"{r['null_q95']:.2f}"
    macros["BestModelName"] = best["model"]
    macros["BestModelKey"] = key(best["key"])

    ctrl_path = res_dir / "control" / "inflation_control.csv"
    if ctrl_path.exists():
        ctrl = pd.read_csv(ctrl_path)
        s = fig_inflation(ctrl, figs / "fig_inflation")
        s.round(3).to_csv(tabs / "inflation_summary.csv")
        for (m, split), r in s.iterrows():
            k = key(m.split(" ")[0]) + ("TrialSplit" if split.startswith("trial") else "SubjectSplit")
            macros[f"Infl{k}TrialAuc"] = f"{r[('trial_auc', 'mean')]:.2f}"
            macros[f"Infl{k}SubjectAuc"] = f"{r[('subject_auc', 'mean')]:.2f}"

    log = pd.read_csv(reports / "preprocessing_log.csv", dtype={"subject_id": str})
    log["included"] = log["included"].astype(bool)
    fig_quality(log, figs / "fig_quality", cfg["exclusion"])
    fig_regions(figs / "fig_regions")
    macros["NIncluded"] = str(int(log["included"].sum()))
    macros["NIncludedMDD"] = str(int((log["included"] & (log["group"] == "MDD")).sum()))
    macros["NIncludedHC"] = str(int((log["included"] & (log["group"] == "HC")).sum()))
    macros["NExcludedMDD"] = str(int((~log["included"] & (log["group"] == "MDD")).sum()))
    macros["NExcludedHC"] = str(int((~log["included"] & (log["group"] == "HC")).sum()))
    for v, nm in (("n_trials_kept_total", "Trials"), ("n_bad_channels", "BadCh"), ("ica_n_excluded", "Ica")):
        inc = log[log["included"]]
        macros[f"Median{nm}"] = f"{inc[v].median():.0f}"
        macros[f"Min{nm}"] = f"{inc[v].min():.0f}"
        macros[f"Max{nm}"] = f"{inc[v].max():.0f}"
    from scipy.stats import fisher_exact

    tab = pd.crosstab(log["group"], log["included"])
    macros["ExclusionFisherP"] = f"{fisher_exact(tab.to_numpy()).pvalue:.2f}"

    # Confusores
    conf = res_dir / "confounds"
    if (conf / "eeg_predicts_confounds.csv").exists():
        c = pd.read_csv(conf / "eeg_predicts_confounds.csv")
        rows = [r"\begin{tabular}{@{}lrll@{}}", r"\toprule",
                r"Target predicted from EEG features & $n_\text{pos}/n$ & AUC [95\% CI] & $p_\text{perm}$ \\", r"\midrule"]
        for _, r in c.iterrows():
            rows.append(f"{r['target']} & {r['n_pos']}/{r['n']} & {fmt_ci(r['auc'], r['auc_ci_low'], r['auc_ci_high'])}"
                        f" & {fmt_p(r['p_perm'], r['n_perm'])} \\\\")
            k = key(r["target"].split(" ")[0])
            macros[f"Conf{k}Auc"] = f"{r['auc']:.2f}"
            macros[f"Conf{k}AucLo"] = f"{r['auc_ci_low']:.2f}"
            macros[f"Conf{k}AucHi"] = f"{r['auc_ci_high']:.2f}"
            macros[f"Conf{k}P"] = fmt_p(r["p_perm"], r["n_perm"])
        sens = json.loads((conf / "sensitivity_jun_aug_2015.json").read_text())
        rows += [r"\midrule", r"\multicolumn{4}{@{}l}{\textit{MDD vs.\ HC, primary pipeline, restricted subset}} \\",
                 f"Recorded Jun--Aug 2015 & {sens['n_mdd']}/{sens['n']} & "
                 f"{fmt_ci(sens['auc'], sens['auc_ci_low'], sens['auc_ci_high'])} & {fmt_p(sens['p_perm'], sens['n_perm'])} \\\\"]
        macros.update({"SensN": str(sens["n"]), "SensNMDD": str(sens["n_mdd"]), "SensAuc": f"{sens['auc']:.2f}",
                       "SensAucLo": f"{sens['auc_ci_low']:.2f}", "SensAucHi": f"{sens['auc_ci_high']:.2f}",
                       "SensP": fmt_p(sens["p_perm"], sens["n_perm"])})
        a53 = models_dir.parent / "models_all53" / "PRIMARY_LR-L2_all"
        if (a53 / "ci.csv").exists():
            ci = pd.read_csv(a53 / "ci.csv", index_col=0)
            meta = json.loads((a53 / "meta.json").read_text())
            rows.append(f"All 53 participants (no exclusions) & 24/53 & "
                        f"{fmt_ci(ci.loc['auc', 'estimate'], ci.loc['auc', 'ci_low'], ci.loc['auc', 'ci_high'])} & "
                        f"{fmt_p(meta.get('p_value', np.nan), meta.get('n_perm', 0))} \\\\")
            macros.update({"AllAuc": f"{ci.loc['auc', 'estimate']:.2f}", "AllAucLo": f"{ci.loc['auc', 'ci_low']:.2f}",
                           "AllAucHi": f"{ci.loc['auc', 'ci_high']:.2f}",
                           "AllP": fmt_p(meta.get("p_value", np.nan), meta.get("n_perm", 0))})
        rows += [r"\bottomrule", r"\end{tabular}"]
        (tabs / "confounds.tex").write_text("\n".join(rows), encoding="utf-8")
        bs = pd.read_csv(conf / "primary_by_sex.csv")
        for _, r in bs.iterrows():
            macros[f"BySex{key(r['sex'])}Auc"] = f"{r['auc']:.2f}"
            macros[f"BySex{key(r['sex'])}N"] = str(int(r["n_subjects"]))
        q = pd.read_csv(conf / "quality_by_group.csv")
        for _, r in q.iterrows():
            macros[f"Qual{key(r['variable'])}P"] = f"{r['p']:.2f}"

    # Interpretación
    it = res_dir / "interpret"
    if (it / "permutation_importance_families.csv").exists():
        fa = pd.read_csv(it / "permutation_importance_families.csv", index_col=0)
        fig, ax = P.plt.subplots(figsize=(3.4, 1.9))
        fa = fa.sort_values("importance_mean")
        yy = np.arange(len(fa))
        ax.axvline(0, color=P.INK2, lw=0.8)
        ax.plot([fa["ci_low"], fa["ci_high"]], [yy, yy], color=P.SERIES[0], lw=2)
        ax.plot(fa["importance_mean"], yy, "o", color=P.SERIES[0], mec="white", mew=1.2, ms=6)
        ax.set_yticks(yy, [f"{i} ({int(n)})" for i, n in zip(fa.index, fa["n_features"])])
        ax.set_xlabel("Drop in test-fold AUC when the family is permuted")
        ax.grid(axis="y", visible=False)
        P.save(fig, figs / "fig_family_importance")
        for fam, r in fa.iterrows():
            macros[f"Imp{key(fam)}"] = f"{r['importance_mean']:.3f}"
        uni = pd.read_csv(it / "univariate_effects.csv")
        macros["UniNSig"] = str(int((uni["p"] < 0.05).sum()))
        macros["UniNFdr"] = str(int((uni["p_fdr"] < 0.05).sum()))
        macros["UniNFeat"] = str(len(uni))
        macros["UniMinPfdr"] = f"{uni['p_fdr'].min():.2f}"
        top = uni.sort_values("p").head(10)
        t0 = top.iloc[0]
        macros.update({"UniTopFeature": t0["feature"].replace("_", " "), "UniTopFamily": t0["family"],
                       "UniTopD": f"{t0['cohen_d']:.2f}", "UniTopP": f"{t0['p']:.4f}", "UniTopPfdr": f"{t0['p_fdr']:.3f}"})
        rows = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
                r"Feature & Family & Cohen's $d$ & $p$ & $p_\text{FDR}$ \\", r"\midrule"]
        for _, r in top.iterrows():
            pv = "$<$0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
            rows.append(f"\\texttt{{{r['feature'].replace('_', chr(92) + '_')}}} & {r['family']} & {r['cohen_d']:.2f} & "
                        f"{pv} & {r['p_fdr']:.2f} \\\\")
        rows += [r"\bottomrule", r"\end{tabular}"]
        (tabs / "univariate_top.tex").write_text("\n".join(rows), encoding="utf-8")
        st = pd.read_csv(it / "erp_spatiotemporal_clusters.csv")
        macros["ErpClusterMinP"] = f"{st['p'].min():.2f}" if len(st) else "n/a"
        macros["ErpNClusters"] = str(len(st))
        pc = pd.read_csv(it / "power_cluster_tests.csv")
        macros["PowerClusterMinP"] = f"{pc['p'].min():.2f}" if len(pc) else "n/a"

    demo = pd.read_csv(reports / "tables" / "demographics.csv")
    rows = [r"\begin{tabular}{@{}lllll@{}}", r"\toprule", r"Variable & MDD ($n=24$) & HC ($n=29$) & Test & $p$ \\",
            r"\midrule"]
    names = {"age": "Age (years)", "education": "Education (years)", "gender": "Sex (F/M)", "PHQ-9": "PHQ-9$^\\dagger$",
             "GAD-7": "GAD-7$^\\dagger$", "PSQI": "PSQI$^\\dagger$"}
    for _, r in demo.iterrows():
        if r["variable"] not in names:
            continue
        if r["variable"] == "gender":
            mdd = r["MDD_counts"].replace("F=", "").replace(", M=", "/")
            hc = r["HC_counts"].replace("F=", "").replace(", M=", "/")
        else:
            mdd, hc = r["MDD_mean_sd"].replace("±", r"$\pm$"), r["HC_mean_sd"].replace("±", r"$\pm$")
        p = r["p_value"]
        rows.append(f"{names[r['variable']]} & {mdd} & {hc} & {r['test'].replace('Mann-Whitney U', 'Mann--Whitney')} & "
                    f"{p:.2g} \\\\" if p >= 0.001 else
                    f"{names[r['variable']]} & {mdd} & {hc} & {r['test'].replace('Mann-Whitney U', 'Mann--Whitney')} & "
                    f"$<$0.001 \\\\")
        macros[f"Demo{key(r['variable'])}P"] = f"{p:.3f}"
    rows += [r"\bottomrule", r"\end{tabular}"]
    (tabs / "demographics.tex").write_text("\n".join(rows), encoding="utf-8")

    # Comparaciones entre modelos en los MISMOS pliegues: t corregido de Nadeau-Bengio sobre ΔAUC por pliegue,
    # y ajuste de Holm de los p de permutación de los modelos secundarios (multiplicidad).
    comp_rows = _comparisons(models_dir, df, macros, key)
    if comp_rows:
        (tabs / "comparisons.tex").write_text("\n".join(comp_rows), encoding="utf-8")

    paper = Path(cfg["paths"]["reports_dir"]).parent / "docs" / "paper"
    gen = paper / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    import shutil

    for src, dst in ((figs, paper / "figures"), (tabs, paper / "tables")):
        dst.mkdir(exist_ok=True)
        for f in src.iterdir():
            if f.suffix in (".pdf", ".tex"):
                shutil.copy2(f, dst / f.name)
    (gen / "numbers.tex").write_text(
        "% Generado por scripts/11_figures_tables.py: no editar a mano.\n" +
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())), encoding="utf-8")
    print(f"{len(macros)} macros -> {gen / 'numbers.tex'}")


if __name__ == "__main__":
    main()
