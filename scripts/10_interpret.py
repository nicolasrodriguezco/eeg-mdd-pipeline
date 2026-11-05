"""Paso 10: interpretabilidad y análisis neurofisiológico descriptivo.

1. Importancia por permutación DENTRO de la CV (pliegues de prueba del modelo primario),
   por característica y por familia (se permuta la familia completa a la vez).
2. SHAP del modelo primario ajustado a todos los sujetos (descriptivo).
3. Efectos univariados por característica (d de Cohen, Mann-Whitney, FDR de Benjamini-Hochberg).
4. Topomapas de d de Cohen de la potencia por canal y banda, con test de permutación por clústeres espaciales.
5. ERP promedio por grupo y condición y test de permutación por clústeres espacio-temporales.
"""

import argparse
import json
from pathlib import Path

import _common  # noqa: F401
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from statsmodels.stats.multitest import multipletests

from eeg_mdd import plotting as P
from eeg_mdd.config import load_config
from eeg_mdd.resources import free_gb, limit_threads, safe_n_jobs
from eeg_mdd.evaluation.cv import outer_splits
from eeg_mdd.features import erp_clusters, family_of
from eeg_mdd.features.spectral import BANDS
from eeg_mdd.models import tabular_models


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


def _fold_importance(pipe, grid, X, y, g, tr, te, fams, n_shuffle, seed):
    inner = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(tr)), y[tr], g[tr]))
    gs = GridSearchCV(clone(pipe), grid, scoring="roc_auc", cv=inner, n_jobs=1).fit(X.iloc[tr], y[tr])
    m = gs.best_estimator_
    Xte = X.iloc[te].copy()
    base = roc_auc_score(y[te], m.predict_proba(Xte)[:, 1])
    rng = np.random.default_rng(seed)
    feat_imp, fam_imp = {}, {}
    for c in X.columns:
        drops = []
        for _ in range(n_shuffle):
            Xp = Xte.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            drops.append(base - roc_auc_score(y[te], m.predict_proba(Xp)[:, 1]))
        feat_imp[c] = float(np.mean(drops))
    for fam, cols in fams.items():
        drops = []
        for _ in range(n_shuffle):
            Xp = Xte.copy()
            perm = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].to_numpy()[perm]          # permuta la familia entera junta
            drops.append(base - roc_auc_score(y[te], m.predict_proba(Xp)[:, 1]))
        fam_imp[fam] = float(np.mean(drops))
    coef = pd.Series(m.named_steps["clf"].coef_[0], index=X.columns)
    return feat_imp, fam_imp, coef


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args()
    limit_threads()
    args.n_jobs = safe_n_jobs(0.5, max_jobs=args.n_jobs)
    print(f"Procesos: {args.n_jobs} (RAM libre {free_gb():.1f} GB)", flush=True)
    P.setup()

    cfg = load_config(args.config)
    feat_dir = Path(cfg["paths"]["processed_dir"]) / "features"
    out = Path(cfg["paths"]["reports_dir"]) / "results" / "interpret"
    figs = Path(cfg["paths"]["reports_dir"]) / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(feat_dir / "subject_features.parquet").sort_values("subject_id").reset_index(drop=True)
    feats = feats[feats["included"]].reset_index(drop=True)
    cols = [c for c in feats.columns if family_of(c)]
    X, y, g = feats[cols], (feats["group"] == "MDD").astype(int).to_numpy(), feats["subject_id"].to_numpy()
    fams = {f: [c for c in cols if family_of(c) == f] for f in ("spectral", "erp", "connectivity", "complexity")}

    # 1. Importancia por permutación dentro de la CV
    pipe, grid = tabular_models()["LR-L2"]
    jobs = list(outer_splits(y, g, 5, 10))
    res = Parallel(n_jobs=args.n_jobs)(delayed(_fold_importance)(pipe, grid, X, y, g, tr, te, fams, 5, 1000 * r + k)
                                       for r, k, tr, te in jobs)
    fi = pd.DataFrame([r[0] for r in res])
    fa = pd.DataFrame([r[1] for r in res])
    coefs = pd.DataFrame([r[2] for r in res])
    imp = pd.DataFrame({"importance_mean": fi.mean(), "importance_sd": fi.std(),
                        "frac_folds_positive": (fi > 0).mean(), "coef_mean": coefs.mean(),
                        "coef_sign_consistency": (np.sign(coefs) == np.sign(coefs.mean())).mean()})
    imp["family"] = [family_of(c) for c in imp.index]
    imp.sort_values("importance_mean", ascending=False).to_csv(out / "permutation_importance_features.csv")
    fam_tab = pd.DataFrame({"importance_mean": fa.mean(), "importance_sd": fa.std(),
                            "ci_low": fa.quantile(0.025), "ci_high": fa.quantile(0.975),
                            "frac_folds_positive": (fa > 0).mean(), "n_features": {f: len(c) for f, c in fams.items()}})
    fam_tab.to_csv(out / "permutation_importance_families.csv")
    print("Importancia por familia (caída de AUC en pliegues de prueba):")
    print(fam_tab.round(4).to_string())

    # 2. SHAP (descriptivo, modelo ajustado con todos los sujetos)
    import shap

    gs = GridSearchCV(clone(pipe), grid, scoring="roc_auc",
                      cv=StratifiedGroupKFold(5, shuffle=True, random_state=0), n_jobs=1).fit(X, y, groups=g)
    m = gs.best_estimator_
    Xs = m[:-1].transform(X)
    expl = shap.LinearExplainer(m.named_steps["clf"], Xs)
    sv = expl.shap_values(Xs)
    shap_tab = pd.DataFrame({"mean_abs_shap": np.abs(sv).mean(axis=0)}, index=cols).sort_values("mean_abs_shap",
                                                                                               ascending=False)
    shap_tab["family"] = [family_of(c) for c in shap_tab.index]
    shap_tab.to_csv(out / "shap_primary.csv")
    (out / "shap_meta.json").write_text(json.dumps({"best_C": float(gs.best_params_["clf__C"])}))
    top = shap_tab.index[:15][::-1]
    fig, ax = P.plt.subplots(figsize=(5.2, 3.8))
    idx = [cols.index(c) for c in top]
    for row, j in enumerate(idx):
        vals = sv[:, j]
        colors = [P.GROUP_COLORS["MDD"] if yy else P.GROUP_COLORS["HC"] for yy in y]
        jitter = np.random.default_rng(row).uniform(-0.18, 0.18, len(vals))
        ax.scatter(vals, row + jitter, s=9, c=colors, alpha=0.8, linewidths=0)
    ax.set_yticks(range(len(top)), [t.replace("__", " – ").replace("_", " ") for t in top], fontsize=6.5)
    ax.axvline(0, color=P.INK2, lw=0.8)
    ax.set_xlabel("SHAP value (log-odds of MDD)")
    ax.set_title("Primary model: top-15 features by mean |SHAP|")
    ax.scatter([], [], c=P.GROUP_COLORS["MDD"], s=12, label="MDD")
    ax.scatter([], [], c=P.GROUP_COLORS["HC"], s=12, label="HC")
    ax.legend(loc="lower right", fontsize=7)
    P.save(fig, figs / "fig_shap")

    # 3. Efectos univariados
    rows = []
    for c in cols:
        a, b = feats.loc[y == 1, c].dropna(), feats.loc[y == 0, c].dropna()
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"feature": c, "family": family_of(c), "MDD_mean": a.mean(), "HC_mean": b.mean(),
                     "cohen_d": cohen_d(a, b), "p": u.pvalue})
    uni = pd.DataFrame(rows)
    uni["p_fdr"] = multipletests(uni["p"], method="fdr_bh")[1]
    uni.sort_values("p").to_csv(out / "univariate_effects.csv", index=False)
    print(f"Efectos univariados: {int((uni['p'] < 0.05).sum())} con p<0.05 sin corregir, "
          f"{int((uni['p_fdr'] < 0.05).sum())} con FDR<0.05 (de {len(uni)})")

    # 4. Topomapas de potencia por canal (d de Cohen sobre log10 potencia) + clústeres espaciales
    power = pd.read_parquet(feat_dir / "channel_power.parquet")
    power = power[power["subject_id"].isin(g)]
    chans = [c for c in power.columns if c not in ("band", "subject_id", "group")]
    mont = mne.channels.make_standard_montage("GSN-HydroCel-129")
    info = mne.create_info(chans, 125.0, "eeg")
    info.set_montage(mont)
    adjacency, _ = mne.channels.find_ch_adjacency(info, "eeg")
    fig, axes = P.plt.subplots(1, len(BANDS), figsize=(7.2, 2.0), gridspec_kw=dict(wspace=0.08))
    clus_rows, dmaps = [], {}
    for ax, band in zip(axes, BANDS):
        pb = power[power["band"] == band].set_index("subject_id").loc[g, chans]
        lp = np.log10(pb.to_numpy())
        d = np.array([cohen_d(lp[y == 1, i], lp[y == 0, i]) for i in range(len(chans))])
        dmaps[band] = d
        t_obs, clusters, pvals, _ = mne.stats.permutation_cluster_test(
            [lp[y == 1], lp[y == 0]], adjacency=adjacency, n_permutations=args.n_perm, tail=0, seed=0,
            out_type="mask", verbose="error")
        mask = np.zeros(len(chans), bool)
        for cl, p in zip(clusters, pvals):
            clus_rows.append({"band": band, "n_channels": int(cl.sum()), "p": float(p)})
            if p < 0.05:
                mask |= cl
        im, _ = mne.viz.plot_topomap(d, info, axes=ax, show=False, cmap=P.DIVERGING, vlim=(-1, 1), mask=mask,
                                     mask_params=dict(marker="o", markerfacecolor="white", markeredgecolor=P.INK,
                                                      markersize=3), contours=0, sensors=False)
        band_name = band.replace("lowgamma", "low gamma")
        ax.set_title(f"{band_name}\n{BANDS[band][0]:g}–{BANDS[band][1]:g} Hz", fontsize=7)
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.03, fraction=0.025)
    cb.set_label("Cohen's d (MDD − HC)\nlog10 power", fontsize=7)
    P.save(fig, figs / "fig_topomaps_power")
    pd.DataFrame(clus_rows).to_csv(out / "power_cluster_tests.csv", index=False)
    pd.DataFrame(dmaps, index=chans).to_csv(out / "power_cohen_d_by_channel.csv")

    # 5. ERP por grupo y condición + test espacio-temporal
    evk = {}
    for s in g:
        evs = mne.read_evokeds(feat_dir / "evokeds" / f"sub-{s}-ave.fif", verbose="error")
        evk[s] = {e.comment: e for e in evs}
    conds = list(next(iter(evk.values())))
    times = next(iter(evk.values()))[conds[0]].times
    cl = erp_clusters()
    ch_names = next(iter(evk.values()))[conds[0]].ch_names
    idx = {c: i for i, c in enumerate(ch_names)}
    fig, axes = P.plt.subplots(3, len(conds), figsize=(7.2, 5.2), sharex=True)
    titles = {"n170_L": "Left occ.-temp. (≈P7)", "n170_R": "Right occ.-temp. (≈P8)",
              "centroparietal": "Centro-parietal (≈Pz)"}
    windows = {"n170_L": (0.14, 0.2), "n170_R": (0.14, 0.2), "centroparietal": (0.25, 0.5)}
    for j, cond in enumerate(conds):
        for i, (cname, chs) in enumerate(cl.items()):
            ax = axes[i, j]
            pick = [idx[c] for c in chs]
            ax.axvspan(*windows[cname], color=P.GRID, alpha=0.6, lw=0)
            for grp, lab in ((0, "HC"), (1, "MDD")):
                w = np.array([evk[s][cond].data[pick].mean(0) * 1e6 for s, yy in zip(g, y) if yy == grp])
                mu, se = w.mean(0), w.std(0, ddof=1) / np.sqrt(len(w))
                ax.plot(times * 1000, mu, color=P.GROUP_COLORS[lab], lw=1.6, label=f"{lab} (n={len(w)})")
                ax.fill_between(times * 1000, mu - se, mu + se, color=P.GROUP_COLORS[lab], alpha=0.2, lw=0)
            ax.axvline(0, color=P.INK2, lw=0.6)
            ax.axhline(0, color=P.INK2, lw=0.6)
            if i == 0:
                ax.set_title(cond.capitalize() + " faces")
            if j == 0:
                ax.set_ylabel(f"{titles[cname]}\nµV", fontsize=7)
            if i == 2:
                ax.set_xlabel("Time from faces onset (ms)", fontsize=7)
    axes[0, 0].legend(fontsize=6.5, loc="lower left")
    P.save(fig, figs / "fig_erp_groups")

    # Test espacio-temporal (promedio de las tres condiciones por sujeto), descriptivo
    data = np.array([np.mean([evk[s][c].data for c in conds], axis=0).T * 1e6 for s in g])   # (n, times, ch)
    info_e = evk[g[0]][conds[0]].info
    adj_e, _ = mne.channels.find_ch_adjacency(info_e, "eeg")
    F, clusters, pvals, _ = mne.stats.spatio_temporal_cluster_test(
        [data[y == 1], data[y == 0]], adjacency=adj_e, n_permutations=args.n_perm, seed=0, tail=1,
        out_type="mask", verbose="error")
    st_rows = []
    for c, p in zip(clusters, pvals):
        tmask = c.any(axis=1)
        st_rows.append({"p": float(p), "t_start_ms": float(times[tmask].min() * 1000),
                        "t_end_ms": float(times[tmask].max() * 1000), "n_channels": int(c.any(axis=0).sum()),
                        "mass_F": float(F[c].sum())})
    st = pd.DataFrame(st_rows).sort_values("p") if st_rows else pd.DataFrame(columns=["p"])
    st.to_csv(out / "erp_spatiotemporal_clusters.csv", index=False)
    print("Clústeres ERP espacio-temporales (MDD vs HC, condiciones promediadas):")
    print(st.head(5).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
