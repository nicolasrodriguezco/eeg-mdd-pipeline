"""Paso 17: recursos del Jupyter Book (``book/``).

Copia al libro **solo resultados agregados**: figuras del artículo y tablas resumen por modelo. Ningún archivo con
filas por participante (oof, features, demografía individual) sale de ``reports/``. El libro no ejecuta código:
se construye sin datos, así que puede publicarse (por ejemplo en GitHub Pages).

Salidas en ``book/_generated/``:
- ``figures/*.png``: figuras del artículo;
- ``main_results.md``, ``audit_models.md`` e ``inflation.md``: tablas agregadas;
- ``deviations.md``: copia del registro de desviaciones.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "results"
OUT = ROOT / "book" / "_generated"

FIGURES = ["fig_model_comparison", "fig_roc", "fig_permutation", "fig_inflation", "fig_quality", "fig_regions",
           "fig_topomaps_power", "fig_erp_groups", "fig_family_importance", "fig_shap", "fig_power_curve", "fig_emg"]
ID_RE = re.compile(r"\b020[123]\d{4}\b")  # IDs de MODMA: no deben aparecer en tablas del libro


def ci(row, k, nd=2):
    return f"{row[k]:.{nd}f} [{row[k + '_lo']:.{nd}f}, {row[k + '_hi']:.{nd}f}]"


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False))
    text = head + body + "\n"
    assert not ID_RE.search(text), "una tabla del libro contiene un ID de participante"
    return text


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    missing = []
    for f in FIGURES:
        src = RES / "figures" / f"{f}.png"
        if src.exists():
            shutil.copy2(src, OUT / "figures" / src.name)
        else:
            missing.append(f)

    m = pd.read_csv(RES / "tables" / "main_results.csv")
    m = m.sort_values(["cat_rank", "auc"], ascending=[True, False])
    t = pd.DataFrame({
        "Model": m["model"], "Family": m["category"],
        "AUC [95% CI]": [ci(r, "auc") for _, r in m.iterrows()],
        "Balanced acc. [95% CI]": [ci(r, "balanced_accuracy") for _, r in m.iterrows()],
        "p (perm.)": [fp(p) if pd.notna(p) else "not run" for p in m["p_perm"]],
        "Repeats": m["n_repeats"].astype(int),
    })
    (OUT / "main_results.md").write_text(md_table(t), encoding="utf-8")

    a = pd.read_csv(RES / "audit" / "exploratory_models.csv")
    t = pd.DataFrame({
        "Model (exploratory)": a["model"],
        "AUC [95% CI]": [ci(r, "auc") for _, r in a.iterrows()],
        "p (perm.)": a["p_perm"].map(fp), "p (Holm)": a["p_holm"].map(fp),
        "ΔAUC vs demographics (p)": [f"{r.dAUC_vs_demo:+.2f} ({fp(r.p_vs_demo)})" for r in a.itertuples()],
    })
    (OUT / "audit_models.md").write_text(md_table(t), encoding="utf-8")

    inf = pd.read_csv(RES / "tables" / "inflation_summary.csv", skiprows=3, header=None,
                      names=["model", "split", "trial_auc_mean", "trial_auc_std", "subject_auc_mean",
                             "subject_auc_std"])
    t = pd.DataFrame({
        "Model": inf.iloc[:, 0], "Split": inf.iloc[:, 1],
        "Trial-level AUC": [f"{x:.2f} ± {s:.2f}" for x, s in zip(inf["trial_auc_mean"], inf["trial_auc_std"])],
        "Participant-level AUC": [f"{x:.2f} ± {s:.2f}"
                                  for x, s in zip(inf["subject_auc_mean"], inf["subject_auc_std"])],
    })
    (OUT / "inflation.md").write_text(md_table(t), encoding="utf-8")

    dev = (ROOT / "reports" / "deviations.md").read_text(encoding="utf-8")
    assert not ID_RE.search(dev), "deviations.md contiene un ID de participante"
    (OUT / "deviations.md").write_text(dev, encoding="utf-8")
    print(f"libro: {len(FIGURES) - len(missing)} figuras, 3 tablas, desviaciones"
          + (f"; faltan {missing}" if missing else ""))


if __name__ == "__main__":
    main()
