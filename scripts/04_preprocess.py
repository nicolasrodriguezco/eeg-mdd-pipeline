"""Paso 4: preprocesamiento por sujeto (en paralelo) y aplicación de los criterios de exclusión.

Salidas:
  data/processed/epochs/sub-<id>-epo.fif        épocas -200..600 ms alineadas al par de rostros
  data/processed/continuous/sub-<id>-seg-epo.fif segmentos limpios de 2 s a 125 Hz
  data/processed/ica/sub-<id>-ica.fif
  configs/ica_exclusions.csv                     componentes quitados (ICLabel-auto)
  reports/preprocessing_log.csv                  qué pasó con cada sujeto
  reports/ica_labels.csv                         etiqueta ICLabel de cada componente
  reports/qc/sub-<id>.html                       mne.Report por sujeto
  data/interim/subjects_final.csv                sujetos que cumplen los criterios
"""

import argparse
import traceback
from pathlib import Path

import _common  # noqa: F401
import pandas as pd
from joblib import Parallel, delayed

from eeg_mdd.checks import check_labels, raise_if_problems
from eeg_mdd.config import ensure_dirs, load_config
from eeg_mdd.io import load_raw
from eeg_mdd.preprocess import apply_exclusion_criteria, load_ica_exclusions, preprocess_subject


def _make_report(res, sid: str, group: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import mne

    rep = mne.Report(title=f"sub-{sid} ({group})", verbose="error")
    if res.ica is not None:
        rep.add_ica(res.ica, title="ICA (excluded components marked)", inst=None, n_jobs=1)
        if res.ica_labels is not None:
            rep.add_html(res.ica_labels.round(3).to_html(index=False), title="ICLabel")
    rep.add_html(pd.Series(res.bad_reasons, dtype=str).to_frame("reason").to_html(), title="Bad channels")
    rep.add_epochs(res.epochs, title="Epochs (face-locked)", psd=False)
    evokeds = [res.epochs[c].average() for c in res.epochs.event_id if len(res.epochs[c])]
    rep.add_evokeds(evokeds, n_time_points=5)
    rep.save(out, overwrite=True, open_browser=False, verbose="error")


def run_one(sid: str, group: str, file: str, pp: dict, manual_excl, dirs: dict, report: bool) -> dict:
    try:
        raw = load_raw(file)
        res = preprocess_subject(raw, pp, sid, manual_excl, ica_dir=dirs["ica"])
        res.epochs.save(dirs["epochs"] / f"sub-{sid}-epo.fif", overwrite=True, verbose="error")
        if res.continuous is not None:
            res.continuous.save(dirs["cont"] / f"sub-{sid}-seg-epo.fif", overwrite=True, verbose="error")
        if res.ica_labels is not None:
            res.ica_labels.assign(subject_id=sid).to_csv(dirs["labels"] / f"sub-{sid}.csv", index=False)
        if report:
            _make_report(res, sid, group, dirs["qc"] / f"sub-{sid}.html")
        log = res.log
        log["error"] = ""
    except Exception as exc:  # noqa: BLE001 - se registra y se sigue con los demás
        log = {"subject_id": sid, "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc(limit=3)}
    log["group"] = group
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--subject", default=None, help="Procesar solo este sujeto")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    paths, pp = cfg["paths"], cfg["preprocessing"]
    interim, processed, reports = Path(paths["interim_dir"]), Path(paths["processed_dir"]), Path(paths["reports_dir"])
    dirs = {"epochs": processed / "epochs", "cont": processed / "continuous", "ica": processed / "ica",
            "labels": interim / "ica_labels", "qc": reports / "qc"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    subjects = pd.read_csv(interim / "subjects.csv", dtype={"eeg_id": str})
    raise_if_problems(check_labels(subjects), "etiquetas")
    audit = pd.read_csv(interim / "audit.csv", dtype={"subject_id": str})
    file_by_id = dict(zip(audit["subject_id"], audit["file"]))
    manual = load_ica_exclusions(paths["ica_exclusions"])
    if args.subject:
        subjects = subjects[subjects["eeg_id"] == args.subject]

    logs = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_one)(s["eeg_id"], s["group"], file_by_id[s["eeg_id"]], pp, manual.get(s["eeg_id"]), dirs,
                         not args.no_report)
        for _, s in subjects.iterrows()
    )
    log = pd.DataFrame(logs)
    conditions = list(pp["event_id"])
    n_ch = int(cfg["dataset"].get("expected_n_eeg_channels") or 129)
    crit = cfg.get("exclusion") or {"max_bad_channel_frac": 1.0, "min_trials_total": 0, "min_trials_per_condition": 0}
    log = apply_exclusion_criteria(log, crit, n_ch, conditions)
    out = reports / ("preprocessing_log.csv" if not args.subject else f"preprocessing_log_{args.subject}.csv")
    log.to_csv(out, index=False)

    if not args.subject:
        labels = [pd.read_csv(p, dtype={"subject_id": str}) for p in sorted(dirs["labels"].glob("sub-*.csv"))]
        if labels:
            lab = pd.concat(labels)
            lab.to_csv(reports / "ica_labels.csv", index=False)
            excl = (lab[lab["exclude"]].groupby("subject_id")["component"]
                    .agg(lambda s: ",".join(str(int(c)) for c in s)))
            rows = pd.DataFrame({"subject_id": log["subject_id"]})
            rows["exclude_components"] = rows["subject_id"].map(excl).fillna("")
            rows["reviewer"] = f"ICLabel-auto (p>={pp['ica']['iclabel_threshold']})"
            rows["notes"] = "automatic, rule fixed in docs/preregistration.md"
            rows.to_csv(paths["ica_exclusions"], index=False)
        final = subjects.merge(log[["subject_id", "included", "exclusion_reason"]],
                               left_on="eeg_id", right_on="subject_id", how="left")
        final[final["included"]].drop(columns=["subject_id"]).to_csv(interim / "subjects_final.csv", index=False)
        print(f"\nIncluidos: {int(final['included'].sum())}/{len(final)}")
        print(final.loc[~final["included"], ["eeg_id", "group", "exclusion_reason"]].to_string(index=False))
    print(f"Registro: {out}")
    print(log.drop(columns=[c for c in log.columns if c in ("traceback",)]).describe().T.round(1).to_string())


if __name__ == "__main__":
    main()
