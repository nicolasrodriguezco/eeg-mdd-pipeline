"""Paso 8: deep learning (braindecode) sobre las épocas de 129 canales, CV por sujeto.

Los pliegues externos de sujetos son los MISMOS que en el paso 7 (misma función y
semillas). Por coste de cómputo se usan menos repeticiones que en los modelos
rápidos; el número exacto queda en meta.json y en el artículo.
"""

import argparse
import json
import time
from pathlib import Path

import _common  # noqa: F401
import mne
import numpy as np
import pandas as pd
import torch

from eeg_mdd.config import load_config
from eeg_mdd.evaluation import CVResult, aggregate_to_subject, subject_metrics
from eeg_mdd.evaluation.cv import outer_splits
from eeg_mdd.models.deep import DLConfig, train_fold


def load_trials(processed: Path, subjects: pd.DataFrame):
    X, y, g = [], [], []
    for _, s in subjects.iterrows():
        ep = mne.read_epochs(processed / "epochs" / f"sub-{s['eeg_id']}-epo.fif", verbose="error")
        d = (ep.get_data(picks="eeg") * 1e6).astype(np.float32)          # µV
        X.append(d)
        y += [int(s["group"] == "MDD")] * len(d)
        g += [s["eeg_id"]] * len(d)
    return np.concatenate(X), np.array(y), np.array(g), float(ep.info["sfreq"])


def run_model(name, X, y, g, sfreq, n_repeats, cfg, device, perm_seed=None):
    yy = y.copy()
    if perm_seed is not None:                       # etiquetas permutadas ENTRE sujetos
        subj = pd.DataFrame({"g": g, "y": y}).drop_duplicates("g")
        rng = np.random.default_rng(10_000 + perm_seed)
        mp = dict(zip(subj["g"], rng.permutation(subj["y"].to_numpy())))
        yy = np.array([mp[s] for s in g])
    oof, folds, info = [], [], []
    for r, k, tr, te in outer_splits(yy, g, 5, n_repeats):
        cfg_k = DLConfig(**{**cfg.__dict__, "seed": 1000 * r + k})
        out = train_fold(name, X, yy, g, tr, te, sfreq, cfg_k, device)
        subj = aggregate_to_subject(out["prob"], yy[te], g[te]).assign(repeat=r, fold=k)
        oof.append(subj)
        folds.append({"repeat": r, "fold": k, **subject_metrics(subj["y"], subj["prob"])})
        info.append({"repeat": r, "fold": k, "epochs": out["epochs"], "best_val_auc": out["best_val_auc"],
                     "seconds": round(out["seconds"], 1)})
        if perm_seed is None:
            print(f"  {name} rep{r} fold{k}: test AUC={folds[-1]['auc']:.3f} val AUC={out['best_val_auc']:.3f} "
                  f"epochs={out['epochs']} {out['seconds']:.0f}s", flush=True)
    oof = pd.concat(oof, ignore_index=True)
    rep = pd.DataFrame([{"repeat": r, **subject_metrics(d["y"], d["prob"])} for r, d in oof.groupby("repeat")])
    return CVResult(name, oof, pd.DataFrame(folds), rep, info)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*", default=["EEGNet", "ShallowFBCSPNet", "TCN"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=0, help="permutaciones (1 repetición c/u) para cada modelo")
    args = ap.parse_args()

    cfg = load_config(args.config)
    processed = Path(cfg["paths"]["processed_dir"])
    out = Path(cfg["paths"]["reports_dir"]) / "results" / "models"
    subjects = (pd.read_csv(Path(cfg["paths"]["interim_dir"]) / "subjects_final.csv", dtype={"eeg_id": str})
                .sort_values("eeg_id"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    X, y, g, sfreq = load_trials(processed, subjects)
    print(f"Ensayos {X.shape}, sujetos {len(set(g))}, dispositivo {device} "
          f"({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})", flush=True)
    dl_cfg = DLConfig()

    for name in args.models:
        t0 = time.time()
        res = run_model(name, X, y, g, sfreq, args.repeats, dl_cfg, device)
        d = out / f"DL-{name}"
        d.mkdir(parents=True, exist_ok=True)
        res.oof.to_csv(d / "oof.csv", index=False)
        res.fold_scores.to_csv(d / "fold_scores.csv", index=False)
        res.repeat_scores.to_csv(d / "repeat_scores.csv", index=False)
        pd.DataFrame(res.best_params).to_csv(d / "training_log.csv", index=False)
        ci = res.ci()
        ci.to_csv(d / "ci.csv")
        meta = {"seconds": round(time.time() - t0, 1), "n_repeats": args.repeats, "config": dl_cfg.__dict__,
                "device": device, "n_rows": int(len(y)), "n_subjects": int(len(set(g)))}
        if args.n_perm:
            obs = float(res.repeat_scores["auc"].iloc[0])
            null = []
            for i in range(args.n_perm):
                pr = run_model(name, X, y, g, sfreq, 1, dl_cfg, device, perm_seed=i)
                null.append(float(pr.repeat_scores["auc"].iloc[0]))
                print(f"  perm {i + 1}/{args.n_perm}: AUC={null[-1]:.3f}", flush=True)
            null = np.array(null)
            np.save(d / "perm_null.npy", null)
            meta.update({"observed_auc": obs, "n_perm": args.n_perm,
                         "p_value": float((1 + (null >= obs).sum()) / (1 + len(null))),
                         "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95))})
        (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        s = ci["estimate"]
        print(f"DL-{name}: AUC={s['auc']:.3f} [{ci.loc['auc', 'ci_low']:.3f}, {ci.loc['auc', 'ci_high']:.3f}] "
              f"BalAcc={s['balanced_accuracy']:.3f}  ({meta['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
