"""Paso 6: características por sujeto (biomarcadores) y datos por ensayo para Riemann/DL.

Las características no usan etiquetas, así que pueden calcularse una vez por sujeto;
todo lo que aprende de los datos (escalado, selección, modelos) va dentro de la CV.

Salidas (data/processed/features/):
  subject_features.parquet   una fila por sujeto: subject_id, group + ~390 características
  channel_power.parquet      potencia absoluta por canal y banda (para topomapas)
  trial_regions.npz          señal regional por ensayo (n_trials, 10, 201) + subject_id + condición
  evokeds/sub-<id>-ave.fif   ERP promedio por condición
"""

import argparse
from pathlib import Path

import _common  # noqa: F401
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from eeg_mdd.config import load_config
from eeg_mdd.features import subject_features
from eeg_mdd.features.spectral import BANDS


def one(sid: str, group: str, processed: Path, out: Path) -> dict:
    ep = mne.read_epochs(processed / "epochs" / f"sub-{sid}-epo.fif", verbose="error")
    seg = mne.read_epochs(processed / "continuous" / f"sub-{sid}-seg-epo.fif", verbose="error")
    res = subject_features(ep, seg)
    mne.write_evokeds(out / "evokeds" / f"sub-{sid}-ave.fif", res["evokeds"], overwrite=True, verbose="error")
    row = {"subject_id": sid, "group": group, **res["features"]}
    power = pd.DataFrame(res["channel_power"], index=list(BANDS), columns=res["channel_names"])
    power = power.rename_axis("band").reset_index().assign(subject_id=sid, group=group)
    return {"row": row, "power": power, "trials": res["trial_regions"], "cond": res["trial_conditions"],
            "regions": res["region_names"], "times": ep.times}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    processed = Path(cfg["paths"]["processed_dir"])
    out = processed / "features"
    (out / "evokeds").mkdir(parents=True, exist_ok=True)
    # Se calculan para TODOS los sujetos preprocesados (no usan etiquetas); la columna
    # ``included`` marca a los que cumplen los criterios de exclusión preregistrados.
    interim = Path(cfg["paths"]["interim_dir"])
    subjects = pd.read_csv(interim / "subjects.csv", dtype={"eeg_id": str})
    final = set(pd.read_csv(interim / "subjects_final.csv", dtype={"eeg_id": str})["eeg_id"])

    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(one)(s["eeg_id"], s["group"], processed, out) for _, s in subjects.iterrows())

    feats = pd.DataFrame([r["row"] for r in res])
    feats.insert(2, "included", feats["subject_id"].isin(final))
    feats.to_parquet(out / "subject_features.parquet", index=False)
    pd.concat([r["power"] for r in res]).to_parquet(out / "channel_power.parquet", index=False)
    keep = [r["row"]["subject_id"] in final for r in res]
    res = [r for r, k in zip(res, keep) if k]          # Riemann/DL: solo sujetos incluidos
    np.savez_compressed(
        out / "trial_regions.npz",
        X=np.concatenate([r["trials"] for r in res]),
        subject_id=np.concatenate([[r["row"]["subject_id"]] * len(r["trials"]) for r in res]),
        group=np.concatenate([[r["row"]["group"]] * len(r["trials"]) for r in res]),
        condition=np.concatenate([r["cond"] for r in res]),
        regions=np.array(res[0]["regions"]), times=res[0]["times"],
    )
    n_feat = feats.shape[1] - 3
    print(f"Sujetos: {len(feats)}  |  características: {n_feat}  |  NaN: {int(feats.isna().sum().sum())}")
    print(f"Ensayos totales (Riemann/DL): {sum(len(r['trials']) for r in res)}")


if __name__ == "__main__":
    main()
