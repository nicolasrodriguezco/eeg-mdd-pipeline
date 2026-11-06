"""Paso 12 (D6, exploratorio): características de la auditoría para los 53 sujetos.

Salida: ``data/processed/features/audit_features.parquet`` con una fila por sujeto.

- Conducta: tiempos de reacción ``*dot`` → ``resp`` desde el ``.raw``.
- Músculo: potencia de 55–95 Hz en el registro crudo (pasa-altos 1 Hz, notch 50/100 Hz, referencia promedio,
  periodo de la tarea, sin canales malos).
- Aperiódico 1/f (2–35 Hz) por región y global, frecuencia del pico alfa posterior y theta frontal de línea media.
  Todo se calcula sobre los segmentos limpios.
- Variabilidad entre ensayos de la P300 (clúster centro-parietal).
- Contrastes ERP entre condiciones y fuerza de conectividad wPLI, derivados de ``subject_features.parquet``.
- ``compact_cols``: conjunto teórico reducido, definido antes de evaluarlo.
"""

import argparse
import json
from pathlib import Path

import _common  # noqa: F401
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import welch

from eeg_mdd.config import load_config
from eeg_mdd.events import event_onsets
from eeg_mdd.features import erp_clusters
from eeg_mdd.features.audit import (aperiodic_fit, aperiodic_residual, alpha_peak, behavioural_rt,
                                    muscle_index)
from eeg_mdd.features.regions import EQUIV_1020, neighbors_of, region_map
from eeg_mdd.io import load_raw
from eeg_mdd.preprocess import prepare_raw

REGIONS = region_map()
POSTERIOR = sorted(set(REGIONS["parietal_L"] + REGIONS["parietal_R"] + REGIONS["occipital_L"]
                       + REGIONS["occipital_R"]))
FM_THETA = sorted((set(neighbors_of(EQUIV_1020["Fz"], 4)) | {"E6"}) - {"Cz"})     # Fz/FCz y vecinos
CONDS = {"fear": "fdot", "happy": "hdot", "sad": "sdot"}


def _region_mean(values: np.ndarray, ch_names: list[str], chans: list[str]) -> float:
    idx = [ch_names.index(c) for c in chans if c in ch_names]
    return float(np.mean(values[idx])) if idx else float("nan")


def one_subject(sid: str, raw_file: str, bads: list[str], processed: Path, pp: dict) -> dict:
    row: dict = {"subject_id": sid}
    # --- crudo: conducta y músculo
    raw = prepare_raw(load_raw(raw_file, preload=True), pp)
    sf = raw.info["sfreq"]
    dots = {c: event_onsets(raw, ch) / sf for c, ch in CONDS.items() if ch in raw.ch_names}
    resp = event_onsets(raw, "resp") / sf if "resp" in raw.ch_names else np.array([])
    row.update(behavioural_rt(dots, resp))
    starts = [event_onsets(raw, n) for n in ("fcue", "hcue", "scue") if n in raw.ch_names]
    t0 = max(0.0, min(s.min() for s in starts) / sf - 1.0)
    t1 = min(raw.times[-1], max(s.max() for s in starts) / sf + 2.0)
    eeg = raw.copy().pick("eeg").crop(t0, t1)
    eeg.info["bads"] = [b for b in bads if b in eeg.ch_names]
    eeg.drop_channels(eeg.info["bads"])
    eeg.filter(1.0, None, verbose="error").notch_filter([50.0, 100.0], verbose="error")
    eeg.set_eeg_reference("average", verbose="error")
    hf, hf_rel = muscle_index(eeg.get_data() * 1e6, sf)
    chs = eeg.ch_names
    row["emg_logpow_global"] = float(np.mean(hf))
    row["emg_rel_global"] = float(np.mean(hf_rel))
    for reg, chans in REGIONS.items():
        row[f"emg_rel_{reg}"] = _region_mean(hf_rel, chs, chans)
    del raw, eeg

    # --- segmentos limpios: aperiódico, pico alfa y theta de línea media
    seg = mne.read_epochs(processed / "continuous" / f"sub-{sid}-seg-epo.fif", verbose="error")
    d = seg.get_data(picks="eeg")
    sfs = seg.info["sfreq"]
    f, p = welch(d, fs=sfs, nperseg=d.shape[-1], axis=-1)
    psd = p.mean(0) * 1e12                         # (canales, frecuencias) en µV²/Hz
    ch = seg.ch_names
    used = sorted({c for v in REGIONS.values() for c in v})
    groups = {**REGIONS, "global": used}
    for name, chans in groups.items():
        idx = [ch.index(c) for c in chans if c in ch]
        off, exp_ = aperiodic_fit(f, psd[idx].mean(0))
        row[f"ap_exponent_{name}"], row[f"ap_offset_{name}"] = exp_, off
    idx = [ch.index(c) for c in POSTERIOR if c in ch]
    pp_ = psd[idx].mean(0)
    off, exp_ = aperiodic_fit(f, pp_)
    row["alpha_peak_freq_post"], row["alpha_peak_power_post"] = alpha_peak(f, aperiodic_residual(f, pp_, off, exp_))
    idx = [ch.index(c) for c in FM_THETA if c in ch]
    fm = psd[idx].mean(0)
    th = (f >= 4) & (f < 8)
    tot = (f >= 1) & (f < 40)
    off, exp_ = aperiodic_fit(f, fm)
    row["fmtheta_logpow"] = float(np.log10(np.trapezoid(fm[th], f[th])))
    row["fmtheta_rel"] = float(np.trapezoid(fm[th], f[th]) / np.trapezoid(fm[tot], f[tot]))
    row["fmtheta_periodic"] = float(np.mean(aperiodic_residual(f, fm, off, exp_)[th]))
    del seg, d

    # --- variabilidad P300 de ensayo a ensayo
    ep = mne.read_epochs(processed / "epochs" / f"sub-{sid}-epo.fif", verbose="error")
    cp = [c for c in erp_clusters()["centroparietal"] if c in ep.ch_names]
    x = ep.get_data(picks=cp).mean(1) * 1e6
    w = (ep.times >= 0.25) & (ep.times <= 0.5)
    amp = x[:, w].mean(1)
    row["p300_trial_sd"] = float(np.std(amp, ddof=1))
    row["p300_trial_snr"] = float(np.mean(amp) / np.std(amp, ddof=1))
    return row


def derived(feats: pd.DataFrame) -> pd.DataFrame:
    """Contrastes ERP, promedios entre condiciones y fuerza wPLI (solo transforman columnas existentes)."""
    out = feats[["subject_id", "faa_F3F4", "faa_F7F8"]].copy()
    for m in ("n170L_amp", "n170R_amp", "n170L_lat", "n170R_lat", "p300_amp", "p300_lat", "lpp_amp"):
        out[f"erpavg_{m}"] = feats[[f"erp_{c}_{m}" for c in ("fear", "happy", "sad")]].mean(1)
    for m in ("n170L_amp", "n170R_amp", "p300_amp", "lpp_amp"):
        out[f"erpcon_sad_happy_{m}"] = feats[f"erp_sad_{m}"] - feats[f"erp_happy_{m}"]
        out[f"erpcon_fear_happy_{m}"] = feats[f"erp_fear_{m}"] - feats[f"erp_happy_{m}"]
    regs = list(REGIONS)
    for band in ("theta", "alpha", "beta"):
        cols = [c for c in feats if c.startswith(f"wpli_{band}_")]
        out[f"wplistr_{band}_global"] = feats[cols].mean(1)
        for r in regs:
            rc = [c for c in cols if c.endswith(f"_{r}") or f"_{r}__" in c]
            assert len(rc) == len(regs) - 1, (band, r, len(rc))
            out[f"wplistr_{band}_{r}"] = feats[rc].mean(1)
    for b in ("delta", "theta", "alpha", "beta", "lowgamma"):
        out[f"relglob_{b}"] = feats[[f"rel_{b}_{r}" for r in regs]].mean(1)
    for m in ("sampen", "hfd", "lzc"):
        out[f"cplxglob_{m}"] = feats[[f"{m}_{r}" for r in regs]].mean(1)
    return out


# Conjunto teórico reducido, fijado aquí antes de evaluarlo
COMPACT = (["erpavg_n170L_amp", "erpavg_n170R_amp", "erpavg_n170L_lat", "erpavg_n170R_lat", "erpavg_p300_amp",
            "erpavg_p300_lat", "erpavg_lpp_amp"]
           + [f"erpcon_{c}_{m}" for c in ("sad_happy", "fear_happy") for m in ("n170L_amp", "n170R_amp", "p300_amp",
                                                                               "lpp_amp")]
           + ["ap_exponent_global", "ap_offset_global", "alpha_peak_freq_post", "alpha_peak_power_post",
              "fmtheta_rel", "fmtheta_periodic", "faa_F3F4", "faa_F7F8", "p300_trial_sd", "p300_trial_snr"]
           + [f"wplistr_{b}_global" for b in ("theta", "alpha", "beta")]
           + [f"relglob_{b}" for b in ("delta", "theta", "alpha", "beta", "lowgamma")]
           + [f"cplxglob_{m}" for m in ("sampen", "hfd", "lzc")])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="solo los primeros N sujetos (perfilado)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pp = cfg["preprocessing"]
    processed = Path(cfg["paths"]["processed_dir"])
    log = pd.read_csv(Path(cfg["paths"]["reports_dir"]) / "preprocessing_log.csv", dtype={"subject_id": str})
    audit = pd.read_csv(Path(cfg["paths"]["interim_dir"]) / "audit.csv", dtype={"subject_id": str})
    files = dict(zip(audit["subject_id"], audit["file"]))
    jobs = []
    for _, r in log.iterrows():
        bads = [b for b in str(r["bads_interpolated"]).split(",") if b and b != "nan"]
        jobs.append((r["subject_id"], files[r["subject_id"]], bads))
    if args.limit:
        jobs = jobs[: args.limit]
    rows = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(one_subject)(s, f, b, processed, pp) for s, f, b in jobs)
    new = pd.DataFrame(rows)
    feats = pd.read_parquet(processed / "features" / "subject_features.parquet")
    out = (feats[["subject_id", "group", "included"]].merge(new, on="subject_id")
           .merge(derived(feats), on="subject_id"))
    missing = [c for c in COMPACT if c not in out]
    assert not missing, missing
    if not args.limit:
        out.to_parquet(processed / "features" / "audit_features.parquet", index=False)
        (processed / "features" / "audit_compact_cols.json").write_text(json.dumps(COMPACT, indent=1))
    print(out.shape, f"compactas: {len(COMPACT)}")
    print(out.drop(columns=["subject_id", "group", "included"]).describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
