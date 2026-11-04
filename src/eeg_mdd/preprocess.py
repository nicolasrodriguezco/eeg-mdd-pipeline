"""Preprocesamiento por sujeto con MNE.

Todo se ajusta dentro de un mismo sujeto (detección de canales malos, ICA,
rechazo), así que no hay fuga de información entre sujetos en esta etapa.
El plan completo está fijado en ``docs/preregistration.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .audit import get_events
from .events import find_egi_events


# --------------------------------------------------------------------------- utilidades
def load_ica_exclusions(path: str | Path) -> dict[str, list[int]]:
    """Exclusiones de ICA fijadas a mano (tienen prioridad sobre ICLabel)."""
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    if "reviewer" in df.columns:
        # Las filas escritas automáticamente por ICLabel no son decisiones manuales.
        df = df[~df["reviewer"].str.startswith("ICLabel-auto")]
    out: dict[str, list[int]] = {}
    for sid, comps in zip(df.get("subject_id", []), df.get("exclude_components", [])):
        if sid and comps:
            out[sid.strip()] = [int(c) for c in str(comps).replace(";", ",").split(",") if c.strip()]
    return out


def prepare_raw(raw: mne.io.BaseRaw, pp: dict) -> mne.io.BaseRaw:
    """Renombra canales y pone el montaje (los .raw de EGI no lo traen)."""
    raw = raw.copy().load_data(verbose="error")
    ren = {k: v for k, v in (pp.get("rename_channels") or {}).items() if k in raw.ch_names}
    if ren:
        raw.rename_channels(ren)
    if pp.get("montage"):
        raw.set_montage(pp["montage"], on_missing="ignore", verbose="error")
    return raw


def find_task_events(raw: mne.io.BaseRaw, pp: dict) -> np.ndarray:
    """Eventos por nombre de EGI (si la config los define) o por canal de estímulo."""
    if pp.get("egi_event_codes"):
        return find_egi_events(raw, pp["egi_event_codes"])
    return get_events(raw, pp.get("stim_channel"))


# --------------------------------------------------------------------------- canales malos
def detect_bad_channels(raw: mne.io.BaseRaw, cfg: dict) -> tuple[list[str], dict[str, str]]:
    """Canales malos por planitud, desviación robusta y baja correlación con vecinos.

    Se calcula sobre una copia filtrada 1-40 Hz de los canales EEG. Devuelve la
    lista y el motivo de cada uno. Con ``method: pyprep`` se usa la detección del
    pipeline PREP (PyPREP ``NoisyChannels``, sin RANSAC).
    """
    if cfg.get("method") == "pyprep":
        return _detect_bad_pyprep(raw, cfg)
    eeg = raw.copy().pick("eeg").filter(1.0, 40.0, verbose="error")
    names = eeg.ch_names
    data = eeg.get_data()
    sd = data.std(axis=1)
    reasons: dict[str, str] = {}
    exempt = set(cfg.get("exempt_flat", []))

    for n, s in zip(names, sd):
        if s < cfg.get("flat_sd_uv", 0.1) * 1e-6 and n not in exempt:
            reasons[n] = "flat"

    live = np.array([n not in exempt or s > 1e-12 for n, s in zip(names, sd)])
    log_sd = np.log(np.maximum(sd, 1e-15))
    med = np.median(log_sd[live])
    mad = np.median(np.abs(log_sd[live] - med)) * 1.4826 or 1e-12
    z = (log_sd - med) / mad
    for n, zz, ok in zip(names, z, live):
        if ok and abs(zz) > cfg.get("robust_z", 5.0) and n not in reasons:
            reasons[n] = f"robust_z={zz:.1f}"

    pos = np.array([eeg.info["chs"][i]["loc"][:3] for i in range(len(names))])
    if np.isfinite(pos).all() and np.abs(pos).sum() > 0:
        k = cfg.get("n_neighbors", 6)
        corr = np.corrcoef(data)
        dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        for i, n in enumerate(names):
            if n in reasons or not live[i]:
                continue
            nb = [j for j in np.argsort(dist[i])[1:] if live[j]][:k]
            c = np.median(corr[i, nb])
            if c < cfg.get("min_neighbor_corr", 0.4):
                reasons[n] = f"neighbor_corr={c:.2f}"
    return list(reasons), reasons


def _detect_bad_pyprep(raw: mne.io.BaseRaw, cfg: dict) -> tuple[list[str], dict[str, str]]:
    import logging

    import pyprep

    exempt = set(cfg.get("exempt_flat", []))
    eeg = raw.copy().pick("eeg")
    old = mne.set_log_level("ERROR", return_old_level=True)
    logging.getLogger("pyprep").setLevel(logging.ERROR)
    try:
        nc = pyprep.NoisyChannels(eeg, do_detrend=True, random_state=cfg.get("random_state", 97))
        nc.find_all_bads(ransac=bool(cfg.get("ransac", False)))
    finally:
        mne.set_log_level(old)
    reasons: dict[str, list[str]] = {}
    for kind, chans in nc.get_bads(as_dict=True).items():
        if kind == "bad_all":
            continue
        for c in chans:
            if c in exempt:      # la referencia de registro (Cz) es plana por definición
                continue
            reasons.setdefault(c, []).append(kind.replace("bad_by_", ""))
    out = {c: "pyprep:" + "+".join(v) for c, v in reasons.items()}
    return list(out), out


# --------------------------------------------------------------------------- ICA
def fit_ica_iclabel(raw: mne.io.BaseRaw, ica_cfg: dict, notch: list[float]) -> tuple[mne.preprocessing.ICA, pd.DataFrame]:
    """Ajusta ICA en una copia 1-100 Hz con referencia promedio y la etiqueta con ICLabel."""
    fit_raw = raw.copy().pick("eeg")
    nyq = fit_raw.info["sfreq"] / 2
    h = ica_cfg.get("lowpass_for_fit")
    h = h if h and h < nyq else None
    fit_raw.filter(ica_cfg.get("highpass_for_fit", 1.0), h, verbose="error")
    notch_fit = [g for f in notch for g in (f, 2 * f) if g < nyq - 1]
    if notch_fit:
        fit_raw.notch_filter(sorted(set(notch_fit)), verbose="error")
    fit_raw.set_eeg_reference("average", projection=False, verbose="error")

    ica = mne.preprocessing.ICA(
        n_components=ica_cfg.get("n_components", 30),
        method=ica_cfg.get("method", "picard"),
        fit_params=ica_cfg.get("fit_params"),
        random_state=ica_cfg.get("random_state", 97),
        max_iter="auto",
    )
    ica.fit(fit_raw, picks="eeg", decim=ica_cfg.get("decim_for_fit"), verbose="error")

    labels = pd.DataFrame({"component": range(ica.n_components_), "label": "", "proba": np.nan})
    try:
        from mne_icalabel import label_components

        res = label_components(fit_raw, ica, method="iclabel")
        labels["label"] = res["labels"]
        labels["proba"] = res["y_pred_proba"]
    except ImportError:  # pragma: no cover - mne-icalabel es dependencia del proyecto
        pass
    thr = ica_cfg.get("iclabel_threshold", 0.8)
    reject = set(ica_cfg.get("iclabel_reject", []))
    labels["exclude"] = labels["label"].isin(reject) & (labels["proba"] >= thr)
    return ica, labels


# --------------------------------------------------------------------------- pipeline
@dataclass
class SubjectResult:
    epochs: mne.Epochs
    log: dict
    continuous: mne.Epochs | None = None
    ica: mne.preprocessing.ICA | None = None
    ica_labels: pd.DataFrame | None = None
    bad_reasons: dict = field(default_factory=dict)


def _reject_abs(epochs: mne.BaseEpochs, thr_uv: float | None, reason: str) -> None:
    if not thr_uv or not len(epochs):
        return
    peak = np.abs(epochs.get_data(picks="eeg")).max(axis=(1, 2))
    bad = np.flatnonzero(peak > thr_uv * 1e-6)
    if len(bad):
        epochs.drop(bad, reason=reason, verbose="error")


def preprocess_subject(
    raw: mne.io.BaseRaw,
    pp: dict,
    subject_id: str,
    ica_exclude: list[int] | None = None,
    ica_dir: str | Path | None = None,
) -> SubjectResult:
    """De registro continuo a épocas limpias (y segmentos continuos) de un sujeto."""
    if not pp.get("event_id"):
        raise ValueError(
            "preprocessing.event_id está vacío en config.yaml. "
            "Complétalo con los códigos de evento que mostró la auditoría."
        )
    log: dict = {"subject_id": subject_id}
    raw = prepare_raw(raw, pp)

    # 1) Eventos antes de quitar los canales de estímulo.
    events = find_task_events(raw, pp)
    log["n_events_found"] = int(len(events))

    # 2) Solo EEG.
    raw.pick("eeg")
    log["n_eeg_channels"] = len(raw.ch_names)

    # 3) Canales malos (automáticos + los ya marcados en el archivo).
    reasons: dict[str, str] = {b: "marked_in_file" for b in raw.info["bads"]}
    if pp.get("bad_channels"):
        _, auto = detect_bad_channels(raw, pp["bad_channels"])
        reasons.update(auto)
    raw.info["bads"] = list(reasons)
    log["n_bad_channels"] = len(reasons)
    log["bad_channels"] = ";".join(f"{k}:{v}" for k, v in reasons.items())

    # 4) ICA (ajustada en una copia propia del sujeto).
    ica_cfg = pp.get("ica", {})
    notch = [float(f) for f in pp.get("notch_freqs", []) or []]
    ica, labels = None, None
    log["ica_excluded"] = ""
    if ica_cfg.get("enabled"):
        if ica_cfg.get("method", "fastica") == "picard" or ica_cfg.get("iclabel_threshold") is not None:
            ica, labels = fit_ica_iclabel(raw, ica_cfg, notch)
        else:  # modo simple (FastICA sin ICLabel), usado en pruebas antiguas
            fit_raw = raw.copy().filter(ica_cfg.get("highpass_for_fit", 1.0), None, verbose="error")
            ica = mne.preprocessing.ICA(n_components=ica_cfg.get("n_components", 0.99),
                                        method=ica_cfg.get("method", "fastica"),
                                        random_state=ica_cfg.get("random_state", 97), max_iter="auto")
            ica.fit(fit_raw, verbose="error")
        if ica_dir:
            Path(ica_dir).mkdir(parents=True, exist_ok=True)
            ica.save(Path(ica_dir) / f"sub-{subject_id}-ica.fif", overwrite=True, verbose="error")
        if ica_exclude is not None:
            exclude, source = list(ica_exclude), "manual"
        elif labels is not None:
            exclude, source = labels.loc[labels["exclude"], "component"].astype(int).tolist(), "iclabel"
        else:
            exclude, source = [], "none"
        ica.exclude = exclude
        log["ica_n_components"] = int(ica.n_components_)
        log["ica_excluded"] = ",".join(str(c) for c in exclude)
        log["ica_n_excluded"] = len(exclude)
        log["ica_exclusion_source"] = source
        if labels is not None:
            for lab in ("eye blink", "muscle artifact", "heart beat", "line noise", "channel noise"):
                log[f"ica_n_{lab.replace(' ', '_')}"] = int((labels["exclude"] & (labels["label"] == lab)).sum())

    # 5) Copia de análisis: filtros -> referencia promedio (sin malos) -> ICA -> interpolación -> referencia.
    raw.filter(pp["l_freq"], pp["h_freq"], fir_design="firwin", verbose="error")
    notch_ok = [f for f in notch if f < raw.info["sfreq"] / 2]
    if notch_ok:
        raw.notch_filter(notch_ok, verbose="error")
    log["notch_freqs"] = ",".join(str(f) for f in notch)
    if ica is not None and ica.exclude:
        raw.set_eeg_reference("average", projection=False, verbose="error")  # igual que al ajustar
        ica.apply(raw, verbose="error")
    log["bads_interpolated"] = ",".join(raw.info["bads"])
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True, verbose="error")
    raw.set_eeg_reference(pp.get("reference", "average"), projection=False, verbose="error")

    # 6) Épocas.
    event_id = {k: int(v) for k, v in pp["event_id"].items()}
    present = {k: v for k, v in event_id.items() if len(events) and v in set(events[:, 2])}
    for cond, code in event_id.items():
        log[f"n_trials_raw_{cond}"] = int((events[:, 2] == code).sum()) if len(events) else 0
    if not present:
        raise RuntimeError(f"{subject_id}: ninguno de los códigos de event_id aparece en el registro")

    reject = {"eeg": pp["reject_peak_to_peak_uv"] * 1e-6} if pp.get("reject_peak_to_peak_uv") else None
    baseline = tuple(pp["baseline"]) if pp.get("baseline") else None
    target = pp.get("resample_sfreq")
    sfreq = raw.info["sfreq"]
    decim, method = 1, "none"
    if target and sfreq != target:
        ratio = sfreq / target
        if float(ratio).is_integer() and pp["h_freq"] < target / 2:
            decim, method = int(ratio), f"decim x{int(ratio)}"
        else:
            method = "resample"
    log["downsample_method"] = method

    epochs = mne.Epochs(raw, events, event_id=present, tmin=pp["tmin"], tmax=pp["tmax"], baseline=baseline,
                        reject=reject, decim=decim, preload=True, on_missing="ignore", verbose="error")
    if method == "resample" and len(epochs):
        epochs.resample(target, verbose="error")
    _reject_abs(epochs, pp.get("reject_abs_uv"), f"ABS_{pp.get('reject_abs_uv')}uV")

    for cond in event_id:
        log[f"n_trials_kept_{cond}"] = len(epochs[cond]) if cond in epochs.event_id else 0
    log["n_trials_kept_total"] = len(epochs)
    log["sfreq_final"] = float(epochs.info["sfreq"])

    # 7) Segmentos continuos limpios (para espectro, conectividad y complejidad).
    cont = None
    ccfg = pp.get("continuous")
    if ccfg and len(events):
        sf = raw.info["sfreq"]
        t0 = max(0.0, (events[0, 0] - raw.first_samp) / sf - 1.0)
        t1 = min(raw.times[-1], (events[-1, 0] - raw.first_samp) / sf + 2.0)
        task = raw.copy().crop(t0, t1)
        if ccfg.get("sfreq") and ccfg["sfreq"] < sf:
            task.resample(ccfg["sfreq"], verbose="error")
        cont = mne.make_fixed_length_epochs(task, duration=ccfg.get("segment_s", 2.0), preload=True,
                                            verbose="error")
        n_seg = len(cont)
        _reject_abs(cont, ccfg.get("reject_abs_uv"), "ABS_cont")
        log["n_segments_total"] = n_seg
        log["n_segments_kept"] = len(cont)

    return SubjectResult(epochs=epochs, log=log, continuous=cont, ica=ica, ica_labels=labels, bad_reasons=reasons)


def preprocess_raw(raw, pp, subject_id, ica_exclude=None, ica_dir=None) -> tuple[mne.Epochs, dict]:
    """Interfaz anterior: devuelve solo las épocas y el registro."""
    res = preprocess_subject(raw, pp, subject_id, ica_exclude, ica_dir)
    return res.epochs, res.log


def apply_exclusion_criteria(log: pd.DataFrame, crit: dict, n_channels: int,
                             conditions: list[str]) -> pd.DataFrame:
    """Marca a los sujetos que no cumplen los criterios fijados de antemano."""
    out = log.copy()
    reasons = []
    for _, r in out.iterrows():
        why = []
        if isinstance(r.get("error"), str) and r["error"]:
            why.append("preprocessing error")
        else:
            if r.get("n_bad_channels", 0) > crit["max_bad_channel_frac"] * n_channels:
                why.append(f"bad channels {int(r['n_bad_channels'])} > {crit['max_bad_channel_frac']:.0%}")
            if r.get("n_trials_kept_total", 0) < crit["min_trials_total"]:
                why.append(f"trials {int(r['n_trials_kept_total'])} < {crit['min_trials_total']}")
            for c in conditions:
                n = r.get(f"n_trials_kept_{c}", 0)
                if n < crit["min_trials_per_condition"]:
                    why.append(f"{c} trials {int(n)} < {crit['min_trials_per_condition']}")
        reasons.append("; ".join(why))
    out["exclusion_reason"] = reasons
    out["included"] = out["exclusion_reason"] == ""
    return out
