"""Preprocesamiento por sujeto con MNE.

Todo se ajusta dentro de un mismo sujeto (filtros, ICA, rechazo), así que no
hay fuga de información entre sujetos en esta etapa.
"""

from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd

from .audit import get_events


def load_ica_exclusions(path: str | Path) -> dict[str, list[int]]:
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    out: dict[str, list[int]] = {}
    for sid, comps in zip(df.get("subject_id", []), df.get("exclude_components", [])):
        if sid and comps:
            out[sid.strip()] = [int(c) for c in str(comps).replace(";", ",").split(",") if c.strip()]
    return out


def preprocess_raw(
    raw: mne.io.BaseRaw,
    pp: dict,
    subject_id: str,
    ica_exclude: list[int] | None = None,
    ica_dir: str | Path | None = None,
) -> tuple[mne.Epochs, dict]:
    """De registro continuo a épocas limpias de un sujeto.

    Devuelve las épocas y un registro con lo que pasó en cada paso.
    """
    if not pp.get("event_id"):
        raise ValueError(
            "preprocessing.event_id está vacío en config.yaml. "
            "Complétalo con los códigos de evento que mostró la auditoría."
        )
    log: dict = {"subject_id": subject_id}
    raw = raw.copy().load_data(verbose="error")

    # 1) Eventos antes de quitar el canal de estímulo.
    events = get_events(raw, pp.get("stim_channel"))
    log["n_events_found"] = int(len(events))

    # 2) Solo EEG.
    raw.pick("eeg")
    log["n_eeg_channels"] = len(raw.ch_names)

    # 3) Filtros (FIR de fase cero, el predeterminado de MNE).
    raw.filter(pp["l_freq"], pp["h_freq"], fir_design="firwin", verbose="error")
    if pp.get("notch_freqs"):
        notch = [f for f in pp["notch_freqs"] if f < raw.info["sfreq"] / 2]
        if notch:
            raw.notch_filter(notch, verbose="error")
    log["notch_freqs"] = ",".join(str(f) for f in pp.get("notch_freqs", []))

    # 4) Canales malos marcados: se interpolan (requiere montaje con posiciones).
    log["bads_interpolated"] = ",".join(raw.info["bads"])
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True, verbose="error")

    # 5) ICA: se ajusta y se guarda; los componentes a quitar se eligen a mano.
    ica_cfg = pp.get("ica", {})
    log["ica_excluded"] = ""
    if ica_cfg.get("enabled"):
        fit_raw = raw.copy().filter(ica_cfg.get("highpass_for_fit", 1.0), None, verbose="error")
        ica = mne.preprocessing.ICA(
            n_components=ica_cfg.get("n_components", 0.99),
            method=ica_cfg.get("method", "fastica"),
            random_state=ica_cfg.get("random_state", 97),
            max_iter="auto",
        )
        ica.fit(fit_raw, verbose="error")
        if ica_dir:
            Path(ica_dir).mkdir(parents=True, exist_ok=True)
            ica.save(Path(ica_dir) / f"sub-{subject_id}-ica.fif", overwrite=True, verbose="error")
        if ica_exclude:
            ica.exclude = list(ica_exclude)
            ica.apply(raw, verbose="error")
            log["ica_excluded"] = ",".join(str(c) for c in ica_exclude)

    # 6) Referencia promedio.
    raw.set_eeg_reference(pp.get("reference", "average"), projection=False, verbose="error")

    # 7) Épocas con corrección de línea base y rechazo por amplitud pico a pico.
    event_id = {k: int(v) for k, v in pp["event_id"].items()}
    present = {k: v for k, v in event_id.items() if v in set(events[:, 2])}
    for cond, code in event_id.items():
        log[f"n_trials_raw_{cond}"] = int((events[:, 2] == code).sum()) if len(events) else 0
    if not present:
        raise RuntimeError(f"{subject_id}: ninguno de los códigos de event_id aparece en el registro")

    reject = {"eeg": pp["reject_peak_to_peak_uv"] * 1e-6} if pp.get("reject_peak_to_peak_uv") else None
    baseline = tuple(pp["baseline"]) if pp.get("baseline") else None

    # 8) Reducción de la frecuencia de muestreo. Si el factor es entero y el
    #    pasa-bajos ya está por debajo de la nueva frecuencia de Nyquist, se
    #    diezma al crear las épocas: conserva muestras exactas (sin desplazar
    #    eventos) y mantiene ambos extremos (-0.2 y 0.6 s -> 201 muestras a 250 Hz).
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

    epochs = mne.Epochs(
        raw,
        events,
        event_id=present,
        tmin=pp["tmin"],
        tmax=pp["tmax"],
        baseline=baseline,
        reject=reject,
        decim=decim,
        preload=True,
        on_missing="ignore",
        verbose="error",
    )
    if method == "resample" and len(epochs):
        epochs.resample(target, verbose="error")

    for cond in event_id:
        log[f"n_trials_kept_{cond}"] = len(epochs[cond]) if cond in epochs.event_id else 0
    log["n_trials_kept_total"] = len(epochs)
    log["sfreq_final"] = float(epochs.info["sfreq"])
    return epochs, log
