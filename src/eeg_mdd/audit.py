"""Auditoría por registro: qué hay realmente dentro de cada archivo."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .events import count_egi_events
from .io import extract_subject_id, load_raw


def get_events(raw: mne.io.BaseRaw, stim_channel: str | None = None) -> np.ndarray:
    """Eventos desde un canal de estímulo o, si no hay, desde las anotaciones."""
    stim_picks = mne.pick_types(raw.info, stim=True)
    if stim_channel or len(stim_picks):
        try:
            events = mne.find_events(raw, stim_channel=stim_channel, shortest_event=1, verbose="error")
            if len(events):
                return events
        except (ValueError, RuntimeError):
            pass
    if len(raw.annotations):
        events, _ = mne.events_from_annotations(raw, verbose="error")
        return events
    return np.empty((0, 3), dtype=int)


def audit_raw(raw: mne.io.BaseRaw, stim_channel: str | None = None) -> dict:
    """Métricas básicas de un registro ya cargado."""
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    events = get_events(raw, stim_channel)
    codes, counts = np.unique(events[:, 2], return_counts=True) if len(events) else ([], [])

    row = {
        "sfreq": float(raw.info["sfreq"]),
        "n_channels_total": len(raw.ch_names),
        "n_eeg_channels": len(eeg_picks),
        "duration_s": round(raw.times[-1] if len(raw.times) else 0.0, 2),
        "n_events": int(len(events)),
        "event_counts": json.dumps({int(c): int(n) for c, n in zip(codes, counts)}),
        # Conteo por nombre de canal de EGI: es el que sirve para las condiciones.
        "egi_event_counts": json.dumps(count_egi_events(raw)),
        "bads_in_file": ",".join(raw.info["bads"]),
    }

    if len(eeg_picks):
        # Se lee como mucho 60 s para no cargar el registro entero en memoria.
        stop = min(raw.n_times, int(60 * raw.info["sfreq"]))
        data = raw.get_data(picks=eeg_picks, start=0, stop=stop)
        stds = data.std(axis=1)
        flat = [raw.ch_names[eeg_picks[i]] for i in np.where(stds < 1e-7)[0]]
        row.update(
            {
                "flat_channels_first60s": ",".join(flat),
                "n_flat_channels": len(flat),
                "max_abs_uv_first60s": round(float(np.abs(data).max() * 1e6), 1),
            }
        )
    return row


def audit_file(path: str | Path, subject_id_regex: str, stim_channel: str | None = None) -> dict:
    """Audita un archivo. Si falla la lectura, deja el error registrado en vez de saltarlo."""
    path = Path(path)
    row = {"file": str(path), "subject_id": extract_subject_id(path, subject_id_regex), "error": ""}
    try:
        raw = load_raw(path, preload=False)
        row.update(audit_raw(raw, stim_channel))
    except Exception as exc:  # noqa: BLE001 - queremos registrar cualquier fallo
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def audit_all(paths: list[Path], subject_id_regex: str, stim_channel: str | None = None) -> pd.DataFrame:
    return pd.DataFrame([audit_file(p, subject_id_regex, stim_channel) for p in paths])
