"""Eventos de EGI leídos por NOMBRE, con códigos fijos para todo el dataset.

Por qué existe este módulo: al leer un ``.raw`` de EGI, MNE crea un canal de
estímulo por tipo de evento (``fcue``, ``hdot``, ``resp``...) y además un canal
combinado ``STI 014``. En los archivos de MODMA ``STI 014`` vale 1 para todos
los eventos, así que no distingue condiciones; y cualquier numeración que MNE
asigne depende de qué tipos aparecen en cada archivo. Por eso los eventos se
toman siempre por nombre de canal y el código numérico sale de la
configuración, que es igual para todos los sujetos.
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd

COMBINED_STIM = "STI 014"


def egi_event_channels(raw: mne.io.BaseRaw) -> list[str]:
    """Nombres de los canales de evento de EGI (sin el canal combinado)."""
    return [raw.ch_names[i] for i in mne.pick_types(raw.info, stim=True) if raw.ch_names[i] != COMBINED_STIM]


def event_onsets(raw: mne.io.BaseRaw, channel: str) -> np.ndarray:
    """Muestras (relativas al inicio del registro) donde el canal pasa de 0 a >0."""
    row = raw.get_data(picks=[channel])[0] > 0
    return np.flatnonzero(np.diff(np.r_[False, row].astype(np.int8)) == 1)


def count_egi_events(raw: mne.io.BaseRaw) -> dict[str, int]:
    """Número de eventos por nombre de canal de EGI."""
    return {ch: int(len(event_onsets(raw, ch))) for ch in egi_event_channels(raw)}


def find_egi_events(raw: mne.io.BaseRaw, codes: dict[str, int]) -> np.ndarray:
    """Matriz de eventos de MNE (muestra, 0, código) para los nombres pedidos.

    ``codes`` mapea nombre de canal de EGI -> código entero fijo, por ejemplo
    ``{"fcue": 11, "hcue": 12, "scue": 13}``. Los nombres que no estén en el
    archivo simplemente no aportan eventos (se registra aparte quién falta).
    """
    if len(set(codes.values())) != len(codes):
        raise ValueError(f"Hay códigos repetidos en el mapa de eventos: {codes}")
    present = set(egi_event_channels(raw))
    rows = []
    for name, code in codes.items():
        if name not in present:
            continue
        for s in event_onsets(raw, name):
            rows.append((int(s) + raw.first_samp, 0, int(code)))
    if not rows:
        return np.empty((0, 3), dtype=int)
    events = np.array(sorted(rows), dtype=int)
    same = np.flatnonzero(np.diff(events[:, 0]) == 0)
    if len(same):
        raise RuntimeError(
            f"{len(same)} pares de eventos caen en la misma muestra "
            f"(primer caso: muestra {events[same[0], 0]}); revisar el mapa de eventos."
        )
    return events


def event_table(raw: mne.io.BaseRaw, channels: list[str] | None = None) -> pd.DataFrame:
    """Tabla ordenada (tiempo, nombre) con todos los eventos de los canales indicados."""
    channels = channels or egi_event_channels(raw)
    sf = raw.info["sfreq"]
    rows = [(s / sf, ch) for ch in channels for s in event_onsets(raw, ch)]
    return pd.DataFrame(sorted(rows), columns=["t_s", "name"])
