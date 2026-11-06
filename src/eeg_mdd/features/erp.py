"""Componentes de ERP con ventanas fijadas a priori (no derivadas de los datos).

Ventanas (desde la literatura de rostros emocionales):
- N170: 140-200 ms, mínimo en electrodos occipito-temporales (≈P7/P8).
- P300: 250-500 ms, media y latencia del máximo en un clúster centro-parietal (≈Pz).
- LPP: 400-600 ms, media en el mismo clúster.

La LPP se limita a 600 ms por el diseño: el punto de la tarea aparece >= 632 ms después del rostro.
"""

from __future__ import annotations

import numpy as np

WINDOWS = {"n170": (0.140, 0.200), "p300": (0.250, 0.500), "lpp": (0.400, 0.600)}


def _win(times: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (times >= lo - 1e-9) & (times <= hi + 1e-9)


def erp_features(evoked_uv: np.ndarray, times: np.ndarray, ch_names: list[str], clusters: dict[str, list[str]],
                 prefix: str) -> dict[str, float]:
    """``evoked_uv``: (n_channels, n_times) en µV. ``clusters``: n170_L, n170_R, centroparietal."""
    idx = {c: i for i, c in enumerate(ch_names)}

    def wave(name):
        return evoked_uv[[idx[c] for c in clusters[name] if c in idx]].mean(axis=0)

    out: dict[str, float] = {}
    m = _win(times, *WINDOWS["n170"])
    for side in ("L", "R"):
        w = wave(f"n170_{side}")[m]
        i = int(np.argmin(w))
        out[f"{prefix}_n170{side}_amp"] = float(w[i])
        out[f"{prefix}_n170{side}_lat"] = float(times[m][i] * 1000)
    cp = wave("centroparietal")
    m = _win(times, *WINDOWS["p300"])
    out[f"{prefix}_p300_amp"] = float(cp[m].mean())
    out[f"{prefix}_p300_lat"] = float(times[m][int(np.argmax(cp[m]))] * 1000)
    m = _win(times, *WINDOWS["lpp"])
    out[f"{prefix}_lpp_amp"] = float(cp[m].mean())
    return out
