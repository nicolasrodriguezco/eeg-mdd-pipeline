"""Características de la auditoría (D6, exploratorias y post hoc).

Ninguna usa etiquetas. Se calculan por sujeto y por eso pueden quedar fuera de la CV.

- ``aperiodic_fit``: ajuste 1/f robusto (estilo FOOOF, sin "knee") del espectro log-log.
- ``alpha_peak``: frecuencia del pico alfa (centroide del residuo positivo) y su altura.
- ``behavioural_rt``: tiempos de reacción desde los eventos de EGI (``*dot`` → ``resp``).
- ``muscle_index``: potencia de 55–95 Hz en el registro crudo, como índice de EMG.
"""

from __future__ import annotations

import numpy as np

FIT_RANGE = (2.0, 35.0)          # lejos del pasa-altos de 0.5 Hz y del pasa-bajos de 40 Hz


def aperiodic_fit(freqs: np.ndarray, psd: np.ndarray, frange: tuple[float, float] = FIT_RANGE) -> tuple[float, float]:
    """(desfase, exponente) de ``log10 P = b - chi * log10 f``.

    Primero hace un ajuste inicial y luego lo repite solo con los puntos que quedan por debajo de él. Así los
    picos oscilatorios no sesgan la pendiente: es la lógica de ``_robust_ap_fit`` de FOOOF.
    """
    m = (freqs >= frange[0]) & (freqs <= frange[1])
    lf, lp = np.log10(freqs[m]), np.log10(psd[m])
    A = np.c_[np.ones_like(lf), -lf]
    coef, *_ = np.linalg.lstsq(A, lp, rcond=None)
    below = (lp - A @ coef) <= 0
    if below.sum() >= 4:
        coef, *_ = np.linalg.lstsq(A[below], lp[below], rcond=None)
    return float(coef[0]), float(coef[1])


def aperiodic_residual(freqs: np.ndarray, psd: np.ndarray, offset: float, exponent: float) -> np.ndarray:
    """log10 P menos el componente aperiódico (en unidades log10)."""
    f = np.where(freqs > 0, freqs, np.nan)
    return np.log10(psd) - (offset - exponent * np.log10(f))


def alpha_peak(freqs: np.ndarray, resid: np.ndarray, band: tuple[float, float] = (7.0, 13.0)) -> tuple[float, float]:
    """(frecuencia del pico alfa, altura máxima del residuo) con el centroide del residuo positivo en ``band``."""
    m = (freqs >= band[0]) & (freqs <= band[1])
    r = np.clip(resid[m], 0, None)
    if not np.isfinite(r).all() or r.sum() <= 0:
        return float("nan"), float("nan")
    return float((freqs[m] * r).sum() / r.sum()), float(r.max())


def behavioural_rt(dot_onsets_s: dict[str, np.ndarray], resp_onsets_s: np.ndarray,
                   valid: tuple[float, float] = (0.15, 1.5)) -> dict[str, float]:
    """Tiempos de reacción por sujeto.

    - ``dot_onsets_s``: inicios del punto por condición en segundos, por ejemplo ``{"fear": ..., "happy": ..., ...}``.
    - ``resp_onsets_s``: inicios de las respuestas en segundos.

    El TR de un ensayo es la primera respuesta posterior al punto, si cae dentro de ``valid``. Si no, el ensayo cuenta
    como omisión o respuesta fuera de rango. Las claves de TRSP (congruencia, acierto) no están en los ``.raw`` de EGI,
    así que no hay índice de sesgo atencional.
    """
    resp = np.sort(np.asarray(resp_onsets_s, dtype=float))
    out, all_rt, n_all = {}, [], 0
    for cond, dots in dot_onsets_s.items():
        dots = np.sort(np.asarray(dots, dtype=float))
        idx = np.searchsorted(resp, dots, side="right")
        rt = np.full(len(dots), np.nan)
        ok = idx < len(resp)
        rt[ok] = resp[idx[ok]] - dots[ok]
        rt[(rt < valid[0]) | (rt > valid[1])] = np.nan
        out[f"beh_rt_median_{cond}"] = float(np.nanmedian(rt)) * 1000 if np.isfinite(rt).any() else float("nan")
        all_rt.append(rt)
        n_all += len(dots)
    rt = np.concatenate(all_rt)
    good = rt[np.isfinite(rt)]
    out["beh_rt_median"] = float(np.median(good)) * 1000
    out["beh_rt_sd"] = float(np.std(good, ddof=1)) * 1000
    out["beh_rt_cv"] = float(np.std(good, ddof=1) / np.mean(good))
    out["beh_miss_rate"] = float(1 - len(good) / n_all)
    if {"beh_rt_median_sad", "beh_rt_median_happy"} <= out.keys():
        out["beh_rt_sad_minus_happy"] = out["beh_rt_median_sad"] - out["beh_rt_median_happy"]
    return out


def muscle_index(data_uv: np.ndarray, sfreq: float, seg_s: float = 2.0, band: tuple[float, float] = (55.0, 95.0),
                 ref_band: tuple[float, float] = (1.0, 40.0)) -> tuple[np.ndarray, np.ndarray]:
    """Índice de EMG por canal: log10 de la potencia media en ``band`` y su cociente con ``ref_band``.

    ``data_uv`` (n_canales, n_muestras) ya filtrado pasa-altos, con notch y referencia promedio. Se usa la mediana
    sobre segmentos de 2 s, que es robusta a artefactos aislados grandes.
    """
    from scipy.signal import welch

    n = int(seg_s * sfreq)
    n_seg = data_uv.shape[1] // n
    seg = data_uv[:, : n_seg * n].reshape(data_uv.shape[0], n_seg, n).transpose(1, 0, 2)
    f, p = welch(seg, fs=sfreq, nperseg=n, axis=-1)
    hf = (f >= band[0]) & (f <= band[1]) & (np.abs(f - 50) > 3) & (np.abs(f - 100) > 3)
    lf = (f >= ref_band[0]) & (f <= ref_band[1])
    hf_pow = np.median(p[..., hf].mean(-1), axis=0)
    lf_pow = np.median(p[..., lf].mean(-1), axis=0)
    return np.log10(hf_pow), np.log10(hf_pow / lf_pow)
