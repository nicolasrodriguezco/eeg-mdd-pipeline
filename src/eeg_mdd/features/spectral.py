"""Potencia espectral por banda (Welch) por canal y por región, y asimetría alfa frontal."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0),
         "lowgamma": (30.0, 40.0)}
TOTAL = (1.0, 40.0)


def psd_welch(data: np.ndarray, sfreq: float, seg_len_s: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """PSD media sobre segmentos. ``data``: (n_segments, n_channels, n_times) -> (n_channels, n_freqs)."""
    nper = min(int(seg_len_s * sfreq), data.shape[-1])
    freqs, pxx = welch(data, fs=sfreq, nperseg=nper, noverlap=nper // 2, axis=-1)
    return freqs, pxx.mean(axis=0)


def band_power(freqs: np.ndarray, psd: np.ndarray, bands: dict = BANDS) -> dict[str, np.ndarray]:
    """Potencia absoluta (integral de la PSD) por banda, por canal."""
    out = {}
    for name, (lo, hi) in bands.items():
        m = (freqs >= lo) & (freqs < hi)
        out[name] = np.trapezoid(psd[..., m], freqs[m], axis=-1)
    return out


def spectral_features(data: np.ndarray, sfreq: float, ch_names: list[str], regions: dict[str, list[str]],
                      bands: dict = BANDS) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Características espectrales de un sujeto.

    Devuelve (features por región, potencia absoluta por canal y banda).
    - ``pow_<banda>_<región>``: log10 de la potencia absoluta media de la región (µV²).
    - ``rel_<banda>_<región>``: potencia de la banda / potencia total 1-40 Hz.
    """
    freqs, psd = psd_welch(data, sfreq)
    psd = psd * 1e12                                   # V²/Hz -> µV²/Hz
    bp = band_power(freqs, psd, bands)
    total = band_power(freqs, psd, {"total": TOTAL})["total"]
    idx = {c: i for i, c in enumerate(ch_names)}
    feats: dict[str, float] = {}
    for reg, chans in regions.items():
        pick = [idx[c] for c in chans if c in idx]
        if not pick:
            continue
        tot = total[pick].mean()
        for b in bands:
            p = bp[b][pick].mean()
            feats[f"pow_{b}_{reg}"] = float(np.log10(p))
            feats[f"rel_{b}_{reg}"] = float(p / tot)
    return feats, bp


def frontal_alpha_asymmetry(alpha_power: np.ndarray, ch_names: list[str],
                            pairs: dict[str, tuple[str, str]]) -> dict[str, float]:
    """ln(alfa derecha) - ln(alfa izquierda) para cada par (izq, der)."""
    idx = {c: i for i, c in enumerate(ch_names)}
    return {f"faa_{name}": float(np.log(alpha_power[idx[r]]) - np.log(alpha_power[idx[left]]))
            for name, (left, r) in pairs.items()}
