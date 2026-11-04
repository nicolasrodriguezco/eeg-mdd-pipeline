"""Conectividad (wPLI entre regiones) y complejidad (SampEn, Higuchi, Lempel-Ziv)."""

from __future__ import annotations

from itertools import combinations

import numpy as np

CONN_BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}


def region_signals(data: np.ndarray, ch_names: list[str], regions: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
    """Promedio de los canales de cada región. ``data``: (..., n_channels, n_times)."""
    idx = {c: i for i, c in enumerate(ch_names)}
    names, sigs = [], []
    for reg, chans in regions.items():
        pick = [idx[c] for c in chans if c in idx]
        if pick:
            names.append(reg)
            sigs.append(data[..., pick, :].mean(axis=-2))
    return np.stack(sigs, axis=-2), names


def wpli_features(region_data: np.ndarray, region_names: list[str], sfreq: float,
                  bands: dict = CONN_BANDS) -> dict[str, float]:
    """wPLI por banda entre todos los pares de regiones. ``region_data``: (n_seg, n_reg, n_times)."""
    from mne_connectivity import spectral_connectivity_epochs

    fmin = tuple(b[0] for b in bands.values())
    fmax = tuple(b[1] for b in bands.values())
    con = spectral_connectivity_epochs(region_data, method="wpli", mode="multitaper", sfreq=sfreq,
                                       fmin=fmin, fmax=fmax, faverage=True, verbose="error")
    mat = con.get_data(output="dense")               # (n_reg, n_reg, n_bands), triángulo inferior
    out = {}
    for bi, band in enumerate(bands):
        for i, j in combinations(range(len(region_names)), 2):
            out[f"wpli_{band}_{region_names[i]}__{region_names[j]}"] = float(mat[j, i, bi])
    return out


def complexity_features(region_data: np.ndarray, region_names: list[str], max_segments: int = 100,
                        seed: int = 0) -> dict[str, float]:
    """Media sobre segmentos de SampEn, Higuchi FD y Lempel-Ziv (binarizado por la mediana)."""
    import antropy as ant

    rng = np.random.default_rng(seed)
    n = region_data.shape[0]
    pick = np.sort(rng.choice(n, size=min(n, max_segments), replace=False))
    out = {}
    for r, name in enumerate(region_names):
        se, hfd, lz = [], [], []
        for s in pick:
            x = region_data[s, r].astype(np.float64)
            se.append(ant.sample_entropy(x, order=2))
            hfd.append(ant.higuchi_fd(x, kmax=10))
            lz.append(ant.lziv_complexity((x > np.median(x)).astype(int), normalize=True))
        out[f"sampen_{name}"] = float(np.nanmean(np.where(np.isfinite(se), se, np.nan)))
        out[f"hfd_{name}"] = float(np.mean(hfd))
        out[f"lzc_{name}"] = float(np.mean(lz))
    return out
