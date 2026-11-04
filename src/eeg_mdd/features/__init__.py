"""Biomarcadores de EEG por sujeto (no usan etiquetas, así que pueden calcularse fuera de la CV)."""

from __future__ import annotations

import mne
import numpy as np

from .connectivity_complexity import complexity_features, region_signals, wpli_features
from .erp import erp_features
from .regions import EQUIV_1020, neighbors_of, region_map
from .spectral import BANDS, frontal_alpha_asymmetry, spectral_features

FAA_PAIRS = {"F3F4": (EQUIV_1020["F3"], EQUIV_1020["F4"]), "F7F8": (EQUIV_1020["F7"], EQUIV_1020["F8"])}


def erp_clusters() -> dict[str, list[str]]:
    return {"n170_L": neighbors_of(EQUIV_1020["P7"], 2), "n170_R": neighbors_of(EQUIV_1020["P8"], 2),
            "centroparietal": neighbors_of(EQUIV_1020["Pz"], 6)}


FAMILIES = {"spectral": ("pow_", "rel_", "faa_"), "erp": ("erp_",), "connectivity": ("wpli_",),
            "complexity": ("sampen_", "hfd_", "lzc_")}


def family_of(col: str) -> str | None:
    for fam, prefixes in FAMILIES.items():
        if col.startswith(prefixes):
            return fam
    return None


def subject_features(epochs: mne.BaseEpochs, segments: mne.BaseEpochs, complexity_segments: int = 100) -> dict:
    """Todas las características de un sujeto.

    Devuelve un dict con:
      features         dict nombre -> valor (fila de la tabla por sujeto)
      channel_power    (n_bands, n_channels) potencia absoluta por canal (µV²), para topomapas
      evokeds          lista de Evoked por condición
      trial_regions    (n_trials, n_regions, n_times) señal regional por ensayo (para Riemann)
      trial_conditions nombres de condición por ensayo
    """
    regions = region_map()
    feats: dict[str, float] = {}

    seg = segments.get_data(picks="eeg")
    sf = segments.info["sfreq"]
    ch = segments.ch_names
    spec, bp = spectral_features(seg, sf, ch, regions)
    feats.update(spec)
    feats.update(frontal_alpha_asymmetry(bp["alpha"], ch, FAA_PAIRS))

    reg_seg, reg_names = region_signals(seg, ch, regions)
    feats.update(wpli_features(reg_seg, reg_names, sf))
    feats.update(complexity_features(reg_seg, reg_names, max_segments=complexity_segments))

    clusters = erp_clusters()
    evokeds = []
    for cond in epochs.event_id:
        ev = epochs[cond].average()
        ev.comment = cond
        evokeds.append(ev)
        feats.update(erp_features(ev.data * 1e6, ev.times, ev.ch_names, clusters, f"erp_{cond}"))

    trial_regions, _ = region_signals(epochs.get_data(picks="eeg"), epochs.ch_names, regions)
    inv = {v: k for k, v in epochs.event_id.items()}
    return {
        "features": feats,
        "channel_power": np.stack([bp[b] for b in BANDS]),
        "channel_names": ch,
        "evokeds": evokeds,
        "trial_regions": trial_regions.astype(np.float32),
        "trial_conditions": np.array([inv[c] for c in epochs.events[:, 2]]),
        "region_names": reg_names,
    }
