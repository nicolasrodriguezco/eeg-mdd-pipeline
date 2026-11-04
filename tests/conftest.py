"""EEG sintético de 128 canales para probar el pipeline sin datos reales."""

from __future__ import annotations

import mne
import numpy as np
import pytest

CODES = {"fear": 11, "happy": 12, "sad": 13}


def make_raw(sfreq: float = 500.0, duration_s: float = 40.0, n_events_per_code: int = 10,
             line_freq: float = 50.0, seed: int = 0) -> mne.io.RawArray:
    rng = np.random.default_rng(seed)
    n_times = int(duration_s * sfreq)
    t = np.arange(n_times) / sfreq
    eeg_names = [f"E{i}" for i in range(1, 129)]

    # Ruido de ~5 µV más una interferencia de red de 10 µV.
    eeg = rng.normal(0, 5e-6, (128, n_times)) + 10e-6 * np.sin(2 * np.pi * line_freq * t)

    stim = np.zeros((1, n_times))
    codes = np.repeat(list(CODES.values()), n_events_per_code)
    rng.shuffle(codes)
    onsets = np.linspace(2.0, duration_s - 2.0, len(codes))
    for onset, code in zip(onsets, codes):
        stim[0, int(onset * sfreq)] = code

    info = mne.create_info(eeg_names + ["STI 014"], sfreq, ["eeg"] * 128 + ["stim"])
    raw = mne.io.RawArray(np.vstack([eeg, stim]), info, verbose="error")
    raw.set_montage("GSN-HydroCel-128", on_missing="ignore", verbose="error")
    return raw


@pytest.fixture
def synthetic_raw() -> mne.io.RawArray:
    return make_raw()


@pytest.fixture
def pp_config() -> dict:
    return {
        "l_freq": 0.5,
        "h_freq": 40.0,
        "notch_freqs": [50.0],
        "reference": "average",
        "resample_sfreq": 250,
        "tmin": -0.2,
        "tmax": 0.6,
        "baseline": [-0.2, 0.0],
        "reject_peak_to_peak_uv": 100,
        "event_id": dict(CODES),
        "stim_channel": None,
        "ica": {"enabled": False},
    }
