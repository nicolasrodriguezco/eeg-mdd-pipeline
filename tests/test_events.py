"""Lectura de eventos de EGI por nombre de canal, con códigos fijos."""

import mne
import numpy as np
import pytest

from eeg_mdd.checks import check_audit
from eeg_mdd.events import count_egi_events, egi_event_channels, event_table, find_egi_events

import pandas as pd


def _egi_like_raw(order=("fcue", "hcue", "scue"), sfreq=250.0, n=5, seed=0):
    """Imita un .raw de EGI: un canal de estímulo por tipo de evento + STI 014 que vale 1 siempre."""
    rng = np.random.default_rng(seed)
    n_times = int(60 * sfreq)
    names = ["fcue", "hcue", "scue", "fdot", "hdot", "sdot", "resp"]
    stim = {nm: np.zeros(n_times) for nm in names}
    t = 2.0
    truth = []
    for cue in order:
        for _ in range(n):
            s = int(t * sfreq)
            stim[cue][s] = 1                          # en EGI cada canal es 0/1
            dot = cue[0] + "dot"
            stim[dot][s + 175] = 1
            truth.append((s, cue))
            t += 1.5
    combined = np.clip(sum(stim.values()), 0, 1)       # STI 014 = 1 para todo
    eeg = rng.normal(0, 5e-6, (4, n_times))
    info = mne.create_info([f"E{i}" for i in range(1, 5)] + names + ["STI 014"], sfreq,
                           ["eeg"] * 4 + ["stim"] * (len(names) + 1))
    data = np.vstack([eeg] + [stim[nm] for nm in names] + [combined])
    return mne.io.RawArray(data, info, verbose="error"), truth


def test_channels_and_counts_ignore_combined_channel():
    raw, _ = _egi_like_raw(n=4)
    assert "STI 014" not in egi_event_channels(raw)
    counts = count_egi_events(raw)
    assert counts["fcue"] == counts["hcue"] == counts["scue"] == 4
    assert counts["fdot"] == 4 and counts["resp"] == 0


def test_codes_are_fixed_regardless_of_order_in_file():
    codes = {"fcue": 11, "hcue": 12, "scue": 13}
    raw_a, truth_a = _egi_like_raw(order=("fcue", "hcue", "scue"))
    raw_b, truth_b = _egi_like_raw(order=("scue", "fcue", "hcue"))
    for raw, truth in ((raw_a, truth_a), (raw_b, truth_b)):
        ev = find_egi_events(raw, codes)
        assert len(ev) == 15
        expected = sorted((s, codes[name]) for s, name in truth)
        assert [(int(s), int(c)) for s, _, c in ev] == expected


def test_missing_names_give_no_events_and_duplicate_codes_fail():
    raw, _ = _egi_like_raw()
    assert len(find_egi_events(raw, {"nope": 1})) == 0
    with pytest.raises(ValueError):
        find_egi_events(raw, {"fcue": 1, "hcue": 1})


def test_event_table_is_sorted():
    raw, _ = _egi_like_raw(n=2)
    tab = event_table(raw, ["fcue", "fdot"])
    assert tab["t_s"].is_monotonic_increasing
    assert tab["name"].tolist() == ["fcue", "fdot", "fcue", "fdot"]


def test_reference_channel_allowed_flat_and_nan_error_is_not_an_error():
    audit = pd.DataFrame([{"file": "a", "subject_id": "1", "error": np.nan, "n_eeg_channels": 129,
                           "sfreq": 250.0, "n_events": 10, "n_flat_channels": 1,
                           "flat_channels_first60s": "E129"}])
    assert check_audit(audit, 129, 250, allowed_flat=["E129"]) == []
    assert "canales planos" in " ".join(check_audit(audit, 129, 250))
