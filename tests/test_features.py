"""Pruebas de las características con señales sintéticas de propiedades conocidas."""

import mne
import numpy as np
import pytest

from eeg_mdd.features import erp_clusters, subject_features
from eeg_mdd.features.connectivity_complexity import complexity_features, region_signals, wpli_features
from eeg_mdd.features.erp import erp_features
from eeg_mdd.features.regions import REGIONS, region_map
from eeg_mdd.features.spectral import frontal_alpha_asymmetry, spectral_features

SF = 125.0


def test_region_map_is_disjoint_symmetric_and_excludes_face():
    rm = region_map()
    assert set(rm) == set(REGIONS)
    allch = [c for v in rm.values() for c in v]
    assert len(allch) == len(set(allch))                 # sin solapamiento
    for lobe in ("frontal", "central", "temporal", "parietal", "occipital"):
        assert abs(len(rm[f"{lobe}_L"]) - len(rm[f"{lobe}_R"])) <= 1
        assert len(rm[f"{lobe}_L"]) >= 3
    assert "E24" in rm["frontal_L"] and "E124" in rm["frontal_R"]   # F3 / F4
    assert not {"E125", "E126", "E127", "E128", "Cz", "E62"} & set(allch)


def test_alpha_peak_dominates_relative_power_and_faa_sign():
    rng = np.random.default_rng(0)
    t = np.arange(int(2 * SF)) / SF
    names = ["E24", "E124", "E1"]
    data = rng.normal(0, 1e-6, (30, 3, t.size))
    data[:, 1] += 20e-6 * np.sin(2 * np.pi * 10 * t)          # alfa fuerte a la derecha (F4)
    data[:, 0] += 5e-6 * np.sin(2 * np.pi * 10 * t)
    feats, bp = spectral_features(data, SF, names, {"frontal_R": ["E124"], "frontal_L": ["E24"]})
    assert feats["rel_alpha_frontal_R"] > 0.8
    assert feats["rel_alpha_frontal_R"] > feats["rel_alpha_frontal_L"]
    faa = frontal_alpha_asymmetry(bp["alpha"], names, {"F3F4": ("E24", "E124")})
    assert faa["faa_F3F4"] > 0


def test_erp_features_recover_known_peaks():
    times = np.arange(-0.2, 0.6 + 1e-9, 1 / 250)
    names = ["A", "B", "C"]
    ev = np.zeros((3, times.size))
    ev[0] = -6 * np.exp(-((times - 0.170) ** 2) / (2 * 0.01 ** 2))     # N170 izquierda a 170 ms
    ev[1] = -3 * np.exp(-((times - 0.180) ** 2) / (2 * 0.01 ** 2))
    ev[2] = 8 * np.exp(-((times - 0.350) ** 2) / (2 * 0.05 ** 2))      # P300 a 350 ms
    f = erp_features(ev, times, names, {"n170_L": ["A"], "n170_R": ["B"], "centroparietal": ["C"]}, "erp_x")
    assert f["erp_x_n170L_lat"] == pytest.approx(170, abs=4)
    assert f["erp_x_n170L_amp"] == pytest.approx(-6, abs=0.2)   # el pico cae entre muestras (168/172 ms)
    assert f["erp_x_n170R_lat"] == pytest.approx(180, abs=4)
    assert f["erp_x_p300_lat"] == pytest.approx(350, abs=4)
    assert f["erp_x_p300_amp"] > f["erp_x_lpp_amp"] > 0


def test_wpli_detects_lagged_coupling_only_in_coupled_pair():
    rng = np.random.default_rng(1)
    n_seg, n_t = 60, int(2 * SF)
    t = np.arange(n_t) / SF
    data = rng.normal(0, 1, (n_seg, 3, n_t))
    for s in range(n_seg):
        ph = rng.uniform(0, 2 * np.pi)
        data[s, 0] += 3 * np.sin(2 * np.pi * 10 * t + ph)
        data[s, 1] += 3 * np.sin(2 * np.pi * 10 * t + ph - np.pi / 2)   # desfase constante de 90°
    f = wpli_features(data, ["a", "b", "c"], SF)
    assert f["wpli_alpha_a__b"] > 0.6
    assert f["wpli_alpha_a__c"] < 0.3


def test_complexity_noise_more_complex_than_sine():
    rng = np.random.default_rng(2)
    t = np.arange(int(2 * SF)) / SF
    data = np.stack([np.stack([np.sin(2 * np.pi * 5 * t), rng.normal(size=t.size)]) for _ in range(5)])
    f = complexity_features(data, ["sine", "noise"], max_segments=5)
    for m in ("sampen", "hfd", "lzc"):
        assert f[f"{m}_noise"] > f[f"{m}_sine"]


def test_region_signals_shape():
    data = np.zeros((4, 3, 10))
    data[:, 0] = 1.0
    data[:, 1] = 3.0
    sig, names = region_signals(data, ["x", "y", "z"], {"r1": ["x", "y"], "r2": ["z"], "r3": ["nope"]})
    assert names == ["r1", "r2"] and sig.shape == (4, 2, 10)
    assert np.allclose(sig[:, 0], 2.0)


def test_subject_features_end_to_end_on_synthetic_epochs():
    rng = np.random.default_rng(3)
    mont = mne.channels.make_standard_montage("GSN-HydroCel-129")
    info = mne.create_info(mont.ch_names, 250.0, "eeg")
    info.set_montage(mont)
    n_ep = 12
    ep = mne.EpochsArray(rng.normal(0, 5e-6, (n_ep, 129, 201)), info, tmin=-0.2,
                         events=np.c_[np.arange(n_ep) * 300, np.zeros(n_ep, int), np.tile([11, 12, 13], 4)],
                         event_id={"fear": 11, "happy": 12, "sad": 13}, verbose="error")
    info2 = mne.create_info(mont.ch_names, SF, "eeg")
    info2.set_montage(mont)
    seg = mne.EpochsArray(rng.normal(0, 5e-6, (20, 129, 250)), info2, verbose="error")
    out = subject_features(ep, seg, complexity_segments=3)
    f = out["features"]
    assert len([k for k in f if k.startswith("pow_")]) == 50
    assert len([k for k in f if k.startswith("wpli_")]) == 3 * 45
    assert len([k for k in f if k.startswith("erp_")]) == 3 * 7
    assert len([k for k in f if k.startswith(("sampen_", "hfd_", "lzc_"))]) == 30
    assert all(np.isfinite(v) for v in f.values())
    assert out["trial_regions"].shape == (n_ep, 10, 201)
    assert set(erp_clusters()) == {"n170_L", "n170_R", "centroparietal"}
