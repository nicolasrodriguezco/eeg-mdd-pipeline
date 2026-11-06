"""Pruebas sintéticas de las características y la desconfusión de la auditoría (D6)."""

import numpy as np

from eeg_mdd.features.audit import aperiodic_fit, aperiodic_residual, alpha_peak, behavioural_rt, muscle_index
from eeg_mdd.models.deconfound import ConfoundRegressor


def test_aperiodic_recovers_exponent_with_alpha_peak():
    f = np.arange(0.5, 40.5, 0.5)
    psd = 10 ** (1.0 - 1.5 * np.log10(f)) * (1 + 3 * np.exp(-0.5 * ((f - 10) / 1.0) ** 2))
    off, exp_ = aperiodic_fit(f, psd)
    assert abs(exp_ - 1.5) < 0.1 and abs(off - 1.0) < 0.15
    apf, height = alpha_peak(f, aperiodic_residual(f, psd, off, exp_))
    assert abs(apf - 10) < 0.5 and height > 0.3


def test_behavioural_rt_pairs_first_response_and_counts_misses():
    dots = {"happy": np.array([1.0, 3.0, 5.0]), "sad": np.array([7.0, 9.0])}
    resp = np.array([1.3, 3.4, 7.25, 12.0])        # el tercer punto (5.0) no tiene respuesta dentro de 1.5 s
    r = behavioural_rt(dots, resp)
    assert np.isclose(r["beh_rt_median_happy"], 350.0)
    assert np.isclose(r["beh_miss_rate"], 2 / 5)
    assert np.isclose(r["beh_rt_sad_minus_happy"], 250.0 - 350.0)


def test_muscle_index_increases_with_high_frequency_noise():
    rng = np.random.default_rng(0)
    sf, n = 250.0, 250 * 60
    base = rng.normal(0, 10, (2, n))
    t = np.arange(n) / sf
    base[1] += 5 * np.sin(2 * np.pi * 75 * t) + rng.normal(0, 5, n)
    hf, rel = muscle_index(base, sf)
    assert hf[1] > hf[0] and rel[1] > rel[0]


def test_confound_regressor_removes_linear_confound_using_train_only():
    rng = np.random.default_rng(1)
    c = rng.normal(size=(200, 1))
    feat = 3 * c + rng.normal(0, 0.1, (200, 1))
    X = np.c_[feat, c]
    cr = ConfoundRegressor(1).fit(X[:150])
    out = cr.transform(X[150:])
    assert out.shape == (50, 1)
    assert abs(np.corrcoef(out[:, 0], c[150:, 0])[0, 1]) < 0.2
