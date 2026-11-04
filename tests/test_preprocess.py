import numpy as np
import pytest

from eeg_mdd.preprocess import preprocess_raw


def _power_at(data: np.ndarray, sfreq: float, freq: float) -> float:
    spectrum = np.abs(np.fft.rfft(data, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(data.shape[-1], 1 / sfreq)
    return float(spectrum[:, np.argmin(np.abs(freqs - freq))].mean())


def test_epochs_per_condition_and_resampling(synthetic_raw, pp_config):
    epochs, log = preprocess_raw(synthetic_raw, pp_config, "02010001")
    assert log["n_eeg_channels"] == 128
    assert log["n_events_found"] == 30
    for cond in ("fear", "happy", "sad"):
        assert log[f"n_trials_raw_{cond}"] == 10
        assert log[f"n_trials_kept_{cond}"] == 10  # el ruido sintético no supera 100 µV
    assert epochs.info["sfreq"] == 250
    assert len(epochs.times) == 201  # -0.2 a 0.6 s a 250 Hz


def test_line_noise_is_removed(synthetic_raw, pp_config):
    raw = synthetic_raw.copy().pick("eeg")
    before = _power_at(raw.get_data(), raw.info["sfreq"], 50.0)
    epochs, _ = preprocess_raw(synthetic_raw, pp_config, "x")
    after = _power_at(epochs.get_data().transpose(1, 0, 2).reshape(128, -1), epochs.info["sfreq"], 50.0)
    assert after < before * 0.01


def test_rejection_by_amplitude(synthetic_raw, pp_config):
    raw = synthetic_raw.copy()
    raw._data[:128] *= 50  # señal de ~250 µV: todas las épocas deben rechazarse
    epochs, log = preprocess_raw(raw, pp_config, "x")
    assert log["n_trials_kept_total"] == 0


def test_requires_event_id(synthetic_raw, pp_config):
    pp_config["event_id"] = {}
    with pytest.raises(ValueError, match="event_id"):
        preprocess_raw(synthetic_raw, pp_config, "x")


def test_bad_channel_is_interpolated(synthetic_raw, pp_config):
    raw = synthetic_raw.copy()
    raw.info["bads"] = ["E10"]
    epochs, log = preprocess_raw(raw, pp_config, "x")
    assert log["bads_interpolated"] == "E10"
    assert "E10" in epochs.ch_names and epochs.info["bads"] == []


def test_ica_is_fitted_and_saved(synthetic_raw, pp_config, tmp_path):
    pp_config["ica"] = {"enabled": True, "n_components": 10, "method": "fastica",
                        "random_state": 97, "highpass_for_fit": 1.0}
    epochs, log = preprocess_raw(synthetic_raw, pp_config, "02010001", ica_exclude=[0], ica_dir=tmp_path)
    assert (tmp_path / "sub-02010001-ica.fif").exists()
    assert log["ica_excluded"] == "0"


def test_pyprep_bad_channels_flags_noisy_and_exempts_reference(synthetic_raw):
    from eeg_mdd.preprocess import detect_bad_channels

    raw = synthetic_raw.copy().pick("eeg")
    raw.rename_channels({"E128": "Cz"})
    raw._data[raw.ch_names.index("Cz")] = 0.0                     # referencia de registro: plana
    rng = np.random.default_rng(0)
    raw._data[raw.ch_names.index("E40")] += rng.normal(0, 150e-6, raw.n_times)   # canal muy ruidoso
    bads, reasons = detect_bad_channels(raw, {"method": "pyprep", "exempt_flat": ["Cz"]})
    assert "E40" in bads
    assert "Cz" not in bads
