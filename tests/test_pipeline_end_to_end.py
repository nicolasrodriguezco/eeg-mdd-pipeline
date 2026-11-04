"""Corre los scripts 01 a 04 sobre un mini dataset sintético en una carpeta temporal."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from conftest import CODES, make_raw

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _run(script: str, config: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--config", str(config)],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def test_scripts_01_to_04(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    for i, sid in enumerate(["02010001", "02010002", "02010003"]):
        make_raw(seed=i, n_events_per_code=4).save(raw_dir / f"{sid}_raw.fif", verbose="error")
    # El sujeto 02010003 no está en el archivo clínico: debe quedar fuera.
    pd.DataFrame({"subject_id": ["02010001", "02010002"], "type": ["MDD", "HC"]}).to_csv(
        raw_dir / "clinical.csv", index=False
    )
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "manual_id_map.csv").write_text("eeg_id,clinical_id,justification\n")

    cfg = {
        "paths": {
            "raw_dir": str(raw_dir),
            "interim_dir": str(tmp_path / "data" / "interim"),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "reports_dir": str(tmp_path / "reports"),
            "manual_id_map": str(tmp_path / "configs" / "manual_id_map.csv"),
            "ica_exclusions": str(tmp_path / "configs" / "ica_exclusions.csv"),
        },
        "dataset": {
            "eeg_extensions": [".fif"],
            "subject_id_regex": r"(\d{8})",
            "expected_n_eeg_channels": 128,
            "expected_sfreq": 500,
            "clinical": {"file": str(raw_dir / "clinical.csv"), "id_column": "subject_id",
                         "group_column": "type", "group_map": {"MDD": "MDD", "HC": "HC"}},
        },
        "preprocessing": {
            "l_freq": 0.5, "h_freq": 40.0, "notch_freqs": [50.0], "reference": "average",
            "resample_sfreq": 250, "tmin": -0.2, "tmax": 0.6, "baseline": [-0.2, 0.0],
            "reject_peak_to_peak_uv": 100, "event_id": dict(CODES), "stim_channel": None,
            "ica": {"enabled": False},
        },
    }
    config = tmp_path / "configs" / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))

    _run("01_inventory.py", config)
    out_audit = _run("02_audit.py", config)
    assert "Problemas detectados: 0" in out_audit
    out_match = _run("03_match_ids.py", config)
    assert "Sujetos usables: 2" in out_match
    _run("04_preprocess.py", config)

    log = pd.read_csv(tmp_path / "reports" / "preprocessing_log.csv", dtype={"subject_id": str})
    assert set(log["subject_id"]) == {"02010001", "02010002"}
    assert log["error"].map(lambda v: pd.isna(v) or v == "").all()
    assert (log["n_trials_kept_total"] == 12).all()
    assert (tmp_path / "data" / "processed" / "epochs" / "sub-02010001-epo.fif").exists()
