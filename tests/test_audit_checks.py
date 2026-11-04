import json

import pandas as pd

from eeg_mdd.audit import audit_file
from eeg_mdd.checks import check_audit, check_labels, check_no_subject_overlap

from conftest import make_raw

REGEX = r"(\d{8})"


def test_audit_file_reports_basics(tmp_path):
    raw = make_raw(n_events_per_code=5)
    raw._data[0] = 0.0  # canal E1 plano
    path = tmp_path / "02010001_raw.fif"
    raw.save(path, verbose="error")

    row = audit_file(path, REGEX)
    assert row["error"] == ""
    assert row["subject_id"] == "02010001"
    assert row["n_eeg_channels"] == 128
    assert row["sfreq"] == 500.0
    assert row["n_events"] == 15
    assert json.loads(row["event_counts"]) == {"11": 5, "12": 5, "13": 5}
    assert row["n_flat_channels"] == 1 and "E1" in row["flat_channels_first60s"]


def test_audit_file_records_errors_instead_of_skipping(tmp_path):
    bad = tmp_path / "02010009_raw.fif"
    bad.write_text("no es un fif")
    row = audit_file(bad, REGEX)
    assert row["error"]  # se registra, no desaparece


def test_check_audit_flags_problems():
    audit = pd.DataFrame(
        [
            {"file": "a", "subject_id": "1", "error": "", "n_eeg_channels": 128, "sfreq": 250.0,
             "n_events": 30, "n_flat_channels": 0, "flat_channels_first60s": ""},
            {"file": "b", "subject_id": "2", "error": "", "n_eeg_channels": 64, "sfreq": 500.0,
             "n_events": 0, "n_flat_channels": 2, "flat_channels_first60s": "E1,E2"},
            {"file": "c", "subject_id": "2", "error": "", "n_eeg_channels": 128, "sfreq": 250.0,
             "n_events": 30, "n_flat_channels": 0, "flat_channels_first60s": ""},
        ]
    )
    problems = " | ".join(check_audit(audit, 128, 250))
    assert "64 canales" in problems
    assert "500.0 Hz" in problems
    assert "sin eventos" in problems
    assert "canales planos" in problems
    assert "más de un archivo" in problems


def test_check_labels_and_overlap():
    ok = pd.DataFrame({"eeg_id": ["1", "2"], "group": ["MDD", "HC"]})
    assert check_labels(ok) == []
    bad = pd.DataFrame({"eeg_id": ["1", "1", "3"], "group": ["MDD", "MDD", None]})
    assert len(check_labels(bad)) == 2
    assert check_no_subject_overlap(["1", "2"], ["3"]) == []
    assert check_no_subject_overlap(["1", "2"], ["2", "3"])
