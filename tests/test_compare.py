import pandas as pd

from eeg_mdd.compare import compare_old_new, counts_from_old


def test_label_mismatch_and_missing_subjects_are_flagged():
    old = pd.DataFrame(
        {
            "Subject": ["1"] * 3 + ["2"] * 2 + ["9"],
            "Group": ["MDD"] * 3 + ["MDD"] * 2 + ["HC"],
            "Condition": ["Fear", "Sad", "Happy", "Fear", "Sad", "Fear"],
        }
    )
    new_log = pd.DataFrame({"subject_id": ["1", "2", "3"], "n_trials_kept_total": [3, 2, 5]})
    subjects = pd.DataFrame({"eeg_id": ["1", "2", "3"], "group": ["MDD", "HC", "HC"]})

    result = compare_old_new(counts_from_old(old), new_log, subjects).set_index("subject_id")
    assert result.loc["1", "issues"] == ""
    assert "ETIQUETA DISTINTA" in result.loc["2", "issues"]
    assert result.loc["3", "issues"] == "solo en el nuevo"
    assert result.loc["9", "issues"] == "solo en el viejo"
