import pandas as pd
import pytest

from eeg_mdd.matching import load_manual_map, match_subjects, normalize_id, usable_subjects

GROUP_MAP = {"MDD": "MDD", "HC": "HC"}


def _clinical(rows):
    df = pd.DataFrame(rows, columns=["clinical_id", "group"])
    return df


def test_normalize_id_only_strips():
    assert normalize_id(" 02010001 ") == "02010001"
    assert normalize_id("2010001.0") == "2010001"  # artefacto típico de Excel
    assert normalize_id("02010001") != normalize_id("2010001")  # NO se quitan ceros


def test_match_statuses():
    clinical = _clinical([("A1", "MDD"), ("A2", "HC"), ("A4", "HC"), ("A5", None), ("A6", "MDD"), ("A6", "HC")])
    eeg_ids = ["A1", "A2", "A3", "A5", "A6"]
    m = match_subjects(eeg_ids, clinical).set_index("eeg_id")

    assert m.loc["A1", "status"] == "matched" and m.loc["A1", "group"] == "MDD"
    assert m.loc["A2", "status"] == "matched" and m.loc["A2", "group"] == "HC"
    assert m.loc["A3", "status"] == "only_eeg"
    assert m.loc["A5", "status"] == "unlabeled"
    assert m.loc["A6", "status"] == "duplicated"  # dos etiquetas para el mismo ID
    only_clin = match_subjects(eeg_ids, clinical)
    assert "A4" in only_clin.loc[only_clin["status"] == "only_clinical", "clinical_id"].tolist()


def test_no_fuzzy_matching():
    clinical = _clinical([("2010001", "MDD")])
    m = match_subjects(["02010001"], clinical)
    assert m.loc[m["eeg_id"] == "02010001", "status"].item() == "only_eeg"


def test_manual_map_requires_justification(tmp_path):
    path = tmp_path / "manual.csv"
    path.write_text("eeg_id,clinical_id,justification\n02010001,2010001,\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_manual_map(path)

    path.write_text("eeg_id,clinical_id,justification\n02010001,2010001,Verificado con la hoja de registro del día 3\n",
                    encoding="utf-8")
    mapping = load_manual_map(path)
    clinical = _clinical([("2010001", "MDD")])
    m = match_subjects(["02010001"], clinical, mapping)
    row = m[m["eeg_id"] == "02010001"].iloc[0]
    assert row["status"] == "matched_manual" and row["group"] == "MDD"
    assert len(usable_subjects(m)) == 1
