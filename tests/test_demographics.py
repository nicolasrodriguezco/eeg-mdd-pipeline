import numpy as np
import pandas as pd

from eeg_mdd.demographics import demographic_table, recording_datetime_from_name


def test_demographic_table_detects_age_difference_and_counts_sex():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "group": ["MDD"] * 20 + ["HC"] * 20,
        "age": np.r_[rng.normal(40, 3, 20), rng.normal(25, 3, 20)].round(),
        "gender": ["F"] * 12 + ["M"] * 8 + ["F"] * 10 + ["M"] * 10,
    })
    tab = demographic_table(df, ["age"], ["gender"]).set_index("variable")
    assert tab.loc["age", "p_value"] < 0.001
    assert tab.loc["age", "MDD_n"] == 20
    assert tab.loc["gender", "test"] == "Fisher exact"
    assert tab.loc["gender", "MDD_counts"] == "F=12, M=8"
    assert tab.loc["gender", "p_value"] > 0.05


def test_recording_datetime_from_name():
    d, t = recording_datetime_from_name("02010002erp 20150416 1131.raw")
    assert d == pd.Timestamp("2015-04-16") and t == "1131"
    d, t = recording_datetime_from_name("02030002erp_new 20151022 14.raw")
    assert d == pd.Timestamp("2015-10-22") and t == "14"   # hora truncada: no se inventa
    assert recording_datetime_from_name("sin_fecha.raw") == (None, "")
