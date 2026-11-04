from pathlib import Path

from eeg_mdd.inventory import build_inventory, summarize_inventory
from eeg_mdd.io import extract_subject_id, find_eeg_files

REGEX = r"(\d{8})"
EXTS = [".mff", ".raw", ".fif", ".mat"]


def _touch(path: Path, size: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0" * size)


def test_extract_subject_id():
    assert extract_subject_id("sub_02010002_task.raw", REGEX) == "02010002"
    assert extract_subject_id("sin_numero.raw", REGEX) is None  # nunca se adivina


def test_real_modma_file_names():
    regex = r"^(\d{8})"
    assert extract_subject_id("02020019_erp 20150703 1052.raw", regex) == "02020019"
    assert extract_subject_id("02030002erp_new 20151022 14.raw", regex) == "02030002"
    assert extract_subject_id("02030006 20151103 1801.raw", regex) == "02030006"
    assert extract_subject_id("erp 20150703 1052.raw", regex) is None  # no toma la fecha


def test_inventory_flags_rerecordings(tmp_path):
    _touch(tmp_path / "02020023_erpnew 20150709 110.raw")
    _touch(tmp_path / "02020019_erp 20150703 1052.raw")
    inv = build_inventory(tmp_path, r"^(\d{8})", [".raw"]).set_index("subject_id_guess")
    assert inv.loc["02020023", "possible_rerecording"]
    assert not inv.loc["02020019", "possible_rerecording"]
    assert inv.loc["02020019", "name_rest"].startswith("erp")


def test_find_eeg_files_counts_mff_folder_once(tmp_path):
    _touch(tmp_path / "a" / "02010001.raw")
    _touch(tmp_path / "b" / "02010002.mff" / "signal1.bin")
    _touch(tmp_path / "b" / "02010002.mff" / "info.xml")
    _touch(tmp_path / "clinical.xlsx")
    found = find_eeg_files(tmp_path, EXTS)
    names = sorted(p.name for p in found)
    assert names == ["02010001.raw", "02010002.mff"]


def test_inventory_and_summary(tmp_path):
    _touch(tmp_path / "02010001.raw", 2_000_000)
    _touch(tmp_path / "notes.txt", 100)
    inv = build_inventory(tmp_path, REGEX, EXTS)
    assert set(inv["relative_path"]) == {"02010001.raw", "notes.txt"}
    row = inv.set_index("relative_path").loc["02010001.raw"]
    assert row["is_eeg"] and row["subject_id_guess"] == "02010001"
    summary = summarize_inventory(inv)
    assert summary["n_files"].sum() == 2
