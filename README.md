# eeg-mdd-pipeline

Subject-independent classification of **major depressive disorder (MDD) vs. healthy controls (HC)**
from 128-channel task EEG (MODMA, EGI HydroCel GSN, emotional-face dot-probe task, Lanzhou 2015).
The protocol is fully reproducible and leakage-free. **This is research code, not a clinical tool.**

Author: Nicolas Rodriguez (Data Science Program, Division of Basic Sciences, Universidad del Norte) ·
[ORCID](https://orcid.org/0009-0005-3286-0704) · [LinkedIn](https://www.linkedin.com/in/nicolasrodriguez04/)

- Paper (LaTeX, English): `docs/paper/main.tex` → `docs/paper/main.pdf`
- Analysis plan, fixed before any result: `docs/preregistration.md`. Deviations: `reports/deviations.md`
- Data card / model card: `docs/data_card.md`, `docs/model_card.md`
- Companion Jupyter Book (code, methods, results; no data): `book/`. Build it with `jupyter-book build book`
  (see `book/guide/reproduce.md`) and publish it with `ghp-import -n -p -f book/_build/html`.
  Online: https://nicolasrodriguezco.github.io/eeg-mdd-pipeline

## Installation

```bash
conda env create -f environment.yml        # Python 3.11, MNE, PyPREP, mne-icalabel, pyRiemann, braindecode...
conda activate eeg-mdd
pip install -e .
# GPU (optional, needed for practical deep-learning run times; RTX 50xx needs CUDA 12.8 wheels):
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Data are **not** distributed. Put the MODMA ERP files in `data/raw/` and leave them read-only:
- `EEG_128channels_ERP_lanzhou_2015/*.raw`
- `subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx`

## Reproduce everything (end to end)

| Step | Command | Output | Time* |
|---|---|---|---|
| 1. Inventory | `python scripts/01_inventory.py` | `data/interim/inventory.csv` | s |
| 2. Audit (EGI events by name) | `python scripts/02_audit.py` | `data/interim/audit.csv` | 2 min |
| 3. ID matching (exact, no fuzzy) | `python scripts/03_match_ids.py` | `data/interim/subjects.csv` | s |
| 3b. Demographics / recording dates | `python scripts/03b_demographics.py` | `reports/tables/` | 1 min |
| 4. Preprocessing + exclusion | `python scripts/04_preprocess.py --n-jobs 10` | epochs, segments, ICA, `reports/qc/*.html` | 15 min |
| 6. Features | `python scripts/06_features.py` | `data/processed/features/` | 1 min |
| 7. Classic + Riemannian models | `python scripts/07_models.py` | `reports/results/models/` | hours (permutations) |
| 7b. Sensitivity, all 53 | `python scripts/07_models.py --all-subjects --only primary` | `reports/results/models_all53/` | min |
| 8. Deep learning (GPU) | `python scripts/08_deep.py --repeats 3` | `reports/results/models/DL-*` | hours |
| 9. Confounds | `python scripts/09_confounds.py` | `reports/results/confounds/` | min |
| 10. Interpretation | `python scripts/10_interpret.py` | `reports/results/interpret/`, figures | min |
| 11. Figures, tables, numbers | `python scripts/11_figures_tables.py` | `reports/results/{figures,tables}`, `docs/paper/generated/numbers.tex` | s |
| 12. Paper | `cd docs/paper && ../../tools/tectonic/tectonic.exe main.tex` | `docs/paper/main.pdf` | s |

Tectonic: the conda-forge build fails on Windows with a DLL entry-point error, so the standalone release is used
(download `tectonic-0.15.0-x86_64-pc-windows-msvc.zip` from the tectonic GitHub releases into `tools/tectonic/`;
the folder is git-ignored). The first compilation downloads the required TeX packages.

*32-core CPU and RTX 5070 Laptop GPU.

Every number in the paper is a LaTeX macro written by step 11 from the saved outputs. No number is typed by hand.

## Guarantees enforced in code

- **Labels** come only from the clinical spreadsheet, by exact ID match. Fuzzy matches go to `configs/manual_id_map.csv`, which needs a written justification.
- **Events** are read by EGI channel name (`fcue`, `hcue`, `scue`) with fixed codes. The combined `STI 014` channel is 1 for every event in these files.
- **Preprocessing** is per participant, so nothing is fitted across participants:
  - PyPREP bad-channel detection, then ICA (Picard) with ICLabel-based component rejection.
  - 0.5–40 Hz band-pass, 50 Hz notch, spline interpolation, average reference.
  - Epochs from −200 to 600 ms, rejected if any channel exceeds ±100 µV.
- **Evaluation** is always subject-wise:
  - 5-fold × 10-repeat outer CV on subjects, with nested inner CV on subjects.
  - A `SubjectLeakError` assertion runs on every outer, inner and validation split.
  - All models share the same subject partitions, so they can be compared pairwise.
- **Prohibited predictors:** demographics and questionnaires are never EEG predictors. Demographics appear only in the demographic baseline and the confound analyses.
- **Reporting:** results are per subject, with 95% bootstrap CIs over subjects and label-permutation p-values that re-run the full nested pipeline.

## Tests

```bash
pytest -q          # synthetic EEG only; no real data needed
```

## Layout

```
configs/      config.yaml (all parameters), manual_id_map.csv, ica_exclusions.csv (ICLabel decisions)
docs/         preregistration, review, data/model cards, paper/
scripts/      numbered pipeline steps
src/eeg_mdd/  io, audit, events, matching, demographics, preprocess, features/, evaluation/, models/, plotting
tests/        pytest suite (synthetic data)
reports/      phase reports, logs, QC reports and results (not versioned: they contain participant data)
```
