# Reproducing the analysis

## Installation

```bash
conda env create -f environment.yml        # Python 3.11, MNE, PyPREP, mne-icalabel, pyRiemann, braindecode...
conda activate eeg-mdd
pip install -e .
# GPU (optional; RTX 50xx needs CUDA 12.8 wheels):
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

The exact versions used are pinned in `requirements-lock.txt`.

## Data

The data are **not** distributed. Request the MODMA ERP data from its maintainers and place the files in
`data/raw/`, which the code treats as read-only:

- `EEG_128channels_ERP_lanzhou_2015/*.raw`
- `subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx`

## Run end to end

| Step | Command | Time* |
|---|---|---|
| Inventory | `python scripts/01_inventory.py` | seconds |
| Audit (EGI events by name) | `python scripts/02_audit.py` | 2 min |
| ID matching | `python scripts/03_match_ids.py` | seconds |
| Demographics | `python scripts/03b_demographics.py` | 1 min |
| Preprocessing + exclusion | `python scripts/04_preprocess.py --n-jobs 10` | 15 min |
| Features | `python scripts/06_features.py` | 1 min |
| Classic + Riemannian models | `python scripts/07_models.py` | hours (permutations) |
| Sensitivity, all 53 | `python scripts/07_models.py --all-subjects --only primary` | minutes |
| Deep learning (GPU) | `python scripts/08_deep.py --repeats 3` | hours |
| Confounds | `python scripts/09_confounds.py` | minutes |
| Interpretation | `python scripts/10_interpret.py` | minutes |
| Figures, tables, macros | `python scripts/11_figures_tables.py` | seconds |
| Audit features | `python scripts/12_audit_features.py` | minutes |
| Audit models | `python scripts/13_audit_models.py` | hours |
| Raw RF permutation | `python scripts/14_rawrf_permutation.py --n-perm 40` | 1.3 h |
| Audit macros and figures | `python scripts/15_audit_report.py` | seconds |
| Sample size | `python scripts/16_sample_size.py` | seconds |
| Paper | `cd docs/paper && tectonic main.tex` | seconds |

*Measured on a 32-core CPU with an RTX 5070 Laptop GPU.

## Tests

```bash
pytest -q          # synthetic EEG only; no real data needed
```

The 53 tests use synthetic data only. Among other things, they check that:

- the leakage guard fires on a leaky split;
- every outer fold is subject-disjoint and covers all subjects;
- a model with no signal stays at chance even when subject identity is strong;
- no fuzzy ID matching is possible;
- the confound regressor uses only the training fold;
- features recover known synthetic effects (alpha peaks, ERP peaks, lagged coupling).

## Building this book

```bash
conda create -n eeg-mdd-book python=3.11
conda activate eeg-mdd-book
pip install -r book/requirements.txt
pip install -e . --no-deps
python scripts/17_book_assets.py      # needs the results; skip it to use the committed _generated/ folder
jupyter-book build book
```

The book is built without data, so it can be published as a static site. It is published to GitHub Pages from the
`gh-pages` branch:

```bash
ghp-import -n -p -f book/_build/html
```
