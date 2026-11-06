# Pipeline overview

The analysis is a chain of numbered scripts in `scripts/` that call a tested package in `src/eeg_mdd/`. Every
parameter lives in `configs/config.yaml`. Each step reads only the outputs of earlier steps.

```{mermaid}
flowchart LR
  A[Raw EGI .raw<br/>+ clinical spreadsheet] --> B[01-03 Inventory,<br/>audit, exact ID match]
  B --> C[04 Per-participant<br/>preprocessing + exclusion]
  C --> D[06 Features<br/>288 per participant]
  C --> E[06 Trial-level data<br/>regional / full epochs]
  D --> F[07 Classic + Riemannian<br/>nested subject-wise CV]
  E --> F
  E --> G[08 Deep learning]
  F --> H[09 Confounds<br/>10 Interpretation]
  G --> H
  H --> I[11 Figures, tables,<br/>LaTeX macros]
  I --> J[12-16 Post-hoc audit]
  J --> K[Paper + this book]
```

| Step | Script | What it does |
|---|---|---|
| 1 | `01_inventory.py` | Lists the raw files without modifying anything |
| 2 | `02_audit.py` | Per-recording audit: channels, sampling rate, EGI events by name, flat channels |
| 3 | `03_match_ids.py` | Exact match of EEG files to the clinical spreadsheet, with no fuzzy matching |
| 3b | `03b_demographics.py` | Demographic table and recording-period confounds |
| 4 | `04_preprocess.py` | PREP bad channels, ICA + ICLabel, filtering, epochs, pre-specified exclusion |
| 5 | `05_compare_old.py` | Comparison with an earlier, flawed version of the dataset |
| 6 | `06_features.py` | 288 subject-level features and trial-level arrays |
| 7 | `07_models.py` | Demographic, handcrafted-feature and Riemannian models with nested CV and permutations |
| 8 | `08_deep.py` | EEGNet, ShallowFBCSPNet and TCN (braindecode, GPU) |
| 9 | `09_confounds.py` | Confound targets, sensitivity analyses and the inflation control |
| 10 | `10_interpret.py` | Permutation importance, SHAP, univariate and cluster tests |
| 11 | `11_figures_tables.py` | Figures, tables and the paper's LaTeX macros |
| 12–16 | `12_…`–`16_…` | Post-hoc audit (exploratory, deviation D6) |
| 17 | `17_book_assets.py` | Aggregated figures and tables for this book |

See [Reproducing the analysis](../guide/reproduce.md) for commands and run times, and [Scripts](../code/scripts.md)
for the source of each step.
