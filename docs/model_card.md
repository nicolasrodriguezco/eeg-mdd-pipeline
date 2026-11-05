# Model card: primary MDD-vs-HC classifier (MODMA 128-channel dot-probe ERP)

> **Not a clinical tool.** This model is a research artefact from a methodological study. It must not be used to
> diagnose, screen or triage anyone.

## Model details
- **Name:** `PRIMARY_LR-L2_all`. The model was fixed a priori in `docs/preregistration.md`.
- **Type:** scikit-learn pipeline with three steps:
  1. median imputation;
  2. standardisation;
  3. L2-regularised logistic regression (class-balanced). C is tuned in an inner 5-fold participant-grouped CV
     over `logspace(-3, 1, 9)`.
- **Input:** 288 subject-level EEG features (see `src/eeg_mdd/features/`):
  - spectral: 100;
  - frontal alpha asymmetry: 2;
  - ERP (N170, P300, LPP): 21;
  - wPLI connectivity: 135;
  - complexity: 30.
- **Output:** probability of MDD per participant.
- **Code:** `scripts/07_models.py`, `src/eeg_mdd/models/__init__.py`.

## Intended use
To estimate, as honestly as possible, how much diagnostic information task EEG carries in this dataset.

**Out of scope:** any clinical or individual-level decision.

## Training and evaluation data
- **Data:** MODMA 128-channel ERP subset (Lanzhou, 2015–2016).
  - 53 recordings: 24 MDD, 29 HC.
  - 45 met the pre-specified quality criteria: 18 MDD, 27 HC.
- **Labels:** taken only from the official subject-information spreadsheet, matched by exact identifier.
- **Excluded from predictors:** demographic variables and questionnaires. They are used only for the baseline and for
  confound analyses.

## Evaluation
- **Protocol:**
  - outer loop: stratified 5-fold CV over participants, repeated 10 times, with identical folds for all models;
  - inner loop: nested CV for hyperparameters;
  - an assertion in every split guarantees no participant appears on both sides.
- **Metrics:** participant-level AUC (the primary outcome), plus:
  - balanced accuracy, sensitivity and specificity at 0.5;
  - F1;
  - Brier score.
- **Uncertainty:**
  - 2,000-resample participant bootstrap;
  - label-permutation test re-running the full nested pipeline (1,000 permutations).
- **Results:** see `reports/results/tables/main_results.csv` and the paper (`docs/paper/main.pdf`). In short:
  - the primary model performed at chance;
  - no EEG model outperformed a demographic baseline (age, sex, education).

## Ethical considerations and caveats
- **Small sample:** confidence intervals span roughly ±0.17 AUC.
- **Limited scope:** single site, single task, no medication information.
- **Group imbalances:**
  - the groups differ in education;
  - they are partly confounded with recording period;
  - exclusions were more frequent in MDD.

  Each of these is analysed in the paper.
- **Data handling:** participant data are not redistributed and were never sent to external services.
