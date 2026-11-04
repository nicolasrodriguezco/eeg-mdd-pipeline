# Analysis plan, fixed before preprocessing and modelling

Date fixed: 2026-11-04. This file is committed to git **before** any epochs, features or model
results exist (see `git log -- docs/preregistration.md`). Any later deviation is listed in
`reports/deviations.md` with its reason.

## Data
- MODMA, 128-channel HydroCel GSN (EGI) ERP dot-probe task, Lanzhou 2015 recordings.
- 53 recordings, one per participant: 24 MDD, 29 HC. Labels come only from
  `subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx`, column `type`.
- The three files marked "new" or "(n)" (02010008, 02020023, 02030002) are the only recording
  for their participant and contain the full task. They are included.
- Task: fixation → face pair (emotional + neutral, 516 ms) → ISI (116–316 ms) → dot probe.
  Three blocks of 160 trials (fear, happy, sad).

## Preprocessing (per participant; nothing is fitted across participants)
1. Read the `.raw` file. Rename E129 (the recording reference) to `Cz` and set the `GSN-HydroCel-129` montage.
2. Events are read by EGI channel name with fixed codes: `fcue`→11 (fear), `hcue`→12 (happy),
   `scue`→13 (sad). Epochs are locked to face-pair onset.
3. Bad channels are detected automatically on a 1–40 Hz copy. A channel is bad if it meets any of these:
   - It is flat (SD < 0.1 µV).
   - Its robust z-score of log-SD is above 5 in absolute value.
   - Its median correlation with its 6 nearest neighbours is below 0.4.

   E129/Cz is the recording reference, so it is exempt from the flat check.
4. ICA is fitted on a copy filtered 1–100 Hz, notched at 50 Hz and average-referenced, with bad channels
   excluded and the data decimated by 2. It uses Picard (extended infomax), 30 components and seed 97.
   Components are labelled with ICLabel. A component is removed when its top label is eye blink,
   muscle artifact, heart beat, line noise or channel noise **and** that label's probability is at least 0.80.
   Decisions are written to `configs/ica_exclusions.csv`.
5. Analysis copy: 0.5–40 Hz zero-phase FIR band-pass, 50 Hz notch, ICA applied, bad channels
   interpolated (spherical spline), average reference.
6. Epochs run from −200 to 600 ms with baseline −200 to 0 ms. An epoch is rejected if any channel exceeds ±100 µV
   after baseline correction.
7. A cleaned continuous copy (125 Hz) is also saved for spectral, connectivity and complexity features.
   It is cut into 2-s segments, and segments with any |x| > 100 µV are dropped.

## Participant exclusion criteria (fixed now, applied mechanically)
A participant is excluded if any of these holds:
- (a) more than 20% of channels are bad (more than 25 of 129);
- (b) fewer than 50% of trials remain (fewer than 240 of 480);
- (c) fewer than 40 trials remain in any condition.

Excluded participants are reported, and nothing else is changed afterwards.

## Features (subject level, computed per participant without using labels)
- **Regions:** 10 regions (L/R × frontal, central, temporal, parietal, occipital), defined from
  montage coordinates. Midline electrodes are assigned to neither hemisphere.
- **Spectral:** Welch power on 2-s segments for delta 1–4, theta 4–8, alpha 8–13, beta 13–30 and
  low gamma 30–40 Hz. Both log absolute and relative power are computed, per region.
- **Frontal alpha asymmetry:** ln α(E124 ≈ F4) − ln α(E24 ≈ F3), plus the F8/F7 pair (E122/E33).
- **ERP per condition**, on per-condition averages:
  - N170: minimum at E58/E96 (≈P7/P8), 140–200 ms; amplitude and latency.
  - P300: mean amplitude 250–500 ms and peak latency at a centro-parietal cluster (Pz ≈ E62 plus neighbours).
  - LPP: mean amplitude 400–600 ms, same cluster.
- **Connectivity:** wPLI between the 10 region-mean signals in the theta, alpha and beta bands.
- **Complexity:** per region, mean over segments of sample entropy, Higuchi fractal dimension and
  Lempel-Ziv complexity.

## Evaluation (all subject-wise)
- **Outer loop:** `StratifiedGroupKFold`, 5 folds, repeated 10 times (seeds 0–9). Each fold asserts that train
  and test participants are disjoint.
- **Inner loop:** hyperparameters are tuned with 5-fold subject-grouped CV inside the training participants.
  Scaling, imputation and feature selection all live inside a `sklearn` Pipeline.
- Trial-level models average their trial probabilities to give one probability per participant.
- **Metrics (per participant):** balanced accuracy at a fixed 0.5 threshold, ROC AUC, sensitivity,
  specificity, F1 and Brier score.
  - Each metric is computed per repetition and averaged.
  - 95% CI from 2,000 participant-level bootstrap resamples of the out-of-fold predictions,
    averaged over repetitions.
  - p-value from a label-permutation test: subject labels permuted 1,000 times with the full nested
    pipeline re-run, using 1 repetition for speed. Deep models get fewer permutations, and the number is reported.
- **Primary model, fixed a priori:** L2-regularised logistic regression on the full subject-level
  handcrafted feature set.
- **Primary outcome:** its subject-level ROC AUC. Every other model is secondary.
  - The model with the best mean AUC is reported as such.
  - Because picking the best model inflates its score, a fully nested "select the model family in the
    inner loop" estimate is also reported.
- **Secondary models:**
  - Demographic baseline: age, sex and education, logistic regression.
  - L1-LR, linear SVM, RBF SVM, Random Forest and LightGBM (max depth 3).
  - Riemannian: trial covariances of region signals → tangent space + LR, and MDM.
  - Deep learning (braindecode): EEGNet, ShallowFBCSPNet and TCN. These use early stopping on an inner
    subject-wise validation split.
- **Invalid control:** the best trial-level model re-evaluated with a trial-wise split, only to
  measure inflation. It is always labelled invalid.

## Confounds
- Education differs between groups (p = 0.005), so it is part of the demographic baseline.
- Recording period is partly confounded with group (Oct 2015–Jan 2016: only HC; Mar–Apr 2016: only MDD).
  Two controls:
  - Can EEG features predict the recording period (before vs after Sep 2015)?
  - A sensitivity analysis restricted to participants recorded June–August 2015.
- Further checks:
  - Can EEG features predict sex and age (median split)?
  - Performance broken down by sex.
  - Group differences in data quality: bad channels, ICA components removed and trials retained.
- Questionnaires (PHQ-9, GAD-7, CTQ-SF, LES, SSRS, PSQI) are never used as predictors.

## Interpretation
- Permutation importance computed inside the CV test folds, and SHAP on the primary model.
- Channel-level group effect sizes are shown as topomaps.
- Grand-average ERPs by group and condition, with a spatio-temporal cluster permutation test.
  This is descriptive only.
