# Deviations from `docs/preregistration.md`

Each deviation is listed with when it was made and what had been seen at that point.
None was made after seeing any feature distribution or model result.

## D1 — Bad-channel detector replaced by PyPREP (2026-11-04)

- **Seen at that point:** only the preprocessing log (channel counts per subject). No features
  and no model results existed yet.
- **Problem:** the pre-specified rule flagged up to 48 of 129 channels. The flagged channels included
  ones adjacent to the recording reference (13° from Cz) with *negative* median correlation with their
  neighbours. Of 441 flags, 414 came from the neighbour-correlation criterion.
  - This is a known failure mode. On reference-dominated data with large artefacts, whole-recording
    correlation is not a valid quality measure.
  - Under that rule, three HC participants would have been excluded for having more than 20% bad
    channels, because of a detector artefact.
- **Change:** bad channels are detected with PyPREP `NoisyChannels`, the PREP pipeline of
  Bigdely-Shamlo et al. (2015), with RANSAC off. It flags channels that are NaN/flat or bad by
  deviation, HF noise, correlation, SNR, dropout or PSD, all with default thresholds.
  E129/Cz, the recording reference, is exempt because it is flat by construction.
- **Unchanged:** the exclusion threshold (more than 20% bad channels) and every other step.

## D2 — Re-run of 14 participants (infrastructure, not a methodological change)

- **Problem:** in the first run, 14 participants failed because PyTorch was being installed while
  ICLabel imported it (`PermissionError` on `shm.dll`).
- **Change:** all 53 participants were re-run after D1.

## D3 — Extra Riemannian variant (added before modelling)

- **Change:** `ERPCovariances` → tangent space → LR (pyRiemann) is added as a secondary trial-level model.
  Its class prototypes are estimated inside each training fold.
- **Why:** it is the standard Riemannian pipeline for ERP data and complements the plain-covariance models.
  It is reported as secondary and exploratory.

## D4 — Sensitivity analysis with all 53 participants (added before any model was run)

- **Seen at that point:** the preprocessing log only.
- **What happened:** the pre-specified criteria excluded 8 of 53 participants: 6 of 24 MDD and 2 of 29 HC.
  In 7 of the 8 cases the reason was fewer than 240 retained trials.
  - The exclusion rate differs by group, so the exclusion itself is a potential selection bias.
- **Change:** the primary model is also re-run on all 53 participants (all features, the same nested
  protocol) as a secondary sensitivity analysis.
- **Unchanged:** the primary analysis still uses the 45 participants who meet the criteria.

## D5 — Riemannian models: precomputed per-trial covariances and fixed C for ERPCov (2026-11-04, before any Riemannian result was seen)

- **Reason:** computational cost. Timing one outer fold showed TS-LR at ~210 s/fold (80 % spent recomputing per-trial OAS covariances in each of the 26 grid fits); with 10 repeats + 200 permutations (1050 outer folds) the three Riemannian models were projected at >12 h.
- **Change 1 (no effect on results):** TS-LR and MDM receive per-trial OAS covariances computed once. Each covariance depends only on its own trial (no learning across samples), so this is leakage-free; the tangent-space reference mean, scaling, LR and MDM centroids remain fitted inside each training fold. A unit test (`test_precomputed_cov_equivalent`) verifies identical predictions.
- **Change 2:** ERPCov-TS-LR uses fixed C = 0.1 (no inner grid), because its ERP prototypes are label-dependent and must be recomputed in-fold. C is fixed a priori, so this cannot inflate performance; at most it is slightly sub-optimal.
- **Change 3:** per-worker RAM budget for trial-level models lowered from 1.2 to 0.75 GB (measured 0.65 GB), allowing more parallel workers.

## D6 — Post-hoc audit analyses (2026-11-05, AFTER all pre-specified results were known)

- **Seen at that point:** every pre-specified result (primary AUC 0.53, demographics 0.78, all secondary models,
  confounds, interpretation). This deviation is therefore **post hoc** and everything it adds is **exploratory**.
- **Reason:** an internal technical audit (`reports/auditoria_paper.md`) found that some questions a reviewer would
  ask were not answered by the pre-specified plan:
  - incremental value of EEG over demographics;
  - education as an EEG confound;
  - minimum detectable effect;
  - residual muscle activity as a confound;
  - quality and exclusions on all 53 participants;
  - theory-driven feature contrasts and aperiodic spectra;
  - task behaviour;
  - subject-level Riemannian representation;
  - subject identifiability;
  - learning curve;
  - permutation test of the raw-amplitude RF control.
- **Unchanged:** the primary model, primary outcome, exclusion criteria, folds and every pre-specified number.
  Nothing pre-specified is replaced or re-ranked.
- **Not possible:** the EGI simple-binary `.raw` files contain event onsets only (`*fix`, `*cue`, `*isi`, `*dot`,
  `*wrp`, `resp`, `TRSP`). The TRSP keys (congruency, side of the emotional face, accuracy) are not stored. A
  dot-probe attentional-bias index, congruent-vs-incongruent probe ERPs and an exact replication of Li et al. (2018)
  are therefore impossible with this release. Reaction time (`*dot` → `resp`) is available.
- **Multiplicity:** the new exploratory classification models are Holm-corrected among themselves and reported
  separately from the pre-specified ones.
- **Raw-amplitude RF permutation budget:** one full subject-wise run takes 159 s (measured), so 100 permutations
  would take about 4.4 h. 40 permutations were run instead, giving a minimum attainable p of 1/41 ≈ 0.024. This was
  decided from the timing alone, before any permuted AUC was seen.
- **Sample-size extrapolation** (`scripts/16_sample_size.py`, added after the audit results were known): this is an
  analytic, optimistic lower bound. It uses an ideal binormal score with the class ratio 18/45, a one-sided α = 0.05
  and the exact Mann–Whitney null. It makes no claim about the data.
- **Raw-RF Holm family:** the RF permutation p is Holm-corrected together with the 7 exploratory models of the audit
  table, a family of 8.
