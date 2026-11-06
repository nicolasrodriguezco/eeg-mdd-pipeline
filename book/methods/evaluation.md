# Evaluation protocol

This is the core of the study. Every design choice serves one goal: **the estimate must describe performance on a
new person**.

| Element | Choice |
|---|---|
| Outer loop | Stratified 5-fold CV over **participants**, repeated 10 times (3 for deep learning) |
| Partitions | Identical for all models, so models can be compared pairwise |
| Inner loop | 5-fold participant-grouped CV inside the training participants, maximising AUC |
| Leakage check | An assertion in every outer, inner and validation split |
| Unit of analysis | The participant. Trial-level probabilities are averaged per participant |
| Primary outcome | Participant-level ROC AUC, fixed in advance |
| Other metrics | Balanced accuracy, sensitivity and specificity at a fixed 0.5 threshold (never tuned), F1, Brier |
| Confidence intervals | 2,000 stratified bootstrap resamples of participants on out-of-fold predictions |
| Significance | Labels permuted *between participants* and the full nested pipeline re-run (1,000 or 200 permutations) |
| Model comparison | Nadeau–Bengio corrected resampled $t$-test; Holm correction across secondary models |

The permutation $p$-value is

$$
p = \frac{1 + \#\{\mathrm{AUC}_\text{perm} \ge \mathrm{AUC}_\text{obs}\}}{1 + n_\text{perm}}.
$$

## The leakage guard

The assertion below runs in every split. If any participant appears on both sides, the run stops.

```{literalinclude} ../../src/eeg_mdd/evaluation/cv.py
:language: python
:pyobject: assert_disjoint
```

```{literalinclude} ../../src/eeg_mdd/evaluation/cv.py
:language: python
:pyobject: outer_splits
```

## Nested cross-validation

```{literalinclude} ../../src/eeg_mdd/evaluation/nested.py
:language: python
:pyobject: nested_cv
```

```{literalinclude} ../../src/eeg_mdd/evaluation/nested.py
:language: python
:pyobject: permutation_test
```

## Participant-level metrics and bootstrap

```{literalinclude} ../../src/eeg_mdd/evaluation/metrics.py
:language: python
:pyobject: bootstrap_ci
```
