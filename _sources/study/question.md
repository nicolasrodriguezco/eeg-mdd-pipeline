# The question

EEG is cheap, portable and widely available, which makes it an attractive candidate for objective markers of major
depressive disorder (MDD). Many machine-learning studies report accuracies above 90% for separating patients from
healthy controls. A large share of them, however, split **trials or segments** rather than **participants** between
training and test sets.

When one person contributes data to both sets, a classifier can succeed by recognising the *person* (head geometry,
impedances, idiosyncratic rhythms, recording session) instead of the *disorder*. Every person carries one label, so
recognising the person is enough. The resulting estimate says nothing about the intended use, which is diagnosing a
new patient.

Small samples make this worse. With a few dozen participants, cross-validated accuracy has error bars of ±10
percentage points or more. Demographic and acquisition differences between groups can also masquerade as a signal.

This study therefore asks a single question:

> **How well can task EEG classify MDD versus healthy controls in participants the model has never seen?**

## What the reader gains

- **A reference benchmark.** Honest, participant-level estimates with fixed folds, against which future MODMA results
  can be compared.
- **A mechanistic explanation of inflated accuracies.** The size of the inflation is measured on the same data, and
  so is its cause: single trials identify the participant almost perfectly.
- **Concrete confound checks.** A demographic baseline, a test of whether EEG adds to it, and a raw-data muscle
  (EMG) index.
- **Design guidance.** The effect sizes this sample can detect, the sample sizes smaller effects would need, and a
  [reporting checklist](../guide/checklist.md).
- **A concrete hypothesis.** The whole ERP waveform, rather than fixed peaks, may carry a weak signal worth testing
  in independent data.

## How the study protects itself against fooling itself

1. The analysis plan was committed to version control **before any result existed**
   ([pre-registration](../guide/preregistration.md)). Every later change is logged with what had been seen at that
   point ([deviations](../guide/deviations.md)).
2. Every evaluation splits by participant, and a code assertion verifies it in every fold.
3. Everything that learns from data (scaling, selection, hyperparameters) happens inside the training fold.
4. The primary model and outcome were fixed in advance. Secondary models are corrected for multiplicity, and the
   selection of the best model is itself evaluated by nested cross-validation.
5. Every number in the paper is a LaTeX macro written by code from saved outputs. None is typed by hand.
