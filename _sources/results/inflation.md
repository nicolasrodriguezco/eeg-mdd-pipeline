# Why trial-wise splits inflate

Two trial-level models were evaluated twice, keeping everything identical except the split:

- with a **trial-wise split** (invalid), where trials of one participant fall in both training and test;
- with a **participant-wise split**.

```{include} ../_generated/inflation.md
```

```{figure} ../_generated/figures/fig_inflation.png
:width: 90%
The same models under a trial-wise and a participant-wise split, at the trial and participant level.
```

## The mechanism

A multinomial classifier on single-trial covariances identified **which of the 45 participants** produced a trial
with 97.9% accuracy, against a chance level of 2.2%. Individual EEG signatures are highly distinctive. Under a
trial-wise split, each test participant has been seen in training, and every participant carries one label. The
model therefore only has to recognise the person to "diagnose" them.

```{important}
Accuracies above 95% on datasets of a few dozen participants should be read with suspicion unless the paper states
that the split is across participants.
```

## The raw-amplitude random forest

The random forest on raw regional amplitudes mimics common practice, including an earlier unpublished analysis of
these data by the author. Split by participant, it reached a participant-level AUC of 0.70, the highest of any EEG
model. A post-hoc permutation test gave $p=0.049$:

- only 40 permutations were run, one of which reached the observed AUC, so the 95% interval of the exceedance
  probability extends to 0.13;
- after Holm correction across the 8 exploratory models, $p=0.39$.

It is therefore a **hypothesis**, not a finding: the whole ERP waveform may carry a weak, temporally distributed
signal that fixed peaks miss.
