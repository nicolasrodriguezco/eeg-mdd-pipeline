# EEG and major depression: a leakage-free benchmark

This book documents the code, methods and results of the study

> **Participant-wise evaluation removes the apparent EEG signature of major depression.**
> A pre-specified, leakage-free benchmark on the MODMA 128-channel dot-probe dataset, with a diagnosis of why
> published accuracies inflate. Nicolas Rodriguez, Data Science Program, Division of Basic Sciences,
> Universidad del Norte, Barranquilla, Colombia.
> [LinkedIn](https://www.linkedin.com/in/nicolasrodriguez04/) ·
> [nmcontreras@uninorte.edu.co](mailto:nmcontreras@uninorte.edu.co)

Source code: [github.com/nicolasrodriguezco/eeg-mdd-pipeline](https://github.com/nicolasrodriguezco/eeg-mdd-pipeline).

It is the companion to the preprint. The preprint makes the scientific argument; this book shows **how every number
was produced**, so that the analysis can be checked, reused and extended.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} The question
Can task EEG classify major depressive disorder (MDD) versus healthy controls **in people the model has never
seen**?
+++
[Read more »](study/question.md)
:::

:::{grid-item-card} The answer in one figure
Split by trial, the same model looks perfect (participant-level AUC 1.00). Split by participant, it is at chance
(0.50).
+++
[See the inflation control »](results/inflation.md)
:::

:::{grid-item-card} What you can reuse
A reference benchmark for MODMA, a reporting checklist, sample-size guidance and a documented, tested pipeline.
+++
[Reporting checklist »](guide/checklist.md)
:::

:::{grid-item-card} The code
18 numbered scripts and a tested Python package (`eeg_mdd`), from raw EGI files to the paper's LaTeX macros.
+++
[Browse the code »](code/scripts.md)
:::
::::

## Key results

| | Participant-level AUC [95% CI] |
|---|---|
| Pre-specified primary model (L2 logistic regression, 288 features) | 0.53 [0.36, 0.70], $p=0.456$ |
| Demographic baseline (age, sex, education) | 0.78 [0.63, 0.91] |
| Same tangent-space model, **trial-wise split (invalid)** | 1.00 |
| Same tangent-space model, participant-wise split | 0.50 |
| Single trials identify *which* of 45 participants produced them | 97.9% accuracy (chance 2.2%) |

All numbers are computed per participant, with bootstrap confidence intervals over participants and permutation
tests that re-run the full nested pipeline.

```{admonition} Data are not distributed
:class: warning
The EEG data belong to the MODMA project and are available from its maintainers for academic use. This book never
executes code on participant data and contains only aggregated results. **This is research code, not a clinical
tool.**
```
