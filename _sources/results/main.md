# Main results

All metrics are per participant, with 45 participants (18 MDD, 27 HC). The 95% CIs come from a participant
bootstrap, and the $p$-values from label permutation of the full nested pipeline. The deep networks were run with 3
repetitions and no permutation test, for computational reasons.

```{include} ../_generated/main_results.md
```

## How to read this table

- The **primary model** (LR-L2, all features) was fixed before any result existed, and it is at chance.
- **L1-LR** is the only nominally significant secondary model. It does not survive multiplicity correction, and
  *nested model selection*, the honest estimate of "pick the best", falls to 0.63, with a CI that includes chance.
- The **demographic baseline** is the strongest model. It reflects recruitment (hospital patients versus controls
  recruited by posters, with a three-year education gap), not a diagnostic signal.

```{figure} ../_generated/figures/fig_model_comparison.png
:width: 90%
Participant-level AUC with 95% bootstrap CIs for all models.
```

```{figure} ../_generated/figures/fig_permutation.png
:width: 90%
Permutation null distributions of the participant-level AUC, with the observed AUC marked.
```

```{figure} ../_generated/figures/fig_roc.png
:width: 70%
Participant-level ROC curves.
```

## Signal quality and neurophysiology (descriptive)

```{figure} ../_generated/figures/fig_quality.png
:width: 90%
Signal-quality measures by group.
```

```{figure} ../_generated/figures/fig_topomaps_power.png
:width: 90%
Band-power topographies by group.
```

```{figure} ../_generated/figures/fig_erp_groups.png
:width: 90%
Grand-average ERPs by group and condition.
```

```{figure} ../_generated/figures/fig_family_importance.png
:width: 70%
Permutation importance of whole feature families, on the test folds of the primary model.
```

```{figure} ../_generated/figures/fig_shap.png
:width: 70%
SHAP values of the primary model (descriptive).
```
