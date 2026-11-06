# Post-hoc audit (exploratory)

After all pre-specified results were known, an expert audit of the study proposed further checks. They were run as
**deviation D6**, labelled post hoc and exploratory, with Holm correction among the new models. None of them changes
the pre-specified result.

| Analysis | Question | Script |
|---|---|---|
| Demographics + EEG with block permutation | Does EEG add anything to demographics? | `13_audit_models.py` |
| In-fold confound regression | Is the null result an artefact of age, sex or education? | `13_audit_models.py` |
| PCA + LR and a compact 36-feature set | Is it an artefact of 288 redundant features? | `13_audit_models.py` |
| Subject-level Riemannian model | Does a mean-covariance representation help? | `13_audit_models.py` |
| Reaction times (`*dot` → `resp`) | Does behaviour separate the groups? | `12_audit_features.py` |
| Raw EMG index (55–95 / 1–40 Hz) | Do complexity and gamma features track muscle activity? | `12_audit_features.py` |
| Robust aperiodic fit and alpha peak | Are aperiodic measures informative? | `12_audit_features.py` |
| EEG → education | Could education explain the EEG? | `13_audit_models.py` |
| Power: ideal score and semi-synthetic | What effect could this sample detect? | `13_audit_models.py` |
| Sample-size extrapolation | How many participants would smaller effects need? | `16_sample_size.py` |
| Learning curve (12–36 training participants) | Would more data help the primary model? | `13_audit_models.py` |
| Subject identification | Why do trial-wise splits inflate? | `13_audit_models.py` |
| Raw-amplitude RF permutation | Is the best EEG control model significant? | `14_rawrf_permutation.py` |

## In-fold confound regression

Confounds are regressed out of the features using coefficients fitted **on the training fold only**:

```{literalinclude} ../../src/eeg_mdd/models/deconfound.py
:language: python
:pyobject: ConfoundRegressor
```

## Muscle index from the raw recording

```{literalinclude} ../../src/eeg_mdd/features/audit.py
:language: python
:pyobject: muscle_index
```

::::{dropdown} Source: aperiodic fit, alpha peak and reaction times
```{literalinclude} ../../src/eeg_mdd/features/audit.py
:language: python
```
::::
