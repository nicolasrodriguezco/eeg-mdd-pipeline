# Scripts

The pipeline steps, in order. Each script reads only the outputs of earlier steps and the configuration in
`configs/config.yaml`. Code comments and docstrings are in Spanish.

::::{dropdown} Configuration: configs/config.yaml
```{literalinclude} ../../configs/config.yaml
:language: yaml
```
::::

## Data preparation

::::{dropdown} 01_inventory.py: inventory of the raw files
```{literalinclude} ../../scripts/01_inventory.py
:language: python
```
::::

::::{dropdown} 02_audit.py: per-recording audit and EGI events
```{literalinclude} ../../scripts/02_audit.py
:language: python
```
::::

::::{dropdown} 03_match_ids.py: exact ID matching
```{literalinclude} ../../scripts/03_match_ids.py
:language: python
```
::::

::::{dropdown} 03b_demographics.py: demographics and recording periods
```{literalinclude} ../../scripts/03b_demographics.py
:language: python
```
::::

::::{dropdown} 04_preprocess.py: preprocessing and exclusion
```{literalinclude} ../../scripts/04_preprocess.py
:language: python
```
::::

::::{dropdown} 05_compare_old.py: comparison with the earlier dataset
```{literalinclude} ../../scripts/05_compare_old.py
:language: python
```
::::

## Features and models

::::{dropdown} 06_features.py: subject-level features and trial-level data
```{literalinclude} ../../scripts/06_features.py
:language: python
```
::::

::::{dropdown} 07_models.py: classic, demographic and Riemannian models
```{literalinclude} ../../scripts/07_models.py
:language: python
```
::::

::::{dropdown} 08_deep.py: deep learning
```{literalinclude} ../../scripts/08_deep.py
:language: python
```
::::

## Analysis and reporting

::::{dropdown} 09_confounds.py: confounds, sensitivity and inflation control
```{literalinclude} ../../scripts/09_confounds.py
:language: python
```
::::

::::{dropdown} 10_interpret.py: interpretation
```{literalinclude} ../../scripts/10_interpret.py
:language: python
```
::::

::::{dropdown} 11_figures_tables.py: figures, tables and LaTeX macros
```{literalinclude} ../../scripts/11_figures_tables.py
:language: python
```
::::

## Post-hoc audit (exploratory, D6)

::::{dropdown} 12_audit_features.py: audit features
```{literalinclude} ../../scripts/12_audit_features.py
:language: python
```
::::

::::{dropdown} 13_audit_models.py: audit models, power, learning curve, identification
```{literalinclude} ../../scripts/13_audit_models.py
:language: python
```
::::

::::{dropdown} 14_rawrf_permutation.py: raw-amplitude RF permutation
```{literalinclude} ../../scripts/14_rawrf_permutation.py
:language: python
```
::::

::::{dropdown} 15_audit_report.py: audit macros, table and figures
```{literalinclude} ../../scripts/15_audit_report.py
:language: python
```
::::

::::{dropdown} 16_sample_size.py: sample-size extrapolation
```{literalinclude} ../../scripts/16_sample_size.py
:language: python
```
::::

::::{dropdown} 17_book_assets.py: assets for this book
```{literalinclude} ../../scripts/17_book_assets.py
:language: python
```
::::
