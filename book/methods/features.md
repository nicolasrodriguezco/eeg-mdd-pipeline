# Features

All 288 features are computed **per participant and without labels**, so they cannot leak test information.
Everything that *learns* from data happens inside the cross-validation loop.

## Regions

There are ten regions: frontal, central, temporal, parietal and occipital, each split into left and right. Each
HydroCel electrode goes to the region of its nearest 10–10 anchor on the fitted head sphere. Midline electrodes and
the lower face/neck ring (more than 115° from the vertex) are not used.

Regions differ in size:

| Region | Electrodes per side |
|---|---|
| Central | 14 |
| Frontal | 12–13 |
| Occipital | 9 |
| Parietal | 7–8 |
| Temporal | 3 |

The temporal regions are therefore the noisiest, and they are the closest to the temporalis muscle.

```{figure} ../_generated/figures/fig_regions.png
:width: 70%
Electrode-to-region assignment.
```

## Feature dictionary

| Name pattern | $n$ | Definition |
|---|---|---|
| `pow_<band>_<region>` | 50 | $\log_{10}$ absolute Welch power in delta 1–4, theta 4–8, alpha 8–13, beta 13–30 and low gamma 30–40 Hz |
| `rel_<band>_<region>` | 50 | Band power divided by 1–40 Hz power. The five bands of a region are nearly collinear |
| `faa_F3F4`, `faa_F7F8` | 2 | Frontal alpha asymmetry, $\ln\alpha_R-\ln\alpha_L$ (E124/E24, E122/E33) |
| `erp_<cond>_n170L/R_amp`, `_lat` | 12 | N170 minimum and its latency at 140–200 ms, around P7/P8, per condition |
| `erp_<cond>_p300_amp`, `_lat` | 6 | P300 mean amplitude (250–500 ms) and peak latency around Pz |
| `erp_<cond>_lpp_amp` | 3 | Late positive potential, mean amplitude 400–600 ms around Pz |
| `wpli_<band>_<A>__<B>` | 135 | Weighted phase-lag index between region-mean signals, in theta, alpha and beta (45 pairs × 3) |
| `sampen_`, `hfd_`, `lzc_<region>` | 30 | Sample entropy ($m=2$, $r=0.2$ SD), Higuchi fractal dimension ($k_{max}=10$) and normalised Lempel–Ziv complexity, over up to 100 2-s segments |

```{warning}
Peak latencies inside fixed windows are fragile. In 34% of participant–condition–side combinations, the N170 minimum
fell on the 200 ms window edge. In 24%, the P300 maximum fell on the 250 ms edge. Treat the latency features as
low-validity measures.
```

## Trial-level representations

- **Riemannian models** use the ten region-mean signals of each epoch (10 × 201 samples).
- **Deep networks** use the full 129 × 201 epochs.

::::{dropdown} Source: spectral features
```{literalinclude} ../../src/eeg_mdd/features/spectral.py
:language: python
```
::::

::::{dropdown} Source: ERP features
```{literalinclude} ../../src/eeg_mdd/features/erp.py
:language: python
```
::::

::::{dropdown} Source: connectivity and complexity
```{literalinclude} ../../src/eeg_mdd/features/connectivity_complexity.py
:language: python
```
::::

::::{dropdown} Source: regions
```{literalinclude} ../../src/eeg_mdd/features/regions.py
:language: python
```
::::
