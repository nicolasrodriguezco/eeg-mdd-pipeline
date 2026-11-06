# Preprocessing

All steps run **separately for each participant** with MNE-Python, so no information is shared across
participants.

1. **Bad channels.** Detected with the PREP pipeline (PyPREP `NoisyChannels`, RANSAC off). The criteria are
   NaN/flat signal, deviation, high-frequency noise, low correlation, low SNR, dropout and abnormal PSD. The vertex
   reference is exempt, being flat by construction. This replaced the pre-specified detector, which failed on these
   data (deviation D1).
2. **ICA.** Picard with the extended-infomax objective and 30 components, fitted on a copy filtered at 1–100 Hz,
   notched at 50 Hz and average-referenced, with bad channels excluded.
3. **ICLabel.** A component is removed when its most probable class is eye, muscle, heart, line noise or channel
   noise **and** that probability is at least 0.80. Decisions are logged in `configs/ica_exclusions.csv`.
4. **Analysis copy.** The analysis copy then goes through these steps:
   - 0.5–40 Hz zero-phase FIR band-pass and 50 Hz notch;
   - average reference, then ICA cleaning;
   - spherical-spline interpolation of bad channels;
   - a second average reference.
5. **Epochs.** From −200 to 600 ms, with a −200 to 0 ms baseline. An epoch is rejected if any channel exceeds
   ±100 µV.
6. **Segments.** For the spectral, connectivity and complexity features, the task period is resampled to 125 Hz and
   cut into 2-s segments, with the same amplitude rule.

## Participant exclusion (fixed before any result)

A participant is excluded if any of these holds:

- more than 20% of channels are bad;
- fewer than 240 of 480 trials survive rejection;
- fewer than 40 trials survive in any condition.

These rules excluded 6 MDD and 2 HC participants, leaving 45. The asymmetry was not significant (Fisher $p=0.12$).
Re-running the primary model on all 53 participants gave the same conclusion.

::::{dropdown} Source: preprocessing module
```{literalinclude} ../../src/eeg_mdd/preprocess.py
:language: python
```
::::
