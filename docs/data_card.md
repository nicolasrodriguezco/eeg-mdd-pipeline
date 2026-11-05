# Data card — MODMA 128-channel ERP (dot-probe), as used in this project

## Source and permissions
- **Dataset:** MODMA, the Multi-modal Open Dataset for Mental-disorder Analysis (Cai et al., *Scientific Data* 9:178, 2022). We use only the 128-channel ERP subset (Lanzhou).
- **Access:** obtained from the MODMA maintainers for academic use. **Not redistributed** by this project, in raw or derived form.
- **Files:** 53 EGI `.raw` recordings (6.6 GB) and `subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx` (Sheet1, 53 rows).

## Participants
| | MDD | HC |
|---|---|---|
| Recordings | 24 | 29 |
| Age (mean ± SD) | 30.9 ± 10.4 | 31.4 ± 9.1 (p = 0.82) |
| Sex (F/M) | 11/13 | 9/20 (Fisher p = 0.39) |
| Education, years | 13.2 ± 3.8 | 16.0 ± 2.3 (**p = 0.005**) |
| Included after pre-specified QC | 18 | 27 |

- **Labels:** taken only from the spreadsheet column `type`, by exact match on the 8-digit ID. All 53 IDs matched one-to-one, with no manual mapping.
- **Questionnaires** (PHQ-9, GAD-7, CTQ-SF, LES, SSRS, PSQI) are descriptive only and never used as predictors.

## Recording
- EGI HydroCel GSN with 128 electrodes plus the vertex reference (E129 = Cz), sampled at 250 Hz.
- Recorded between Dec 2014 and Apr 2016. Recordings run 780–1158 s (median 807 s).
- Mains frequency: 50 Hz.
- **Task:** emotional-face dot-probe with 3 blocks × 160 trials (fear, happy, sad). Each trial runs: fixation (464–652 ms) → face pair (516 ms) → ISI (116–316 ms) → dot (168 ms). All 53 files contain all 480 trials.
- **Events:** decoded by EGI channel name (`fcue`, `hcue`, `scue`, ...). The combined `STI 014` channel is uninformative (code 1 for every event).
- The file-header timestamp is consistently about 5 h ahead of the time in the file name, an acquisition-PC clock offset. Three files carry "new"/"(n)" suffixes; each is the only recording for its participant and is complete.

## Known issues and confounds
- **Recording period vs. group:**

  | Period | HC | MDD |
  |---|---|---|
  | Dec 2014 – May 2015 | 0 | 4 |
  | Oct 2015 – Jan 2016 | 13 | 0 |
  | Mar – Apr 2016 | 0 | 7 |
  | Jun – Aug 2015 (both groups) | 16 | 13 |

- **Education** differs between groups.
- **Exclusions:** the pre-specified QC excluded 6 of 24 MDD and 2 of 29 HC, mostly for having fewer than 240 of 480 trials within ±100 µV. Because the rate differs by group, a sensitivity analysis on all 53 participants is reported.
- **Flat channel:** E114 is flat in recording 02010034 and was interpolated.
- **Previous version of this project:** it mislabelled nothing, but it split 7 participants across two ID spellings, dropped participant 02030009, and used trial-wise splits and a 60 Hz notch. None of its data are reused.

## Derived data produced (local only)
- Per-participant face-locked epochs (−200 to 600 ms, 250 Hz) and 2-s task segments (125 Hz).
- ICA decompositions, ICLabel decisions (`configs/ica_exclusions.csv`) and QC reports.
- 288 subject-level features (`subject_features.parquet`).
