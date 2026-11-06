# The dataset

The study uses the 128-channel ERP recordings of **MODMA** (Multi-modal Open Dataset for Mental-disorder Analysis;
Cai et al., 2022, *Scientific Data*), recorded in Lanzhou in 2014–2016.

| Property | Value |
|---|---|
| Recordings | 53, one per participant (24 MDD, 29 HC) |
| System | EGI HydroCel Geodesic Sensor Net, 128 scalp electrodes + vertex reference (E129/Cz) |
| Sampling rate | 250 Hz |
| Mains frequency | 50 Hz (China), hence the 50 Hz notch |
| Task | Emotional-face dot-probe, three blocks of 160 trials (fear, happy, sad) |
| Analysed after pre-specified quality exclusions | 45 (18 MDD, 27 HC) |

## Participants

According to the dataset description:

- **Patients** were in- and outpatients of Lanzhou University Second Hospital. Each met DSM-IV criteria for major
  depression on a structured MINI interview, had a PHQ-9 score of at least 5, and had received no psychotropic
  drugs in the previous two weeks.
- **Controls** were recruited by posters and excluded if they had a personal or family history of mental disorder.
- **Everyone** was 18–55 years old with at least primary education.

The groups did not differ significantly in age or sex, but patients had fewer years of education. In the 45 analysed
participants, education was 12.8 ± 3.5 years (MDD) versus 16.0 ± 2.4 years (HC), $p=0.001$. This difference
reflects recruitment, hospital versus posters, and it is why a demographic baseline is essential.

## Labels and identifiers

- The diagnostic group comes **only** from the official subject-information spreadsheet, by exact identifier match.
  It is never inferred from file names or ID prefixes. All 53 identifiers matched one-to-one.
- Questionnaires (PHQ-9, GAD-7, CTQ-SF, LES, SSRS, PSQI) describe the sample and are **never** predictors, because
  they measure, or are tied to, the target.

## Task and events

Each trial consists of the following sequence:

1. a fixation cross (464–652 ms);
2. a face pair, one emotional and one neutral (516 ms);
3. an interval (116–316 ms);
4. a dot probe, followed by the response.

Events are decoded from the individual EGI event channels by name (`fcue`, `hcue`, `scue`). The combined stimulus
channel carries the same code for every event in these files, so it is unusable.

Epochs are locked to face-pair onset, from −200 to 600 ms. The probe appears at least 632 ms after onset, so this
window is free of probe-evoked activity.

```{note}
The released EGI files contain event onsets but not trial keys: congruency, side of the emotional face and accuracy
are missing. The attentional-bias index for which the dot-probe was designed therefore cannot be computed. Reaction
times can, and they are analysed in the [audit](../methods/audit.md).
```
