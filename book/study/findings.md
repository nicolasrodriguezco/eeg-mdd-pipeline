# Findings at a glance

1. **No detectable diagnostic signal on unseen participants.** The pre-specified primary model reached AUC 0.53
   [0.36, 0.70] ($p=0.456$). No family of models (handcrafted features, Riemannian geometry or deep networks)
   did better than chance after correction.
2. **Demographics beat EEG.** Age, sex and education alone reached AUC 0.78 [0.63, 0.91]. Adding EEG did not
   improve on that baseline. This result reflects how the sample was recruited, not a diagnostic signal.
3. **Trial-wise splitting manufactures accuracy.** The same tangent-space model gave a participant-level AUC of 1.00
   with a trial-wise split and 0.50 with a participant-wise split.
4. **The mechanism is person recognition.** Single trials identify which of the 45 participants produced them with
   97.9% accuracy (chance 2.2%).
5. **Muscle activity is a checkable confound.** Complexity and high-frequency features tracked an EMG index computed
   on the raw recordings (Spearman ρ up to 0.59).
6. **Only large effects were detectable.** Even an ideal, pre-trained score with true AUC 0.70 would pass the
   permutation threshold only 61% of the time with 18 patients. With the same assumptions, 80% power needs
   about 60 participants for AUC 0.70, 100 for 0.65 and 300 for 0.60.
7. **One hypothesis worth testing.** A random forest on raw ERP waveforms was the only borderline EEG result
   ($p=0.049$ with 40 permutations; Holm-corrected $p=0.39$). It is not a finding, but it is the most concrete lead
   for an independent, pre-registered test.

Points 5–7 come from post-hoc, exploratory analyses ([audit](../methods/audit.md)) and are labelled as such
throughout.
