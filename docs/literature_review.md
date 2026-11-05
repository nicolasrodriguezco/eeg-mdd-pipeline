# Literature review and audit of the previous essay (`main.pdf`, Nov 2025)

Purpose: a resource for the new paper. Every reference that enters `docs/paper/refs.bib` was verified against
a bibliographic source in this session (2026-11-04). Claims that could not be verified are flagged and **not used**.

## 1. Audit of the previous essay (Universidad del Norte, seminar project, 50 pp.)

### 1.1 Citation problems
**Unresolved keys:** citations appear as raw keys (`WHO2024`, `Sadeghi2024`, `Simmatis2023`, `Ay2019`, `Lin2024`,
`Ying2024`), and the document has **no bibliography**.

| Key in old essay | What the essay claims | Verification | Decision |
|---|---|---|---|
| Simmatis2023 | Review of EEG biomarkers in MDD: FAA has moderate effects; P300/LPP differ consistently; heterogeneity is high | **Paper exists:** Simmatis, Russo, Geraci, Harmsen & Samuel (2023), *npj Mental Health Research* 2:18, doi:10.1038/s44184-023-00038-7. It is a narrative review of technical and clinical considerations. The essay's specific bullet points could not be checked against the full text. | Cite for the general point (no objective markers; need for standardised pipelines). Do not reuse the bullets. |
| Ay2019 | CNN-LSTM, **30 MDD / 30 HC**, resting state, **94.7%**, 10-fold CV | **Paper exists:** Ay et al. (2019), *J Med Syst* 43:205, doi:10.1007/s10916-019-1345-y. Its abstract reports **99.12% / 97.66%** (right/left hemisphere), **not 94.7%**. The subject numbers in the essay could not be verified. | The essay's numbers are wrong. Cite only as an example of very high accuracies reported for segment-based CNN-LSTM pipelines. |
| Lin2024 | Table: SVM 76.3%, RF 78.1%, CNN 82.5%, LSTM 79.8%, CNN-LSTM 85.2% | **Not found.** No verifiable source matches. | **Drop.** Probably fabricated by the earlier AI. |
| Ying2024 | "EDT (EEG Depression Transformer)", 87.3% multicentre | **Not found.** | **Drop.** |
| Sadeghi2024 | ERP differences in emotional processing | **Not verified.** | **Drop.** Replaced by verified ERP references. |
| WHO2024 | More than 280 million people with depression | A WHO fact sheet figure; plausible but not re-checked here. | Not needed in a methods paper. Omitted. |

### 1.2 Methodological statements to correct (they contradict what we now know)
- **Sample size:** "12,024 trials of 53 participants". The real counts are 53 recordings × 480 trials = 25,440 face-locked
  trials. The old pipeline kept 12,025 trials from only 52 participants, and 7 of them were split across two IDs.
- **Prefix fallback:** labels fell back to the ID prefix (`0201` → MDD). Labels must come only from the clinical sheet.
- **Notch:** a 60/120 Hz notch was applied, but mains in China is 50 Hz.
- **Trial-wise split:** the essay calls stratified 5-fold "rigorous", but it split trials, not subjects. The ∆ ≈ −0.2
  train–test gap was read as "overfitting". Most of the reported test performance is plausibly
  subject-identity leakage (see §2.2).
- **Gender as a predictor:** gender was used as a predictor and was the top random-forest feature.
- **Group topographies:** "more frontal activity in MDD, occipital in HC" came from visual inspection of group mean
  topographies, with no statistics. It is replaced by cluster-based permutation tests.
- **"Future work: add PHQ-9/GAD-7 as features":** this would be target leakage, because the questionnaires measure
  depression.

### 1.3 Content worth keeping
- The clinical motivation, the description of the questionnaires and the ERP component overview (P1/N1/P2/N2/P3/LPP).
  The ERP overview should be supported by the verified references below.
- The attentional-bias framing of the dot-probe task (now cited to MacLeod et al. 1986 and Li et al. 2018).

## 2. Verified literature (used in the new paper)

### 2.1 Dataset and the same paradigm
- **Cai et al. (2022)**, *Scientific Data* 9:178. The MODMA dataset paper. The 128-channel ERP subset has 24 MDD and
  29 HC, 250 Hz, and a dot-probe task with fear/sad/happy–neutral face pairs. (An arXiv version, 2002.09283, exists.)
- **Li, Li, Hu et al. (2018)**, *Computer Methods and Programs in Biomedicine* 164:169–179,
  doi:10.1016/j.cmpb.2018.07.003. This is the Lanzhou group's ERP study of the same paradigm family.
  - **Sample and design:** 17 MDD vs 17 HC; emotional–neutral face pairs. The main finding is larger P300 when the
    probe replaces a sad face, which the authors read as difficulty disengaging from negative faces. The study also
    tested ERP components for classification (CFS feature selection performed best).
  - **Not verified:** the classification accuracy is not in the abstract, so we do not quote it. The cue duration is
    reported as 100 ms in secondary sources; in our data the face pair lasts 516 ms.

### 2.2 Leakage and evaluation methodology
- **Saeb et al. (2017)**, *GigaScience* 6(5). Record-wise versus subject-wise cross-validation: record-wise splits
  overestimate performance.
- **Brookshire et al. (2024)**, *Frontiers in Neuroscience* 18:1373515, doi:10.3389/fnins.2024.1373515.
  The authors state that most translational DNN-EEG studies suffer from leakage due to segment-wise splits;
  segment-based CV overestimates performance.
- **Kapoor & Narayanan (2023)**, *Patterns* 4(9):100804. Leakage taxonomy across ML-based science.
- **Varoquaux (2018)**, *NeuroImage* 180:68–77. Small-sample CV error bars.
- **Combrisson & Jerbi (2015)**, *J Neurosci Methods* 250:126–136. Empirical chance levels via permutation.
- **Cawley & Talbot (2010)**, *JMLR* 11:2079–2107. Selection bias when the best model is chosen on the same CV.
- **Nadeau & Bengio (2003)**, *Machine Learning* 52:239–281. Corrected resampled t-test.

### 2.3 EEG biomarkers of MDD (reviews)
- **de Aguiar Neto & Rosa (2019)**, *Neurosci Biobehav Rev* 105:83–93, doi:10.1016/j.neubiorev.2019.07.021.
  A systematic mapping of studies from 2014 to 2018.
  - Depressed individuals show more random network structure.
  - Gamma and theta power look promising.
  - Alpha asymmetry relates to specific symptoms.
- **Olbrich & Arns (2013)**, *Int Rev Psychiatry* 25(5):604–618, doi:10.3109/09540261.2013.816269. Reviews qEEG,
  connectivity, vigilance, sleep EEG and ERP markers for diagnosis and treatment prediction, and calls for
  standardised procedures.
- **Simmatis et al. (2023)**, *npj Mental Health Research* 2:18. Technical and clinical considerations for EEG
  biomarkers.
- **van der Vinne et al. (2017)**, *NeuroImage: Clinical* 16:79–87. Meta-analysis: frontal alpha asymmetry has
  **no** diagnostic value for MDD.
- **Allen & Reznik (2015)**, *Curr Opin Psychol* 4:93–97. FAA as a vulnerability marker, and the methodological
  issues involved.

### 2.4 MODMA machine-learning studies (context; numbers quoted only where verified)
- **Zhang et al. (2025)**, arXiv:2503.13475 (DepL-GCN). MODMA 128-channel *resting* EEG, 53 subjects,
  leave-one-out cross-validation. Reports about 81% accuracy for multi-class *severity* recognition; the
  per-dataset number could not be verified.
- **Ay et al. (2019)**, *J Med Syst* 43:205. 97.66–99.12% accuracy with a CNN-LSTM on segments. This is typical of the
  very high accuracies reported for segment-based evaluation; it does not use MODMA.
- **Not used:**
  - A 2026 *Applied AI Letters* paper reporting about 71–75% cross-subject accuracy on the 128-channel ERP subset
    (the publisher page could not be accessed).
  - A GitHub repository reporting 74.3% (segment-wise) vs 49.6% (subject-wise) on MODMA, which is not peer-reviewed.
  - These are mentioned here only as leads for the author.

### 2.5 Methods references (already in `refs.bib`)
- **EEG processing:** MNE (Gramfort 2013), PREP (Bigdely-Shamlo 2015), Picard (Ablin 2018),
  ICLabel (Pion-Tonachini 2019), MNE-ICALabel (Li 2022).
- **Features and statistics:** wPLI (Vinck 2011), SampEn (Richman & Moorman 2000), Higuchi FD (1988),
  Lempel–Ziv (1976), cluster permutation (Maris & Oostenveld 2007), FDR (Benjamini & Hochberg 1995), SHAP (Lundberg & Lee 2017).
- **Riemannian methods:** Barachant 2012, Congedo 2017, OAS shrinkage (Chen 2010).
- **Deep learning:** EEGNet (Lawhern 2018), ShallowConvNet (Schirrmeister 2017), TCN (Bai 2018; Gemein 2020).
- **ERP components:** N170 (Bentin 1996), P300 (Polich 2007), LPP/emotion (Hajcak 2010).

## 3. How this shapes the new paper
1. **Related work** focuses on (a) MODMA and the dot-probe literature, (b) biomarker reviews, which show weak and
   heterogeneous effects (FAA is not diagnostic), and (c) leakage. We do not compare against unverifiable numbers.
2. **Positioning:** the honest subject-wise estimate is the contribution. Inflated literature numbers are explained
   by the leakage literature and by our own inflation-control experiment.
3. **Discussion:** link any null or weak EEG result to the meta-analytic evidence (van der Vinne 2017) and to the
   demographic and recording-period confounds found here.
