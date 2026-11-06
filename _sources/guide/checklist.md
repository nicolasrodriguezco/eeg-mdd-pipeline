# Reporting checklist

These checks apply to any EEG classification study of depression. Each one comes with the number from this study
that shows why it matters.

| Check | Why it matters here |
|---|---|
| Split by participant, and say so | Trial-wise splitting gave a participant-level AUC of 1.00, versus 0.50 for the same model split by participant |
| Fit everything inside the fold | Scaling, selection and tuning on the full data leak the test set |
| Report a demographic baseline | Age, sex and education alone reached AUC 0.78 |
| Test the increment over that baseline | Adding EEG to demographics did not improve on them (0.52) |
| Permute the full pipeline | Chance AUCs up to 0.68 occurred in 5% of permutations with $n=45$ |
| Correct for, or nest, model selection | The best of 18 models (0.72) fell to 0.63 when its selection was nested |
| Control for muscle activity | Complexity and gamma features tracked a raw EMG index (ρ up to 0.59) |
| Report participant-level CIs and power | With 18 patients, only large effects were detectable |
| Pre-specify and log deviations | This keeps a negative result interpretable and a positive one credible |

## Questions a reviewer can ask

1. Can any participant contribute data to both training and test? Show the code that prevents it.
2. Is the reported metric per trial or per participant?
3. Were hyperparameters, feature selection or thresholds chosen on data that include the test set?
4. What does a model using only age, sex and education achieve on the same folds?
5. How many models were tried, and is the reported one corrected for that?
6. What is the 95th percentile of the permutation null for this sample size?
7. Could muscle, eye or recording-period differences produce the effect?
