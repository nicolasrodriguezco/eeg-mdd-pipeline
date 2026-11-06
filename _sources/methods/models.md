# Models

Eighteen configurations from five families are evaluated, all on the same participant partitions.

## Demographic baseline
L2 logistic regression on age, sex and education. It shows how much of any EEG result demographics alone could
explain. Demographics are **never** used as EEG predictors; they appear only here and in the confound analyses.

## Handcrafted features
Each model is a scikit-learn pipeline: median imputation, standardisation, then the classifier.

- **Primary (fixed a priori):** L2 logistic regression on all 288 features.
- **Secondary:**
  - L1 logistic regression, linear and RBF SVM, random forest and depth-limited LightGBM;
  - L2 logistic regression on each feature family separately.
- **Nested model selection:** the inner loop chooses both the model family and its hyperparameters. This gives an
  honest estimate of "pick the best model".

## Riemannian (trial level)
- OAS-shrunk covariances of the regional epoch signals, either mapped to the tangent space and fed to LR, or
  classified with MDM;
- ERP-augmented covariances, with class prototypes estimated inside each training fold, then tangent space and LR
  with fixed $C=0.1$ (deviation D5).

## Deep learning (trial level)
EEGNet, ShallowFBCSPNet and a temporal convolutional network (braindecode), with the following training setup:

- AdamW (learning rate $10^{-3}$, weight decay $10^{-3}$), cosine annealing and class-weighted cross-entropy;
- augmentation by time shifts (±40 ms), amplitude scaling and Gaussian noise;
- early stopping (patience 10, at most 60 epochs) on the participant-level AUC of a validation split of the
  **training participants**;
- channel normalisation statistics from the inner training participants only.

Trial-level probabilities are averaged within each participant to give one prediction per participant.

::::{dropdown} Source: model definitions
```{literalinclude} ../../src/eeg_mdd/models/__init__.py
:language: python
```
::::

::::{dropdown} Source: deep-learning training loop
```{literalinclude} ../../src/eeg_mdd/models/deep.py
:language: python
```
::::
