Conformal Prediction – Exercises
=================================

Two Jupyter notebooks exploring conformal prediction from first principles,
using small synthetic or standard datasets.


FILES
-----
ConformalPred_regression.ipynb
ConformalPred_classification.ipynb


DEPENDENCIES
------------
Regression:    numpy, matplotlib, scikit-learn
Classification: numpy, matplotlib, torch, torchvision, pandas


REGRESSION NOTEBOOK
-------------------
Dataset: 18 generated test points (numpy random).

Sections:
  1. Setup & Dataset
     Generates the 8-point dataset and 10 new test points.

  2. Baseline Linear Regression
     Fits OLS on the 8 points and plots the fit with residuals.

  3. Naive Conformal Prediction (all data)
     Computes conformity scores on the full dataset (no split) and draws a
     symmetric prediction band. Also shows the band with the 10 new points.
     alpha = 0.25 (75% coverage).

  4. Split Conformal Prediction – Mean SCP
     Splits the 18-point dataset into train / calibration / test (6 each).
     Fits a fresh OLS on train, calibrates the quantile on the calibration
     residuals, and plots the symmetric band.

  5. Quantile Regression SCP
     Fits lower (alpha/2) and upper (1-alpha/2) quantile regressors on the
     train set. Conformity scores are max(lower-y, y-upper). The band is
     asymmetric and adapts to the data distribution.

  6. Weighted SCP
     Fits a secondary linear model to predict the magnitude of training
     residuals (heteroscedastic weights). Conformity scores are normalized
     by these weights, producing an adaptive-width band.

  7. Full Conformal Prediction (FCP)
     No train/calibration split. For each candidate point (i, j) in a 10x10
     grid, temporarily appends it to the full dataset, refits the model, and
     includes the point in the prediction set if its residual falls within the
     (1-alpha) quantile of all residuals.


CLASSIFICATION NOTEBOOK
-----------------------
Dataset: MNIST (20% of training set used, full test set for evaluation).
Model: lightweight CNN – one conv layer + two fully-connected layers.

Sections:
  1. Setup & Dataset
     Loads MNIST, draws a 20% random subset for training.

  2. Model Definition
     Sequential CNN: Conv2d(1,1) -> ReLU -> MaxPool2d(4) -> Linear(49,14)
     -> ReLU -> Linear(14,10).

  3. Training
     1 epoch, CrossEntropyLoss, Adam (lr=0.001).

  4. Evaluation & Softmax Probabilities
     Reports test accuracy. Draws 14 random test examples with true/predicted
     labels and builds a probability DataFrame (columns: example, true, pred,
     0..9). The first 10 rows are used as the calibration set, the last 4 as
     the test set.

  5. Vanilla Split Conformal Classification (SCP)
     Conformity score = softmax probability of the true class.
     Threshold q = alpha-quantile of calibration scores.
     Prediction set for each test point = classes whose probability >= q.
     alpha = 0.25.

  6. Adaptive Prediction Sets (APS)
     Classes are sorted by descending probability. Calibration score =
     cumulative probability down to and including the true class. Prediction
     sets are built by accumulating classes until the cumulative sum exceeds
     the threshold q, producing smaller sets for confident predictions.


NOTES
-----
- Both notebooks use fixed random seeds for reproducibility.
- alpha = 0.25 throughout, giving a 75% marginal coverage guarantee.
- The classification notebook requires an internet connection on first run
  to download MNIST via torchvision.
