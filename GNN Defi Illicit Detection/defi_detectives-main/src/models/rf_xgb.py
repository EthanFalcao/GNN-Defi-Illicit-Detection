# THIS CODE RUNS THE RANDOM FOREST AND XGBOOST BASELINE MODELS USING TABULAR DATA

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, roc_curve
from xgboost import XGBClassifier
from src.utils.load_data import load_tabular
import matplotlib.pyplot as plt
import numpy as np

CODE_SEED = 42

# -------------------
# EVALUATE FUNCTION
# ------------------
def evaluate(model, X, y, name, threshold=0.5):
  # y_pred = model.predict(X)
  y_prob = model.predict_proba(X)[:, 1]
  y_pred = (y_prob >= threshold).astype(int)

  print(f"\n====================")
  print(f" Model: {name}")
  print(f" Threshold: {threshold}")
  print(f"====================")

  print("\nClassification Report:")
  print(classification_report(y, y_pred))

  print("ROC-AUC:", roc_auc_score(y, y_prob))
  print("Average Precision (AP):", average_precision_score(y, y_prob))

  print("\nFraud-class metrics (label=1):")
  print("Precision:", precision_score(y, y_pred, pos_label=1))
  print("Recall:", recall_score(y, y_pred, pos_label=1))
  print("F1-score:", f1_score(y, y_pred, pos_label=1))

  print("\n Non Fraud-class metrics (label=0):")
  print("Precision:", precision_score(y, y_pred, pos_label=0))
  print("Recall:", recall_score(y, y_pred, pos_label=0))
  print("F1-score:", f1_score(y, y_pred, pos_label=0))

# -------------------
# ROC PLOT
# -------------------
def plot_roc(models, X, y):

    plt.figure()

    for model, name in models:
        y_prob = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc = roc_auc_score(y, y_prob)

        plt.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()

    plt.savefig("rf_xgb_roc_curve.png", dpi=300)
    plt.show()
    

# -------------------
# RANDOM FOREST
# ------------------
def run_rf(X_train, y_train):
  rf = RandomForestClassifier(
    n_estimators = 200,
    max_depth = 10,
    n_jobs = -1,
    class_weight = "balanced",
    random_state = CODE_SEED
  )

  rf.fit(X_train, y_train)
  return rf


# -------------------
# XGBOOST
# ------------------
def run_xgb(X_train, y_train):

  scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
  
  xgb = XGBClassifier(
    n_estimators = 300,
    max_depth = 8,
    learning_rate = 0.05,
    subsample = 0.8,
    colsample_bytree = 0.8,
    eval_metric = "logloss",
    scale_pos_weight = scale_pos_weight,  # for imbalance
    random_state = CODE_SEED
  )

  xgb.fit(X_train, y_train)
  return xgb


# -------------------
# MAIN FUNCTION
# ------------------
def main():

  X, y, tx_ids, timesteps, train_idx, test_idx = load_tabular()

  X_train, y_train = X[train_idx], y[train_idx]
  X_test, y_test = X[test_idx], y[test_idx]

  print("Training Random Forest...")
  rf = run_rf(X_train, y_train)

  print("\nTraining XGBoost...")
  xgb = run_xgb(X_train, y_train)

  print("Evaluating models...")
  evaluate(rf, X_test, y_test, "Random Forest", threshold=0.5)
  evaluate(xgb, X_test, y_test, "XGBoost", threshold=0.5)

  print("\nPlotting ROC curves...")
  plot_roc([(rf, "Random Forest"),
      (xgb, "XGBoost")],
      X_test, y_test)

if __name__ == "__main__":
  main()
