"""Shared constants and metric helpers for the Mobile Price project.

Used by both train_models.py (training) and app.py (evaluation only), so the
column order and the way metrics are computed can never drift apart.
"""
from pathlib import Path
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=".*encountered in matmul.*",
    module=r"sklearn\..*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*`probability` parameter was deprecated.*",
    module=r"sklearn\.svm\._base",
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"

TARGET = "price_range"
FEATURES = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
    "touch_screen", "wifi",
]
CLASS_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Very high"}

# display name -> saved file name
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def compute_metrics(y_true, y_pred, y_proba, classes):
    """Six assignment metrics. Multi-class -> weighted average / one-vs-rest AUC."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="weighted", labels=classes
        ),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def evaluate_model(model, X, y):
    """Predict with an already-fitted model (no fitting here) and score it."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return y_pred, compute_metrics(y, y_pred, y_proba, np.asarray(model.classes_))
