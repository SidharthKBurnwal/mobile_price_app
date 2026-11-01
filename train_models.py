"""Train the five classifiers on the Mobile Price dataset and save everything.

Run once (locally or on BITS Virtual Lab):
    python train_models.py

Inputs : data/train.csv   (Kaggle "Mobile Price Classification", 2000 x 21)
Outputs: model/*.joblib   (five fitted pipelines: scaler + classifier)
         test_data.csv    (20 % hold-out split, features + target)
         model/metrics.csv (comparison table used in the README)
"""
import sys

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import (
    FEATURES, MODEL_DIR, MODEL_FILES, ROOT, TARGET, TEST_CSV, evaluate_model,
)

SEED = 2026
RAW_CSV = ROOT / "data" / "train.csv"


def load_data():
    if not RAW_CSV.exists():
        sys.exit(f"Dataset not found: {RAW_CSV}\nPut Kaggle's train.csv there first.")
    df = pd.read_csv(RAW_CSV)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        sys.exit(f"train.csv is missing columns: {missing}")
    df = df[FEATURES + [TARGET]].drop_duplicates().reset_index(drop=True)
    print(f"Loaded {len(df)} rows, {len(FEATURES)} features, "
          f"classes: {sorted(df[TARGET].unique())}")
    return df


def build_models():
    """name -> (pipeline, hyperparameter grid searched on the training split only)."""
    cv_grids = {
        "Logistic Regression": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
            {"clf__C": [0.1, 1, 10, 100]},
        ),
        "Decision Tree": (
            Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))]),
            {"clf__max_depth": [3, 5, 8, 12, None], "clf__min_samples_leaf": [1, 5, 10]},
        ),
        "KNN": (
            Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [3, 5, 7, 11, 15, 21], "clf__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes (Gaussian)": (
            Pipeline([("scale", StandardScaler()), ("clf", GaussianNB())]),
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
        ),
        "Support Vector Machine": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", SVC(probability=True, random_state=SEED))]),
            {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]},
        ),
    }
    return cv_grids


def main():
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    MODEL_DIR.mkdir(exist_ok=True)
    pd.concat([X_test, y_test], axis=1).to_csv(TEST_CSV, index=False)
    print(f"Saved hold-out test set -> {TEST_CSV.name}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, (pipe, grid) in build_models().items():
        search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1)
        search.fit(X_train, y_train)          # fit on TRAIN split only
        best = search.best_estimator_
        joblib.dump(best, MODEL_DIR / MODEL_FILES[name])

        _, m = evaluate_model(best, X_test, y_test)
        rows.append({"ML Model Name": name, **m})
        print(f"\n{name}\n  best params: {search.best_params_}"
              f"\n  cv accuracy: {search.best_score_:.4f}"
              f"\n  test: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))

    table = pd.DataFrame(rows).round(4)
    table.to_csv(MODEL_DIR / "metrics.csv", index=False)
    print("\n=== Comparison table (hold-out test set) ===")
    print(table.to_string(index=False))
    print("\nMarkdown for README:\n")
    print(to_markdown(table))


def to_markdown(df):
    header = "| " + " | ".join(df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *body])


if __name__ == "__main__":
    main()
