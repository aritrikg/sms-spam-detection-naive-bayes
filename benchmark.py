"""Leakage-safe comparative benchmark for the SMS spam study.

Run this only after placing the exact 5,559-row CSV beside the script or
passing its path explicitly. The script reports mean +/- standard deviation
across repeated stratified 5-fold cross-validation for six lightweight models.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef,
)

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 3


def load_data(path: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    required = {"type", "text"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {sorted(required)}")
    df = df.dropna(subset=["type", "text"]).copy()
    y = (df["type"].astype(str).str.lower() == "spam").astype(int)
    X = df["text"].astype(str)
    if y.nunique() != 2:
        raise ValueError("Both ham and spam labels are required.")
    return X, y


def models() -> dict[str, Pipeline]:
    return {
        "Count+MNB": Pipeline([
            ("vec", CountVectorizer()),
            ("clf", MultinomialNB()),
        ]),
        "TFIDF+MNB": Pipeline([
            ("vec", TfidfVectorizer()),
            ("clf", MultinomialNB()),
        ]),
        "Word-1to2+MNB": Pipeline([
            ("vec", CountVectorizer(ngram_range=(1, 2))),
            ("clf", MultinomialNB()),
        ]),
        "Char-3to5+MNB": Pipeline([
            ("vec", TfidfVectorizer(analyzer="char", ngram_range=(3, 5))),
            ("clf", MultinomialNB()),
        ]),
        "TFIDF+LR": Pipeline([
            ("vec", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "TFIDF+LinearSVC": Pipeline([
            ("vec", TfidfVectorizer()),
            ("clf", LinearSVC(random_state=RANDOM_STATE)),
        ]),
    }


def score_fold(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="Exact 5,559-row experimental CSV")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.csv"))
    args = parser.parse_args()

    X, y = load_data(str(args.csv))
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    rows: list[dict[str, float | str]] = []
    for name, model in models().items():
        fold_rows = []
        fit_times = []
        score_times = []
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            pred = model.predict(X_test)
            score_times.append(time.perf_counter() - start)

            fold_rows.append(score_fold(y_test.to_numpy(), pred))

        fold_df = pd.DataFrame(fold_rows)
        row: dict[str, float | str] = {"model": name}
        for metric in fold_df.columns:
            row[f"{metric}_mean"] = fold_df[metric].mean()
            row[f"{metric}_std"] = fold_df[metric].std(ddof=1)
        row["fit_time_mean_s"] = float(np.mean(fit_times))
        row["fit_time_std_s"] = float(np.std(fit_times, ddof=1))
        row["score_time_mean_s"] = float(np.mean(score_times))
        row["score_time_std_s"] = float(np.std(score_times, ddof=1))
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    pd.set_option("display.max_columns", None)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
