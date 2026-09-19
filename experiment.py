import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DATA_CANDIDATES = [
    "spamraw.csv",
    "../input/spamraw.csv",
]
RANDOM_STATE = 0
TEST_SIZE = 0.20


def load_data():
    data_file = next((p for p in DATA_CANDIDATES if os.path.exists(p)), None)
    if data_file is None:
        raise FileNotFoundError(
            "Could not find spamraw.csv. Place it in the project directory "
            "or in ../input/spamraw.csv."
        )

    df = pd.read_csv(data_file)
    required = {"type", "text"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected columns {sorted(required)}, found {list(df.columns)}"
        )

    df = df[["type", "text"]].dropna()
    df["type"] = df["type"].astype(str).str.lower().str.strip()
    df["text"] = df["text"].astype(str)
    df = df[df["type"].isin(["ham", "spam"])]
    return df, data_file


def main():
    df, data_file = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["type"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    vectorizer = CountVectorizer()
    X_train_count = vectorizer.fit_transform(X_train)
    X_test_count = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_count, y_train)
    predictions = model.predict(X_test_count)

    labels = ["ham", "spam"]
    cm = confusion_matrix(y_test, predictions, labels=labels)

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    results = pd.DataFrame(
        [{
            "dataset_file": data_file,
            "dataset_size": len(df),
            "ham_count": int((df["type"] == "ham").sum()),
            "spam_count": int((df["type"] == "spam").sum()),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "features": X_train_count.shape[1],
            "accuracy": accuracy_score(y_test, predictions),
            "spam_precision": precision_score(y_test, predictions, pos_label="spam"),
            "spam_recall": recall_score(y_test, predictions, pos_label="spam"),
            "spam_f1": f1_score(y_test, predictions, pos_label="spam"),
            "weighted_precision": report["weighted avg"]["precision"],
            "weighted_recall": report["weighted avg"]["recall"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        }]
    )

    results.to_csv("results.csv", index=False)
    np.save("confusion_matrix.npy", cm)

    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, predictions, digits=4, zero_division=0))

    print("Dataset:", data_file)
    print("Dataset size:", len(df))
    print("Train/Test:", len(X_train), len(X_test))
    print("Vocabulary size:", X_train_count.shape[1])
    print("\nConfusion matrix (ham, spam):\n", cm)
    print("\nClassification report:\n")
    print(classification_report(y_test, predictions, digits=4, zero_division=0))
    print("Accuracy:", accuracy_score(y_test, predictions))


if __name__ == "__main__":
    main()
