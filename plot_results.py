import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def plot_class_distribution():
    results = pd.read_csv("results.csv").iloc[0]
    counts = pd.Series({"ham": int(results["ham_count"]), "spam": int(results["spam_count"])})

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    bars = ax.bar(counts.index, counts.values)
    ax.set_xlabel("Message class")
    ax.set_ylabel("Number of messages")
    ax.set_title("Class Distribution of the SMS Dataset")
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 40,
                str(value), ha="center", va="bottom", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "class_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix():
    cm = np.load("confusion_matrix.npy")
    fig, ax = plt.subplots(figsize=(4.4, 3.7))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"])
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix on the Test Set")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics():
    results = pd.read_csv("results.csv").iloc[0]
    metrics = ["Accuracy", "Spam Precision", "Spam Recall", "Spam F1"]
    values = [
        results["accuracy"],
        results["spam_precision"],
        results["spam_recall"],
        results["spam_f1"],
    ]

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    bars = ax.bar(metrics, values)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Observed Classification Performance")
    ax.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "results_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_error_profile():
    cm = np.load("confusion_matrix.npy")
    tn, fp, fn, tp = cm.ravel()
    spam_recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fnr = fn / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (tn + fp) if (tn + fp) else 0.0

    labels = ["Spam Recall", "Specificity", "False-Negative\nRate", "False-Positive\nRate"]
    values = [spam_recall, specificity, fnr, fpr]

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proportion")
    ax.set_title("Error-Sensitive Profile from Observed Confusion Matrix")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.03, 1.02),
            f"{value * 100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "error_profile.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_class_distribution()
    plot_confusion_matrix()
    plot_metrics()
    plot_error_profile()
