"""Recompute the uncertainty intervals reported in the paper."""
import numpy as np

TN, FP, FN, TP = 979, 0, 11, 122
N = TN + FP + FN + TP


def wilson(successes: int, trials: int, z: float = 1.96):
    p = successes / trials
    denom = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denom
    half = z * np.sqrt(p * (1-p) / trials + z**2 / (4 * trials**2)) / denom
    return p, center - half, center + half

print("Wilson 95% intervals")
for name, x, n in [
    ("accuracy", TN + TP, N),
    ("precision", TP, TP + FP),
    ("recall", TP, TP + FN),
    ("specificity", TN, TN + FP),
]:
    p, lo, hi = wilson(x, n)
    print(f"{name:12s}: {p:.6f} ({lo:.6f}, {hi:.6f})")

rng = np.random.default_rng(42)
B = 100_000
probs = np.array([TN, FP, FN, TP], dtype=float) / N
samples = rng.multinomial(N, probs, size=B)
tn, fp, fn, tp = samples.T
precision = np.divide(tp, tp + fp, out=np.zeros(B), where=(tp + fp) > 0)
recall = np.divide(tp, tp + fn, out=np.zeros(B), where=(tp + fn) > 0)
f1 = np.divide(2 * precision * recall, precision + recall,
                out=np.zeros(B), where=(precision + recall) > 0)
lo, hi = np.quantile(f1, [0.025, 0.975])
print(f"f1          : {2*122/ (2*122 + 0 + 11):.6f} ({lo:.6f}, {hi:.6f})")
