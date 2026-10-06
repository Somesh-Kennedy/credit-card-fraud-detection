"""
evaluate.py — metrics, cost-based threshold tuning and evaluation plots.

Pipeline stage 5 of the workflow.

Why not accuracy?
    With 0.58% fraud, "predict legit for everything" scores 99.42% accuracy.
    We therefore report metrics that focus on the rare class:

    * PR-AUC (average precision) — area under the precision-recall curve. The
      headline metric: it is NOT inflated by the huge number of easy legit rows.
    * ROC-AUC — commonly quoted, but looks great on imbalanced data even for weak
      models, so treat it as secondary.
    * Precision / recall / F1 at the chosen decision threshold.
    * Precision at a fixed recall ("to catch 90% of fraud, how many alerts are real?").
    * Dollar impact — fraud $ caught vs. missed, and analyst cost of false alarms.

Why tune the threshold?
    A classifier outputs a probability. The default cut-off of 0.5 is arbitrary;
    what matters to the business is cost. Missing a $1,200 fraud is far worse than
    reviewing a false alarm, so we choose the threshold that minimises:

        cost = sum(amount of every missed fraud) + FALSE_POSITIVE_COST * (# false alarms)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # render to files without needing a display (servers / CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from fraud import config

# One colour per model so every chart in the report is consistent.
PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2", "#64748b"]


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
def cost_curve(y_true, scores, amounts, fp_cost: float = config.FALSE_POSITIVE_COST):
    """Total business cost for every possible "flag the top-k transactions" cut-off.

    Vectorised trick: sort transactions by score (most suspicious first). Flagging
    the top k means:
        fraud $ caught   = cumulative sum of fraud amounts in the first k rows
        false positives  = cumulative count of legit rows in the first k rows
    so one cumsum gives the cost of ALL cut-offs at once instead of looping.

    Returns (thresholds, costs) where flagging `score >= thresholds[i]` costs costs[i].
    """
    y = np.asarray(y_true)
    s = np.asarray(scores)
    a = np.asarray(amounts)

    order = np.argsort(-s, kind="mergesort")          # descending score
    y_sorted, s_sorted, a_sorted = y[order], s[order], a[order]

    caught_amt = np.cumsum(a_sorted * y_sorted)        # fraud $ inside the flagged set
    false_pos = np.cumsum(1 - y_sorted)                # legit rows inside the flagged set
    total_fraud_amt = (a * y).sum()

    costs = (total_fraud_amt - caught_amt) + fp_cost * false_pos
    # Prepend the "flag nothing" option (threshold above every score).
    costs = np.concatenate([[total_fraud_amt], costs])
    thresholds = np.concatenate([[np.inf], s_sorted])
    return thresholds, costs


def best_cost_threshold(y_true, scores, amounts, fp_cost: float = config.FALSE_POSITIVE_COST) -> float:
    """Score threshold that minimises total business cost (see module docstring)."""
    thresholds, costs = cost_curve(y_true, scores, amounts, fp_cost)
    return float(thresholds[int(np.argmin(costs))])


def best_f1_threshold(y_true, scores) -> float:
    """Threshold maximising F1 — reported for reference / comparison."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision/recall have one more element than thresholds; drop the last point.
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def precision_at_recall(y_true, scores, target_recall: float) -> float:
    """Best precision achievable while still catching >= target_recall of fraud."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ok = recall >= target_recall
    return float(precision[ok].max()) if ok.any() else 0.0


def compute_metrics(y_true, scores, amounts, threshold: float) -> dict:
    """All headline numbers for one model on one dataset split."""
    y = np.asarray(y_true)
    s = np.asarray(scores)
    a = np.asarray(amounts)
    pred = (s >= threshold).astype(int)

    # ravel() flattens the 2x2 matrix [[TN, FP], [FN, TP]] in that order.
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    fraud_amt_total = float((a * y).sum())
    fraud_amt_caught = float((a * y * pred).sum())
    review_cost = float(fp * config.FALSE_POSITIVE_COST)

    metrics = {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "alerts": int(tp + fp),
        "fraud_amt_total": fraud_amt_total,
        "fraud_amt_caught": fraud_amt_caught,
        "fraud_amt_caught_pct": fraud_amt_caught / fraud_amt_total if fraud_amt_total else 0.0,
        "review_cost": review_cost,
        # Net value vs. having no model at all: money saved minus analyst cost.
        "net_savings": fraud_amt_caught - review_cost,
    }
    for r in config.RECALL_TARGETS:
        metrics[f"precision_at_{int(r * 100)}_recall"] = precision_at_recall(y, s, r)
    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _save(fig, name: str):
    """Save a figure into reports/figures at a README-friendly resolution."""
    config.ensure_dirs()
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pr_curves(y_true, scores_by_model: dict[str, np.ndarray], name="pr_curves.png", title=""):
    """Precision-recall curves of several models on the same axes."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for color, (model, scores) in zip(PALETTE, scores_by_model.items()):
        p, r, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(r, p, color=color, lw=2, label=f"{model}  (PR-AUC {ap:.3f})")
    # A random classifier's precision equals the fraud rate: draw it as a reference.
    ax.axhline(np.mean(y_true), color="grey", ls="--", lw=1, label="random")
    ax.set(xlabel="Recall (share of fraud caught)", ylabel="Precision (share of alerts that are fraud)",
           title=title or "Precision-recall curves", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, name)


def plot_roc_curves(y_true, scores_by_model: dict[str, np.ndarray], name="roc_curves.png", title=""):
    """ROC curves of several models on the same axes."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for color, (model, scores) in zip(PALETTE, scores_by_model.items()):
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{model}  (ROC-AUC {roc_auc_score(y_true, scores):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="random")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
           title=title or "ROC curves")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, name)


def plot_confusion(metrics: dict, name="confusion_matrix.png", title="Confusion matrix (test set)"):
    """2x2 confusion matrix annotated with counts."""
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.imshow(cm, cmap="Blues", norm=matplotlib.colors.LogNorm())  # log: counts span 5 orders
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=12,
                color="white" if i == j == 0 else "black")
    ax.set_xticks([0, 1], ["Predicted legit", "Predicted fraud"])
    ax.set_yticks([0, 1], ["Actual legit", "Actual fraud"])
    ax.set_title(title)
    return _save(fig, name)


def plot_cost_curve(y_true, scores, amounts, chosen: float, name="threshold_cost.png"):
    """Business cost as a function of the decision threshold, with the chosen one marked."""
    thresholds, costs = cost_curve(y_true, scores, amounts)
    # Plot against threshold on a 0..1 axis (drop the artificial +inf first point).
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds[1:], costs[1:] / 1e3, color=PALETTE[0], lw=2)
    # Log scale: "flag everything" costs ~100x more than the optimum, which would
    # flatten the interesting region near the minimum on a linear axis.
    ax.set_yscale("log")
    ax.axvline(chosen, color=PALETTE[2], ls="--", label=f"chosen threshold = {chosen:.3f}")
    ax.set(xlabel="Decision threshold (fraud probability)", ylabel="Total cost ($ thousands, log scale)",
           title="Missed-fraud $ + review cost vs. threshold (validation set)", xlim=(0, 1))
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, name)


def plot_score_distribution(y_true, scores, threshold: float, name="score_distribution.png"):
    """Histogram of predicted probabilities for legit vs. fraud transactions."""
    y = np.asarray(y_true)
    s = np.asarray(scores)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 51)
    ax.hist(s[y == 0], bins=bins, alpha=0.6, color=PALETTE[0], label="legit", density=True)
    ax.hist(s[y == 1], bins=bins, alpha=0.6, color=PALETTE[2], label="fraud", density=True)
    ax.axvline(threshold, color="black", ls="--", lw=1, label="threshold")
    ax.set_yscale("log")  # legit rows pile up near 0; log scale keeps both visible
    ax.set(xlabel="Predicted fraud probability", ylabel="Density (log scale)",
           title="Score separation on the test set")
    ax.legend()
    return _save(fig, name)


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Turn {model_name: metrics_dict} into a tidy, sorted DataFrame."""
    cols = ["pr_auc", "roc_auc", "precision", "recall", "f1",
            "precision_at_90_recall", "fraud_amt_caught_pct", "net_savings", "train_seconds"]
    df = pd.DataFrame(results).T
    return df[[c for c in cols if c in df.columns]].astype(float).sort_values("pr_auc", ascending=False)
