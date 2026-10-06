"""
explain.py — why did the model flag this transaction?

Pipeline stage 6 of the workflow.

A fraud analyst cannot act on "probability = 0.97". They need reasons:
"amount is 14x this card's normal, 6 purchases in the last hour, at 2 AM".
Regulators expect the same for automated decisions. This module produces:

  * per-transaction feature CONTRIBUTIONS (how much each feature pushed the score
    up or down, in log-odds units), and
  * a GLOBAL importance ranking (average absolute contribution over many rows).

For LightGBM we use its built-in TreeSHAP (`pred_contrib=True`): exact SHAP
values computed inside the library — no extra `shap` dependency needed.
For logistic regression, contributions are simply coefficient x scaled value.

One-hot columns (category_grocery_pos, category_travel ...) are summed back into
their parent feature ("category") so explanations read in business terms.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud import config


def _parent_feature(column: str) -> str:
    """Map an encoded column back to its original feature.
    'category_shopping_net' -> 'category', 'gender_F' -> 'gender', 'amt' -> 'amt'."""
    for cat in config.CATEGORICAL_FEATURES:
        if column.startswith(cat + "_"):
            return cat
    return column


def contributions(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Per-row, per-feature contribution to the model's log-odds fraud score.

    Returns a DataFrame with one column per ORIGINAL feature (30 columns) plus
    'bias' (the model's baseline score before looking at any feature).
    Row sums equal the model's raw log-odds output.
    """
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    # Apply exactly the same preprocessing the model saw (SMOTE is training-only,
    # so it is correctly NOT applied here).
    Xt = prep.transform(X)
    names = list(prep.get_feature_names_out())

    if hasattr(clf, "booster_"):  # LightGBM -> exact TreeSHAP values
        raw = clf.booster_.predict(Xt, pred_contrib=True)  # last column = bias term
        contrib = pd.DataFrame(raw[:, :-1], columns=names, index=X.index)
        bias = raw[:, -1]
    else:  # linear model -> coefficient x feature value
        contrib = pd.DataFrame(Xt * clf.coef_[0], columns=names, index=X.index)
        bias = np.full(len(X), clf.intercept_[0])

    # Sum one-hot columns into their parent feature: transpose so columns become
    # rows, group those rows by parent name, sum, transpose back.
    grouped = contrib.T.groupby(_parent_feature).sum().T
    grouped["bias"] = bias
    return grouped[config.FEATURES + ["bias"]]


def reason_codes(pipeline, X: pd.DataFrame, top_k: int = 3) -> list[list[str]]:
    """Top-k human-readable reasons that pushed each transaction TOWARDS fraud.

    Example output for one row:
        ["amt_sum_24h = 1,843.2 (+3.12)", "is_night = 1 (+1.40)", "amt = 912.4 (+1.05)"]
    """
    contrib = contributions(pipeline, X).drop(columns="bias")
    reasons = []
    for idx, row in contrib.iterrows():
        top = row[row > 0].sort_values(ascending=False).head(top_k)
        reasons.append([f"{feat} = {_fmt(X.at[idx, feat])} (+{val:.2f})" for feat, val in top.items()])
    return reasons


def _fmt(value) -> str:
    """Pretty-print a feature value for a reason string."""
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return "n/a" if np.isnan(value) else f"{value:,.1f}"
    return str(value)


def global_importance(pipeline, X: pd.DataFrame) -> pd.Series:
    """Mean |contribution| per feature — how much each feature moves scores on average."""
    contrib = contributions(pipeline, X).drop(columns="bias")
    return contrib.abs().mean().sort_values(ascending=False)


def plot_global_importance(importance: pd.Series, name="feature_importance.png", top_n: int = 20):
    """Horizontal bar chart of the top-N most influential features."""
    top = importance.head(top_n)[::-1]  # reverse so the biggest bar is on top
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(top) + 1))
    ax.barh(top.index, top.values, color="#2563eb")
    ax.set(xlabel="Mean |SHAP contribution| (log-odds)",
           title=f"Top {len(top)} features driving fraud predictions")
    ax.grid(axis="x", alpha=0.3)
    config.ensure_dirs()
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
