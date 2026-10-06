"""
anomaly.py — unsupervised anomaly detection baseline (Isolation Forest).

Pipeline stage 7 of the workflow.

Supervised models need labelled fraud, which in real life arrives weeks late
(when the customer disputes a charge) and only covers fraud patterns already seen.
Anomaly detection needs NO labels: it learns what "normal" looks like and flags
anything that deviates. It is a useful safety net for brand-new fraud patterns.

How Isolation Forest works:
    It builds many random trees, each repeatedly splitting the data on a random
    feature at a random value. Outliers sit in sparse regions, so they get
    "isolated" in very few splits; normal points need many. The average path
    length becomes the anomaly score — shorter path = more anomalous.

We fit it on LEGIT training transactions only (the "normal" behaviour) and then
compare how well its score ranks fraud vs. the supervised models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from fraud import config
from fraud.preprocessing import build_preprocessor


def build_isolation_forest() -> Pipeline:
    """Preprocessing + Isolation Forest in one sklearn Pipeline."""
    return Pipeline([
        ("prep", build_preprocessor()),
        ("iforest", IsolationForest(
            n_estimators=300,          # more trees = more stable scores
            max_samples=4_096,         # each tree is built on a 4K-row sample (fast)
            contamination="auto",      # we only use the score, not a hard label
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def fit_isolation_forest(train: pd.DataFrame, max_rows: int = 300_000) -> Pipeline:
    """Fit on (a sample of) legitimate training transactions only."""
    normal = train[train[config.TARGET] == 0]
    # Isolation Forest only looks at small sub-samples anyway, so 300K rows is
    # plenty to learn "normal" while keeping preprocessing fast.
    if len(normal) > max_rows:
        normal = normal.sample(max_rows, random_state=config.RANDOM_STATE)
    model = build_isolation_forest()
    model.fit(normal[config.FEATURES])
    return model


def anomaly_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous.

    sklearn's score_samples returns HIGHER for more NORMAL points, so we negate it
    to get a score that, like a fraud probability, grows with suspiciousness.
    (It is a ranking score, not a probability — fine for PR-AUC / ROC-AUC.)
    """
    return -model.score_samples(X[config.FEATURES])
