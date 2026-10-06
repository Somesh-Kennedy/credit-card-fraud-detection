"""
test_pipeline.py — tests for splitting, models and evaluation logic.
"""

import numpy as np
import pytest

from fraud import config, data, evaluate
from fraud.models import MODEL_SPECS, build_pipeline


# --------------------------------------------------------------------- data
def test_time_split_is_a_chronological_partition(featured):
    train, valid, test = data.time_split(featured)
    # Every row lands in exactly one split ...
    assert len(train) + len(valid) + len(test) == len(featured)
    # ... and the splits are strictly ordered in time (train < valid < test).
    assert train["trans_time"].max() < valid["trans_time"].min()
    assert valid["trans_time"].max() < test["trans_time"].min()


# --------------------------------------------------------------------- models
@pytest.mark.parametrize("name", list(MODEL_SPECS))
def test_every_model_trains_and_predicts_probabilities(featured, name):
    train, valid, _ = data.time_split(featured)
    pipe = build_pipeline(MODEL_SPECS[name], pos_weight=3.0)
    pipe.fit(train[config.FEATURES], train[config.TARGET])
    proba = pipe.predict_proba(valid[config.FEATURES])[:, 1]
    assert proba.shape == (len(valid),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_smote_only_applies_during_fit(featured):
    """imblearn skips samplers at predict time: output length must equal input length."""
    train, valid, _ = data.time_split(featured)
    pipe = build_pipeline(MODEL_SPECS["logreg_smote"])
    pipe.fit(train[config.FEATURES], train[config.TARGET])
    assert len(pipe.predict(valid[config.FEATURES])) == len(valid)


# --------------------------------------------------------------------- evaluation
def test_metrics_on_perfect_scores():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    m = evaluate.compute_metrics(y, s, amounts=np.ones(5) * 100, threshold=0.5)
    assert m["pr_auc"] == 1.0 and m["roc_auc"] == 1.0
    assert m["precision"] == 1.0 and m["recall"] == 1.0
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 0, 0, 3)
    assert m["fraud_amt_caught_pct"] == 1.0


def test_cost_threshold_catches_expensive_fraud():
    # Fraud is worth $1,000 each; a false alarm costs $10 -> flag both frauds even
    # though that also flags one legit transaction scored in between.
    y = np.array([0, 0, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.5, 0.6, 0.9])
    amounts = np.array([50, 50, 1000, 50, 1000])
    thr = evaluate.best_cost_threshold(y, s, amounts, fp_cost=10)
    assert thr == 0.5


def test_cost_threshold_ignores_cheap_fraud():
    # The only fraud is $1 and ranks BELOW a legit row, so catching it means also
    # paying a $10 review for the legit one. Missing it ($1) is cheaper -> flag nothing.
    y = np.array([0, 1])
    s = np.array([0.9, 0.5])
    thr = evaluate.best_cost_threshold(y, s, np.array([5, 1]), fp_cost=10)
    assert thr == np.inf
