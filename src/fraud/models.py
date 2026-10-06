"""
models.py — the candidate models compared in the experiment.

Pipeline stage 4 of the workflow.

The core question of this project is *how to deal with class imbalance*:
only ~0.58% of transactions are fraud, so a model that predicts "never fraud"
is 99.4% accurate and completely useless. We compare three standard strategies
on two model families:

    strategy               | logistic regression | LightGBM (gradient boosting)
    -----------------------+---------------------+-----------------------------
    do nothing             | logreg              | lgbm
    re-weight the classes  |   (not shown)       | lgbm_weighted
    SMOTE oversampling     | logreg_smote        | lgbm_smote

Why imblearn's Pipeline (not sklearn's)?
    SMOTE must ONLY ever touch the training data. If you oversample before
    splitting, synthetic frauds built from test-set frauds leak into training and
    the scores become fiction. imblearn.Pipeline runs samplers during .fit() and
    automatically SKIPS them during .predict(), so this mistake becomes impossible.
"""

from __future__ import annotations

from dataclasses import dataclass

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from fraud import config
from fraud.preprocessing import build_preprocessor


@dataclass(frozen=True)
class ModelSpec:
    """Recipe for one experiment: which classifier, and whether to use SMOTE."""
    name: str
    family: str          # "logreg" or "lgbm" (decides which classifier to build)
    smote: bool = False  # oversample the minority class with SMOTE?
    weighted: bool = False  # up-weight frauds inside the loss function instead?
    description: str = ""


# The experiments, in the order they appear in reports/model_comparison.csv.
MODEL_SPECS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in [
        ModelSpec("logreg", "logreg",
                  description="Logistic regression, no imbalance handling (baseline)"),
        ModelSpec("logreg_smote", "logreg", smote=True,
                  description="Logistic regression + SMOTE"),
        ModelSpec("lgbm", "lgbm",
                  description="LightGBM, no imbalance handling"),
        ModelSpec("lgbm_weighted", "lgbm", weighted=True,
                  description="LightGBM with fraud class up-weighted in the loss"),
        ModelSpec("lgbm_smote", "lgbm", smote=True,
                  description="LightGBM + SMOTE"),
    ]
}


def _make_classifier(spec: ModelSpec, pos_weight: float):
    """Instantiate the bare classifier for a spec."""
    if spec.family == "logreg":
        # Linear baseline: fast, interpretable, and shows how much the non-linear
        # model actually adds. lbfgs handles ~1M rows comfortably; max_iter is raised
        # because the default (100) often stops before converging on big data.
        return LogisticRegression(max_iter=2_000, C=1.0)

    if spec.family == "lgbm":
        return LGBMClassifier(
            n_estimators=600,        # number of boosted trees
            learning_rate=0.05,      # small steps + many trees = better generalisation
            num_leaves=63,           # tree complexity (LightGBM grows leaf-wise)
            min_child_samples=50,    # a leaf needs >= 50 rows -> no memorising single frauds
            subsample=0.8,           # each tree sees a random 80% of rows ...
            subsample_freq=1,        # ... re-drawn for every tree (bagging)
            colsample_bytree=0.8,    # ... and a random 80% of the features
            reg_lambda=1.0,          # L2 penalty on leaf values
            # Class weighting: multiply the loss of every fraud by `pos_weight`.
            # We use sqrt(neg/pos) (~13x) rather than the full ratio (~170x): the full
            # ratio massively over-predicts fraud and wrecks precision.
            scale_pos_weight=pos_weight if spec.weighted else 1.0,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,               # use every CPU core
            verbose=-1,              # silence LightGBM's per-iteration logging
        )

    raise ValueError(f"unknown model family: {spec.family}")


def build_pipeline(spec: ModelSpec, pos_weight: float = 1.0) -> Pipeline:
    """Assemble preprocess -> [SMOTE] -> classifier for one experiment.

    pos_weight: the class weight used by 'weighted' specs (computed from the
                training labels in train.py).
    """
    steps = [("prep", build_preprocessor())]

    if spec.smote:
        # SMOTE (Synthetic Minority Over-sampling TEchnique) creates new, artificial
        # frauds by picking a real fraud, picking one of its k nearest fraud
        # neighbours, and placing a new point at a random spot on the line between
        # them. Unlike simply duplicating frauds, this fills in the fraud region of
        # feature space and reduces overfitting to the exact training examples.
        # It runs AFTER preprocessing because it needs numeric, scaled, NaN-free data.
        steps.append(("smote", SMOTE(
            sampling_strategy=config.SMOTE_RATIO,   # frauds -> 10% of legit count
            k_neighbors=config.SMOTE_K_NEIGHBORS,
            random_state=config.RANDOM_STATE,
        )))

    steps.append(("clf", _make_classifier(spec, pos_weight)))
    return Pipeline(steps)
