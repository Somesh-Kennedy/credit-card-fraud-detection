"""
preprocessing.py — turn the 30 feature columns into a purely numeric matrix.

Pipeline stage 3 of the workflow.

Models (and SMOTE) need a numeric matrix with no missing values. A scikit-learn
ColumnTransformer applies a different recipe to each type of column and — just as
importantly — *learns* its statistics (medians, means, std-devs, category lists)
on the training data only, then re-applies them unchanged to validation / test.
Putting it inside the model Pipeline guarantees that ordering automatically.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud import config


def build_preprocessor() -> ColumnTransformer:
    """Numeric: median-impute -> standardise.  Categorical: one-hot encode."""

    numeric = Pipeline(steps=[
        # NaNs only appear for "no history yet" rows (a card's first purchase has no
        # previous transaction, no historical mean ...). The median is a neutral,
        # outlier-robust stand-in. Only ~0.1% of rows are affected.
        ("impute", SimpleImputer(strategy="median")),
        # Zero-mean / unit-variance scaling. Required for logistic regression and
        # for SMOTE (which measures *distances* between frauds — an unscaled $ amount
        # would dominate every other feature). Tree models simply ignore it.
        ("scale", StandardScaler()),
    ])

    categorical = OneHotEncoder(
        # A category never seen in training (e.g. a new merchant type in production)
        # becomes all-zeros instead of crashing the scorer.
        handle_unknown="ignore",
        # Dense output: SMOTE and LightGBM's SHAP both want a plain numpy array, and
        # with only 14 + 2 categories the matrix stays small.
        sparse_output=False,
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, config.NUMERIC_FEATURES),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        # Any column not listed (cc_num, merchant name, lat/long ...) is dropped so it
        # can never accidentally sneak into the model.
        remainder="drop",
        # Feature names like "category_shopping_net" instead of "cat__category_shopping_net".
        verbose_feature_names_out=False,
    )
