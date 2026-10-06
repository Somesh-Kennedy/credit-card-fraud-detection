"""
fraud — an end-to-end credit card fraud detection package.

The package is split into small single-purpose modules so each stage of the
pipeline can be read, tested and reused on its own:

    config         -> every path, date, column name and hyper-parameter in one place
    data           -> load the raw CSV, clean it, cache it as Parquet, split it by time
    features       -> turn raw transactions into the 30 model features (leak-free)
    preprocessing  -> sklearn ColumnTransformer that imputes / scales / one-hot encodes
    models         -> the candidate models (with and without SMOTE) as imblearn Pipelines
    evaluate       -> metrics, threshold tuning and evaluation plots
    explain        -> SHAP-style "reason codes" for every prediction
    anomaly        -> unsupervised Isolation Forest baseline (no labels needed)
    segmentation   -> K-Means customer segmentation on card-level behaviour
    eda            -> reusable exploratory-analysis figures (used by the notebook + README)
    train          -> the command-line entry point that runs the whole pipeline
    predict        -> load the trained model and score new transactions
"""

# Exposing the version here lets other code (e.g. the dashboard) display it.
__version__ = "1.0.0"
