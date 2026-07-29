import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from data import load_data, preprocess_data, split_data
from utils import encode_and_scale, print_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the fraud detection model.")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample this many rows from the dataset for a quicker run.")
    parser.add_argument("--cv", type=int, default=3, help="Number of CV folds for grid search.")
    parser.add_argument("--n-estimators", type=int, nargs="+", default=[100, 200], help="RandomForest n_estimators values.")
    parser.add_argument("--max-depth", type=int, nargs="+", default=[10, 20], help="RandomForest max_depth values.")
    return parser.parse_args()


def main(sample_size=None, cv=3, n_estimators=None, max_depth=None):
    df = load_data()
    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Using sample size: {sample_size}")

    df = preprocess_data(df)
    print("Loaded dataset with shape:", df.shape)
    print(df["is_fraud"].value_counts())

    X_train, X_test, y_train, y_test = split_data(df)

    X_train_proc, X_test_proc = encode_and_scale(X_train, X_test)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_proc, y_train)
    print("Resampled training shape:", X_resampled.shape)

    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    params = {
        "n_estimators": n_estimators or [100, 200],
        "max_depth": max_depth or [10, 20],
    }

    grid = GridSearchCV(model, param_grid=params, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_resampled, y_resampled)

    print("Best params:", grid.best_params_)
    y_pred = grid.predict(X_test_proc)
    print_metrics(y_test, y_pred)
    print("Test ROC AUC:", roc_auc_score(y_test, y_pred))


if __name__ == "__main__":
    args = parse_args()
    main(sample_size=args.sample_size, cv=args.cv, n_estimators=args.n_estimators, max_depth=args.max_depth)
