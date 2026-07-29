import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from data import load_data, preprocess_data, split_data
from utils import encode_and_scale, print_metrics


def main():
    df = load_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    X_train_proc, X_test_proc = encode_and_scale(X_train, X_test)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_proc, y_train)

    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    params = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
    }

    grid = GridSearchCV(model, param_grid=params, cv=3, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_resampled, y_resampled)

    print("Best params:", grid.best_params_)
    y_pred = grid.predict(X_test_proc)
    print_metrics(y_test, y_pred)
    print("Test ROC AUC:", roc_auc_score(y_test, y_pred))


if __name__ == "__main__":
    main()
