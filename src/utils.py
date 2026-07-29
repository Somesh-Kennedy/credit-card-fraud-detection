import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


def encode_and_scale(X_train: pd.DataFrame, X_val: pd.DataFrame):
    numeric_cols = [col for col in X_train.columns if X_train[col].dtype in ["int64", "float64"]]
    categorical_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    X_train_num = scaler.fit_transform(X_train[numeric_cols])
    X_val_num = scaler.transform(X_val[numeric_cols])

    X_train_cat = encoder.fit_transform(X_train[categorical_cols])
    X_val_cat = encoder.transform(X_val[categorical_cols])

    X_train_proc = np.hstack([X_train_num, X_train_cat])
    X_val_proc = np.hstack([X_val_num, X_val_cat])
    return X_train_proc, X_val_proc


def print_metrics(y_true, y_pred):
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
