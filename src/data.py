import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "credit_card_transactions.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek

    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["age"] = (pd.Timestamp.now() - df["dob"]).dt.days // 365

    df["amt_log"] = df["amt"].replace(0, 1).apply(lambda x: np.log1p(x))

    df["merchant_category"] = df["merchant"].astype(str).str.extract(r"fraud_(.*)$")
    df["merchant_category"] = df["merchant_category"].fillna("unknown")

    df["zip_str"] = df["zip"].astype(str).str.zfill(5)
    df["state"] = df["state"].astype(str)
    df["gender"] = df["gender"].fillna("U")

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    features = df.drop(columns=["is_fraud", "merchant", "trans_date_trans_time", "dob", "trans_num", "unix_time", "cc_num", "first", "last", "street", "city", "job", "merch_zipcode"])
    target = df["is_fraud"].astype(int)
    return train_test_split(features, target, test_size=test_size, random_state=random_state, stratify=target)
