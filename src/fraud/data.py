"""
data.py — load, clean, cache and split the raw transaction data.

Pipeline stage 1 of the workflow:

    dataset/credit_card_transactions.csv
        -> load_raw()        read with explicit dtypes
        -> clean()           parse dates, drop junk + PII, sort by time
        -> data/transactions_clean.parquet  (cached; ~10x faster to reload)
        -> time_split()      train / validation / test by date
"""

from __future__ import annotations

import pandas as pd

from fraud import config

# Reading 1.3M rows is much faster and uses less memory when pandas does not have
# to guess each column's type. "category" dtype stores repeated strings (e.g. the
# 14 merchant categories) once and keeps small integer codes per row.
RAW_DTYPES = {
    "cc_num": "int64",
    "merchant": "category",
    "category": "category",
    "amt": "float64",
    "gender": "category",
    "city": "category",
    "state": "category",
    "zip": "int32",
    "lat": "float64",
    "long": "float64",
    "city_pop": "int64",
    "job": "category",
    "merch_lat": "float64",
    "merch_long": "float64",
    "is_fraud": "int8",
}


def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    """Read the raw Kaggle CSV exactly as downloaded."""
    # Fail early with a helpful message instead of a long pandas traceback.
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}.\n"
            "Download 'credit_card_transactions.csv' from Kaggle "
            "(Credit Card Transactions Dataset) and place it in the dataset/ folder."
        )
    return pd.read_csv(path, dtype=RAW_DTYPES)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw frame into a tidy, typed, time-ordered frame.

    Steps (each one explained inline):
      1. drop the leftover pandas index column and PII
      2. drop columns that are redundant or broken
      3. parse timestamps / date of birth
      4. clean the "fraud_" prefix out of merchant names
      5. sort chronologically per card (required for the velocity features)
    """
    df = df.copy()

    # 1. "Unnamed: 0" is just the row number someone saved with df.to_csv(); it is
    #    not information. Names/street/transaction id are PII (see config.PII_COLS).
    df = df.drop(columns=["Unnamed: 0", *config.PII_COLS], errors="ignore")

    # 2. 'unix_time' is shifted by exactly 7 years relative to the human-readable
    #    timestamp in this synthetic dataset (2012 vs 2019) so it is unreliable;
    #    'merch_zipcode' is missing for ~15% of rows and duplicates merch_lat/long.
    df = df.drop(columns=["unix_time", "merch_zipcode"], errors="ignore")

    # 3. Real datetime types unlock .dt.hour, time-window rolling, date filtering ...
    df[config.TIME_COL] = pd.to_datetime(df.pop("trans_date_trans_time"))
    df["dob"] = pd.to_datetime(df["dob"])

    # 4. Every merchant in the synthetic generator is named "fraud_<company>".
    #    The prefix is meaningless (it appears on legit rows too) and would
    #    confuse anyone reading charts, so strip it.
    df["merchant"] = (
        df["merchant"].astype(str).str.removeprefix("fraud_").astype("category")
    )

    # 5. The file is *almost* but not perfectly time-ordered. Every per-card
    #    "history" feature assumes rows are in chronological order, so sort by
    #    card then time. A stable sort (mergesort) keeps the original order for
    #    identical timestamps so results are deterministic.
    df = df.sort_values([config.CARD_COL, config.TIME_COL], kind="mergesort")
    return df.reset_index(drop=True)


def load_clean(use_cache: bool = True) -> pd.DataFrame:
    """Return the cleaned dataset, building + caching it on first use.

    Parquet is a compressed, typed, columnar format: the 340 MB CSV becomes a
    ~60 MB file that loads in ~1 second and keeps the dtypes we set above.
    """
    config.ensure_dirs()
    if use_cache and config.CLEAN_PARQUET.exists():
        return pd.read_parquet(config.CLEAN_PARQUET)

    df = clean(load_raw())
    df.to_parquet(config.CLEAN_PARQUET, index=False)
    return df


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / validation / test by transaction date (see config.py).

    Returns copies so later modifications to one split never leak into another.
    """
    t = df[config.TIME_COL]
    train = df[t < config.VALID_START]
    valid = df[(t >= config.VALID_START) & (t < config.TEST_START)]
    test = df[t >= config.TEST_START]

    # Sanity check: the three splits must partition the data exactly.
    assert len(train) + len(valid) + len(test) == len(df)
    return train.copy(), valid.copy(), test.copy()


def describe_split(name: str, part: pd.DataFrame) -> str:
    """One-line human-readable summary used in logs, e.g. for the training run."""
    t = part[config.TIME_COL]
    return (
        f"{name:<10} {len(part):>9,} rows | "
        f"{int(part[config.TARGET].sum()):>5,} frauds ({part[config.TARGET].mean():.3%}) | "
        f"{t.min():%Y-%m-%d} -> {t.max():%Y-%m-%d}"
    )
