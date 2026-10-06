"""
conftest.py — shared pytest fixtures.

Tests must be fast and must not depend on the 340 MB Kaggle file, so we generate a
small synthetic dataset with the same columns the cleaned data has. pytest
automatically makes any fixture defined here available to every test file.
"""

import numpy as np
import pandas as pd
import pytest

from fraud import config

CATEGORIES = ["grocery_pos", "shopping_net", "gas_transport", "travel", "misc_net"]


def make_transactions(n_cards: int = 5, n_per_card: int = 300, fraud_rate: float = 0.05,
                      seed: int = 0) -> pd.DataFrame:
    """Random but realistic-looking cleaned transactions spanning 2019-01 .. 2020-06."""
    rng = np.random.default_rng(seed)
    rows = []
    start, end = pd.Timestamp("2019-01-01"), pd.Timestamp("2020-06-20")
    span = (end - start).total_seconds()
    for card in range(n_cards):
        # Each card lives somewhere in the US and has its own date of birth / city size.
        lat, lon = rng.uniform(30, 45), rng.uniform(-120, -75)
        times = start + pd.to_timedelta(np.sort(rng.uniform(0, span, n_per_card)), unit="s")
        for t in times:
            rows.append({
                "cc_num": 1_000_000 + card,
                "trans_time": t,
                "merchant": f"m{rng.integers(0, 40)}",
                "category": rng.choice(CATEGORIES),
                "amt": float(np.round(rng.lognormal(3.5, 1.0), 2)),
                "gender": "F" if card % 2 else "M",
                "dob": pd.Timestamp("1980-01-01") + pd.Timedelta(days=int(card * 1000)),
                "city_pop": int(1000 * (card + 1)),
                "state": "TX",
                "lat": lat, "long": lon,
                "merch_lat": lat + rng.normal(0, 0.5), "merch_long": lon + rng.normal(0, 0.5),
                "is_fraud": int(rng.random() < fraud_rate),
            })
    df = pd.DataFrame(rows)
    for col in ["merchant", "category", "gender", "state"]:
        df[col] = df[col].astype("category")
    return df


@pytest.fixture
def transactions() -> pd.DataFrame:
    """Small synthetic cleaned dataset (1,500 rows)."""
    return make_transactions()


@pytest.fixture
def featured(transactions) -> pd.DataFrame:
    """The synthetic dataset with the 30 features added."""
    from fraud.features import build_features
    return build_features(transactions)


__all__ = ["make_transactions", "config"]
