"""
test_features.py — correctness tests for feature engineering.

The most important test here is `test_no_future_leakage`: if a feature for a
transaction ever used information from LATER transactions, the model would look
brilliant offline and fail in production. We prove it cannot happen.
"""

import numpy as np
import pandas as pd

from fraud import config
from fraud.features import build_features, haversine_km


def test_haversine_known_distance():
    # New York -> Los Angeles is ~3,936 km along the great circle.
    d = haversine_km(40.7128, -74.0060, 34.0522, -118.2437)
    assert abs(d - 3936) < 10


def test_haversine_zero_distance():
    assert haversine_km(10.0, 20.0, 10.0, 20.0) == 0


def test_produces_all_30_features_without_inf(featured):
    # Every advertised feature exists ...
    assert all(f in featured.columns for f in config.FEATURES)
    assert len(config.FEATURES) == 30
    # ... and none contains +/-inf (NaN is allowed: it means "no history yet").
    numeric = featured[config.NUMERIC_FEATURES].to_numpy(dtype=float)
    assert not np.isinf(numeric).any()


def test_preserves_row_order_and_index(transactions):
    shuffled = transactions.sample(frac=1, random_state=1)
    out = build_features(shuffled)
    assert out.index.equals(shuffled.index)


def test_velocity_windows_exact():
    # One card, three purchases: t=0, t=+30min, t=+2h.
    df = pd.DataFrame({
        "cc_num": [1, 1, 1],
        "trans_time": pd.to_datetime(["2019-01-01 00:00", "2019-01-01 00:30", "2019-01-01 02:00"]),
        "amt": [10.0, 20.0, 30.0],
        "category": pd.Categorical(["a", "a", "b"]), "merchant": pd.Categorical(["x", "y", "x"]),
        "gender": pd.Categorical(["F"] * 3), "dob": pd.to_datetime(["1990-01-01"] * 3),
        "city_pop": [100] * 3, "lat": [40.0] * 3, "long": [-75.0] * 3,
        "merch_lat": [40.1] * 3, "merch_long": [-75.1] * 3, "is_fraud": [0, 0, 0],
    })
    out = build_features(df)
    # 1-hour window includes the current transaction plus anything in the previous hour.
    assert out["txn_count_1h"].tolist() == [1, 2, 1]
    assert out["amt_sum_1h"].tolist() == [10.0, 30.0, 30.0]
    assert out["txn_count_24h"].tolist() == [1, 2, 3]
    # Historical mean EXCLUDES the current row.
    assert np.isnan(out["card_amt_mean_hist"].iloc[0])
    assert out["card_amt_mean_hist"].iloc[1] == 10.0
    assert out["card_amt_mean_hist"].iloc[2] == 15.0
    # New-merchant / new-category flags.
    assert out["is_new_merchant"].tolist() == [1, 1, 0]
    assert out["is_new_category"].tolist() == [1, 0, 1]
    # Seconds since previous transaction.
    assert out["secs_since_prev_txn"].iloc[1] == 1800


def test_no_future_leakage(transactions):
    """Features computed with the full history must equal features computed when the
    future has not happened yet (data truncated at a cut-off date)."""
    cutoff = pd.Timestamp("2019-09-01")
    full = build_features(transactions)
    past_only = build_features(transactions[transactions["trans_time"] < cutoff])

    before = full.loc[past_only.index, config.NUMERIC_FEATURES]
    pd.testing.assert_frame_equal(before, past_only[config.NUMERIC_FEATURES], check_dtype=False)


def test_labels_do_not_affect_features(transactions):
    """Flipping every fraud label must not change any feature (no target leakage)."""
    flipped = transactions.assign(is_fraud=1 - transactions["is_fraud"])
    a = build_features(transactions)[config.NUMERIC_FEATURES]
    b = build_features(flipped)[config.NUMERIC_FEATURES]
    pd.testing.assert_frame_equal(a, b)
