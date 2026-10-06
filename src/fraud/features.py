"""
features.py — engineer the 30 model features from raw transactions.

Pipeline stage 2 of the workflow.

The raw data only tells us *what* happened in a single transaction (amount,
merchant category, time, location). Fraud, however, is mostly visible in
*context*: is this amount unusual **for this card**? Has the card suddenly made
8 purchases in an hour? Is the merchant 3,000 km from the last one 10 minutes ago?
This module adds that context as features.

The golden rule: NO LEAKAGE
---------------------------
Every per-card feature for a transaction at time t is computed using ONLY that
card's transactions at or before t — never later ones, and never the fraud label.
That is exactly the information a real-time scoring system would have when the
card is swiped. (tests/test_features.py verifies this by recomputing features on
a truncated history and checking they do not change.)

Because the features only look backwards and never use labels, it is safe to
compute them once on the full time-ordered dataset and split afterwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud import config

# Earth's mean radius, used by the haversine distance formula.
EARTH_RADIUS_KM = 6371.0

# Sliding windows for the velocity features: name -> length in seconds.
WINDOWS = {"1h": 3_600, "24h": 86_400, "7d": 7 * 86_400}

# Floor used when computing travel speed. Two swipes in the same second would
# give "infinite" speed; flooring the gap at one minute keeps the value finite
# while still making "100 km in 1 minute" look as suspicious as it should.
MIN_GAP_SECONDS = 60

# Minimum number of past transactions before we trust a card's standard deviation.
# With only 2-3 purchases the std can be ~$0.05, which makes z-scores explode into
# the thousands; below this threshold the z-score is left missing instead.
MIN_HISTORY_FOR_STD = 5


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two (lat, lon) points, in kilometres.

    Works element-wise on numpy arrays / pandas Series, so a whole column of
    1.3M distances is computed in one vectorised call (no Python loop).
    """
    # Trigonometric functions expect radians, the data is in degrees.
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Standard haversine formula: a is the square of half the chord length.
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _window_count_and_sum(card_codes: np.ndarray, secs: np.ndarray,
                          amt: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """For every row: how many transactions (and how many dollars) did the same
    card make in the trailing `window` seconds, *including* the current one?

    How it works (fast, exact, no Python loop):
      * Rows are sorted by (card, time). We build one sortable key per row:
            key = card_code * 1e10 + seconds
        The 1e10 offset (~317 years) is larger than any time span in the data,
        so keys of different cards can never fall in the same window.
      * For each row, np.searchsorted finds the first row whose key is
        >= key - window  -> that row is where the window starts.
      * count = current position - start position + 1
      * sum   = cumulative_sum[current] - cumulative_sum[start - 1]
    This turns what would be an O(n * window) loop into O(n log n).
    """
    key = card_codes.astype(np.int64) * 10**10 + secs
    start = np.searchsorted(key, key - window, side="left")
    idx = np.arange(len(key))
    count = idx - start + 1
    # Prepend 0 so that cumsum[i + 1] - cumsum[start] is the sum of rows start..i.
    cumsum = np.concatenate([[0.0], np.cumsum(amt)])
    total = cumsum[idx + 1] - cumsum[start]
    return count, total


def _expanding_prior_mean(df: pd.DataFrame, keys: list[str]) -> tuple[pd.Series, pd.Series]:
    """Mean amount of all *previous* transactions within each group (e.g. per card).

    Returns (prior_count, prior_mean). The current row is excluded by subtracting
    it from the running total — so a card's very first transaction has
    prior_count = 0 and prior_mean = NaN (there is no history yet).
    """
    g = df.groupby(keys, sort=False, observed=True)["amt"]
    prior_count = g.cumcount()                      # 0, 1, 2, ... within each group
    prior_sum = g.cumsum() - df["amt"]              # running total minus current row
    prior_mean = prior_sum / prior_count.replace(0, np.nan)  # avoid 0/0
    return prior_count, prior_mean


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 30 model features (see config.FEATURES) to a cleaned transaction frame.

    Input : output of data.clean() — must contain cc_num, trans_time, amt, category,
            gender, dob, city_pop, lat/long, merch_lat/merch_long, merchant.
    Output: the same rows (same index, same order as the input) with new columns.
    """
    # Remember the caller's row order, then sort by card + time because every
    # "history" feature below depends on chronological order within a card.
    original_index = df.index
    out = df.sort_values([config.CARD_COL, config.TIME_COL], kind="mergesort").copy()
    ts = out[config.TIME_COL]

    # ------------------------------------------------------------------
    # Transaction features
    # ------------------------------------------------------------------
    # Amounts are heavily right-skewed ($1 coffee ... $28,000 purchase). log1p
    # compresses the tail so linear models are not dominated by a few huge values.
    out["log_amt"] = np.log1p(out["amt"])

    # ------------------------------------------------------------------
    # Customer features
    # ------------------------------------------------------------------
    # Age at transaction time (not "today") — this is what the bank would know.
    out["age"] = (ts - out["dob"]).dt.days / 365.25
    # City population spans 23 people ... 2.9M, so again use the log.
    out["log_city_pop"] = np.log1p(out["city_pop"])

    # ------------------------------------------------------------------
    # Time features
    # ------------------------------------------------------------------
    out["hour"] = ts.dt.hour
    # Cyclical encoding: hour 23 and hour 0 are 1 hour apart, but as plain numbers
    # they look 23 apart. Mapping the hour onto a circle (sin, cos) fixes that.
    angle = 2 * np.pi * out["hour"] / 24
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)
    out["day_of_week"] = ts.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    # EDA (notebooks/01_eda.ipynb) shows fraud concentrates between 22:00 and 04:00.
    out["is_night"] = ((out["hour"] >= 22) | (out["hour"] < 4)).astype("int8")

    # ------------------------------------------------------------------
    # Geographic features
    # ------------------------------------------------------------------
    # How far from home is the merchant?
    out["cust_merch_dist_km"] = haversine_km(
        out["lat"], out["long"], out["merch_lat"], out["merch_long"]
    )

    # Previous transaction of the same card (shift(1) within each card group).
    card_groups = out.groupby(config.CARD_COL, sort=False)
    prev_time = card_groups[config.TIME_COL].shift(1)
    prev_lat = card_groups["merch_lat"].shift(1)
    prev_long = card_groups["merch_long"].shift(1)

    # Time since the card was last used. NaN for the card's first transaction.
    out["secs_since_prev_txn"] = (ts - prev_time).dt.total_seconds()
    # Distance between this merchant and the previous merchant ...
    out["km_from_prev_txn"] = haversine_km(prev_lat, prev_long,
                                           out["merch_lat"], out["merch_long"])
    # ... and the speed the card-holder would have needed to travel between them.
    # A physically impossible speed suggests the card is being used in two places.
    gap_hours = out["secs_since_prev_txn"].clip(lower=MIN_GAP_SECONDS) / 3600
    out["speed_kmh_from_prev"] = out["km_from_prev_txn"] / gap_hours

    # ------------------------------------------------------------------
    # Velocity features (trailing windows per card)
    # ------------------------------------------------------------------
    # Fraudsters typically "test" a stolen card then spend fast, so a burst of
    # transactions / dollars in a short window is one of the strongest signals.
    # pd.factorize on the sorted card column gives increasing integer codes 0..n_cards-1.
    card_codes, _ = pd.factorize(out[config.CARD_COL], sort=True)
    secs = ((ts - ts.min()).dt.total_seconds()).to_numpy().astype(np.int64)
    amt = out["amt"].to_numpy()
    for name, window in WINDOWS.items():
        count, total = _window_count_and_sum(card_codes, secs, amt, window)
        out[f"txn_count_{name}"] = count
        out[f"amt_sum_{name}"] = total

    # ------------------------------------------------------------------
    # Behavioural-baseline features (is this normal FOR THIS CARD?)
    # ------------------------------------------------------------------
    # A $900 purchase is ordinary for some customers and wildly unusual for others.
    # Comparing each amount to the card's own history captures that.
    prior_n, prior_mean = _expanding_prior_mean(out, [config.CARD_COL])
    out["card_txn_count_hist"] = prior_n
    out["card_amt_mean_hist"] = prior_mean

    # Prior standard deviation from running sums: var = E[x^2] - E[x]^2,
    # then Bessel's correction n / (n - 1) for an unbiased sample estimate.
    sq = out["amt"] ** 2
    prior_sq_sum = sq.groupby(out[config.CARD_COL], sort=False).cumsum() - sq
    n = prior_n.replace(0, np.nan)
    prior_var = (prior_sq_sum / n - prior_mean**2).clip(lower=0) * n / (n - 1)
    out["card_amt_std_hist"] = np.sqrt(prior_var.where(prior_n >= MIN_HISTORY_FOR_STD))

    # z-score: how many standard deviations above this card's usual amount?
    std = out["card_amt_std_hist"].replace(0, np.nan)
    out["amt_zscore_card"] = (out["amt"] - out["card_amt_mean_hist"]) / std
    out["amt_to_card_mean_ratio"] = out["amt"] / out["card_amt_mean_hist"]

    # Same idea but within the category: $300 at "travel" is normal, $300 at
    # "grocery_pos" for a card that usually spends $40 on groceries is not.
    _, prior_cat_mean = _expanding_prior_mean(out, [config.CARD_COL, "category"])
    out["amt_to_card_cat_mean_ratio"] = out["amt"] / prior_cat_mean

    # First time this card has ever used this merchant / this category?
    # cumcount() == 0 marks the first row of each (card, merchant) group.
    out["is_new_merchant"] = (
        out.groupby([config.CARD_COL, "merchant"], sort=False, observed=True).cumcount() == 0
    ).astype("int8")
    out["is_new_category"] = (
        out.groupby([config.CARD_COL, "category"], sort=False, observed=True).cumcount() == 0
    ).astype("int8")

    # ------------------------------------------------------------------
    # Final tidy-up
    # ------------------------------------------------------------------
    # Ratios can hit +/-inf only through division by zero, which we prevent above,
    # but replacing defensively guarantees the model never sees inf.
    out[config.NUMERIC_FEATURES] = out[config.NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    # Restore the caller's original row order.
    return out.loc[original_index]


def feature_table() -> pd.DataFrame:
    """A tidy description of every feature — used by the README and the dashboard."""
    groups = (
        ["Transaction"] * 2 + ["Customer"] * 2 + ["Time"] * 6 + ["Geography"] * 3
        + ["Velocity"] * 7 + ["Behavioural baseline"] * 8 + ["Categorical"] * 2
    )
    return pd.DataFrame({"feature": config.FEATURES, "group": groups})
