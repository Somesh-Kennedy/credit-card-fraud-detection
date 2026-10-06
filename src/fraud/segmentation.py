"""
segmentation.py — customer segmentation with K-Means.

Pipeline stage 8 of the workflow.

Beyond fraud, the dataset describes ~1,000 customers' spending behaviour. Grouping
them into segments is useful for marketing (tailored offers) AND for fraud teams:
fraud that is "normal" for a heavy online shopper is abnormal for a low-spend,
in-store customer, and some segments are targeted far more than others.

Steps:
  1. aggregate transactions to one row per card (RFM-style behaviour profile)
  2. standardise the profile features
  3. report the silhouette score for K = 2..8 (a sanity check), run K-Means with
     config.N_SEGMENTS
  4. describe each segment and give it a readable name from its strongest traits
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from fraud import config

# Human-friendly labels for each profile feature, used to auto-name segments.
PROFILE_LABELS = {
    "txn_per_month": ("Frequent", "Infrequent"),
    "avg_amt": ("High-ticket", "Low-ticket"),
    "monthly_spend": ("Big spender", "Light spender"),
    "online_share": ("Online-heavy", "In-store"),
    "night_share": ("Night owl", "Daytime"),
    "weekend_share": ("Weekend shopper", "Weekday shopper"),
    "age": ("Older", "Younger"),
    "log_city_pop": ("Urban", "Rural"),
}
PROFILE_FEATURES = list(PROFILE_LABELS)


def card_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """One row per card describing its *genuine* spending behaviour.

    Fraudulent transactions are excluded: they describe the fraudster, not the
    customer. The card's fraud rate is kept separately to analyse later.
    """
    legit = df[df[config.TARGET] == 0]
    g = legit.groupby(config.CARD_COL)

    # Active months = span between first and last purchase (at least 1 month).
    span_months = ((g[config.TIME_COL].max() - g[config.TIME_COL].min()).dt.days / 30.44).clip(lower=1)

    profile = pd.DataFrame({
        "txn_per_month": g.size() / span_months,                      # frequency
        "avg_amt": g["amt"].mean(),                                    # ticket size
        "monthly_spend": g["amt"].sum() / span_months,                 # monetary value
        # Categories ending in "_net" are online (card-not-present) purchases.
        "online_share": g["category"].apply(lambda c: c.astype(str).str.endswith("_net").mean()),
        "night_share": g["is_night"].mean(),
        "weekend_share": g["is_weekend"].mean(),
        "age": g["age"].median(),
        "log_city_pop": g["log_city_pop"].first(),
        "state": g["state"].first(),
        "lat": g["lat"].first(),
        "long": g["long"].first(),
    })
    # Fraud exposure per card, computed on ALL its transactions (incl. fraud).
    profile["fraud_rate"] = df.groupby(config.CARD_COL)[config.TARGET].mean()
    profile["was_compromised"] = (df.groupby(config.CARD_COL)[config.TARGET].max() == 1).astype(int)
    return profile


def choose_k(X: np.ndarray, k_range=range(2, 9)) -> pd.Series:
    """Silhouette score for each K (higher = tighter, better-separated clusters)."""
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    return pd.Series(scores, name="silhouette")


def _name_segment(z_row: pd.Series) -> str:
    """Name a segment by its two most distinctive traits (largest |z-score| of its centre)."""
    top = z_row.abs().sort_values(ascending=False).head(2).index
    parts = [PROFILE_LABELS[f][0] if z_row[f] > 0 else PROFILE_LABELS[f][1] for f in top]
    return " / ".join(parts)


def segment_customers(df: pd.DataFrame, k: int = config.N_SEGMENTS):
    """Run the full segmentation.

    Returns:
        profiles  : one row per card with its 'segment' id and 'segment_name'
        summary   : one row per segment (size, average behaviour, fraud exposure)
        silhouette: silhouette score per candidate K (for the elbow chart)
    """
    profiles = card_profiles(df)

    # K-Means uses Euclidean distance, so features must share a scale — otherwise
    # monthly_spend (thousands) would drown out online_share (0..1).
    scaler = StandardScaler()
    X = scaler.fit_transform(profiles[PROFILE_FEATURES])

    silhouette = choose_k(X)
    km = KMeans(n_clusters=k, n_init=20, random_state=config.RANDOM_STATE)
    profiles["segment"] = km.fit_predict(X)

    # Cluster centres are in scaled (z-score) units: perfect for naming, because
    # z = +1.5 on online_share literally means "much more online than average".
    centres_z = pd.DataFrame(km.cluster_centers_, columns=PROFILE_FEATURES)
    names = {i: _name_segment(centres_z.loc[i]) for i in centres_z.index}
    profiles["segment_name"] = profiles["segment"].map(names)

    summary = (
        profiles.groupby(["segment", "segment_name"])
        .agg(customers=("avg_amt", "size"),
             **{f: (f, "mean") for f in PROFILE_FEATURES},
             pct_cards_compromised=("was_compromised", "mean"),
             fraud_rate=("fraud_rate", "mean"))
        .reset_index()
    )
    return profiles, summary, silhouette


def plot_segments(profiles: pd.DataFrame, silhouette: pd.Series, name="segments.png"):
    """Left: silhouette by K. Right: customers in (avg ticket, frequency) space by segment."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(silhouette.index, silhouette.values, marker="o", color="#2563eb")
    ax1.axvline(config.N_SEGMENTS, color="#dc2626", ls="--", label=f"chosen K = {config.N_SEGMENTS}")
    ax1.set(xlabel="Number of clusters K", ylabel="Silhouette score", title="Choosing K")
    ax1.legend()
    ax1.grid(alpha=0.3)

    for seg, part in profiles.groupby("segment"):
        ax2.scatter(part["avg_amt"], part["txn_per_month"], s=14, alpha=0.7,
                    label=f"{seg}: {part['segment_name'].iat[0]}")
    ax2.set(xlabel="Average transaction ($)", ylabel="Transactions per month",
            title="Customer segments", xscale="log")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(alpha=0.3)

    config.ensure_dirs()
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
