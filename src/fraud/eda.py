"""
eda.py — exploratory data analysis figures.

Used in two places:
  * notebooks/01_eda.ipynb  (interactive walkthrough with commentary)
  * train.py                (saves the same charts to reports/figures for the README)

Keeping the plotting code here (instead of only in the notebook) means the
figures in the README are always regenerated from the same logic, and the
notebook stays short and readable.

Every function takes the feature-engineered DataFrame and returns a matplotlib
Figure, so callers can either display it (notebook) or save it (pipeline).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from fraud import config

LEGIT_COLOR, FRAUD_COLOR = "#2563eb", "#dc2626"


def class_balance(df: pd.DataFrame):
    """How rare is fraud? (the reason SMOTE / weighting are needed)."""
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(["Legit", "Fraud"], counts.values, color=[LEGIT_COLOR, FRAUD_COLOR])
    ax.set_yscale("log")  # on a linear axis the fraud bar would be invisible
    for bar, n in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, n, f"{n:,}\n({n / counts.sum():.2%})",
                ha="center", va="bottom", fontsize=9)
    ax.set(title="Class balance (log scale)", ylabel="Transactions")
    ax.set_ylim(top=counts.max() * 8)  # head-room for the labels
    return fig


def amount_distribution(df: pd.DataFrame):
    """Fraud amounts look very different from normal spending."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(0, np.log10(df["amt"].max()), 60)  # log-spaced bins for a skewed variable
    for label, color, name in [(0, LEGIT_COLOR, "Legit"), (1, FRAUD_COLOR, "Fraud")]:
        ax.hist(df.loc[df[config.TARGET] == label, "amt"], bins=bins, density=True,
                alpha=0.55, color=color, label=name)
    ax.set_xscale("log")
    ax.set(title="Transaction amount distribution", xlabel="Amount ($, log scale)", ylabel="Density")
    ax.legend()
    return fig


def fraud_rate_by(df: pd.DataFrame, column: str, title: str, sort_by_rate: bool = False):
    """Fraud rate (bars) and transaction volume (line) for each value of `column`."""
    stats = df.groupby(column, observed=True)[config.TARGET].agg(["mean", "size"])
    if sort_by_rate:
        stats = stats.sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(stats))
    ax.bar(x, stats["mean"], color=FRAUD_COLOR, alpha=0.8, label="Fraud rate")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xticks(x, stats.index.astype(str), rotation=45 if len(stats) > 8 else 0, ha="right")
    ax.set(title=title, ylabel="Fraud rate")

    # Volume on a secondary axis shows whether a high rate is on many or few transactions.
    ax2 = ax.twinx()
    ax2.plot(x, stats["size"], color="grey", marker="o", ms=3, lw=1, label="Volume")
    ax2.set_ylabel("Transactions")
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v / 1e3:,.0f}K"))
    return fig


def monthly_trend(df: pd.DataFrame):
    """Volume and fraud rate over time (seasonality, December spike, drift)."""
    monthly = df.groupby(df[config.TIME_COL].dt.to_period("M"))[config.TARGET].agg(["size", "mean"])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = monthly.index.to_timestamp()
    ax.bar(x, monthly["size"], width=20, color=LEGIT_COLOR, alpha=0.35, label="Transactions")
    ax.set_ylabel("Transactions")
    ax2 = ax.twinx()
    ax2.plot(x, monthly["mean"], color=FRAUD_COLOR, marker="o", lw=2, label="Fraud rate")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2))
    ax2.set_ylabel("Fraud rate")
    # Shade the validation and test periods so readers see the time-based split.
    for start, end, lbl in [(config.VALID_START, config.TEST_START, "validation"),
                            (config.TEST_START, str(df[config.TIME_COL].max()), "test")]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.12)
        ax.text(pd.Timestamp(start), ax.get_ylim()[1] * 0.95, f" {lbl}", fontsize=8, va="top")
    ax.set_title("Monthly volume and fraud rate (shaded = held-out periods)")
    return fig


def feature_signal(df: pd.DataFrame):
    """How differently does each numeric feature behave for fraud vs. legit?

    Uses the standardised mean difference (Cohen's d): (mean_fraud - mean_legit) / pooled std.
    It is unit-free, so a $ feature and a 0/1 flag can be compared on one chart.
    """
    num = df[config.NUMERIC_FEATURES]
    fraud, legit = num[df[config.TARGET] == 1], num[df[config.TARGET] == 0]
    pooled = np.sqrt((fraud.var() + legit.var()) / 2)
    d = ((fraud.mean() - legit.mean()) / pooled).sort_values()

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(d.index, d.values, color=np.where(d.values > 0, FRAUD_COLOR, LEGIT_COLOR))
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title="Feature signal: fraud vs. legit (Cohen's d)",
           xlabel="Standardised mean difference (positive = higher for fraud)")
    ax.grid(axis="x", alpha=0.3)
    return fig


def fraud_map(df: pd.DataFrame, sample: int = 60_000):
    """Geospatial view: where are customers, and where does fraud happen?"""
    legit = df[df[config.TARGET] == 0].sample(min(sample, len(df)), random_state=config.RANDOM_STATE)
    fraud = df[df[config.TARGET] == 1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(legit["long"], legit["lat"], s=2, alpha=0.15, color=LEGIT_COLOR, label="Legit (sample)")
    ax.scatter(fraud["long"], fraud["lat"], s=4, alpha=0.5, color=FRAUD_COLOR, label="Fraud")
    ax.set(title="Card-holder locations: legit vs. fraud", xlabel="Longitude", ylabel="Latitude",
           xlim=(-126, -66), ylim=(24, 50))  # zoom on the contiguous US
    ax.legend(markerscale=4)
    return fig


def save_all(df: pd.DataFrame) -> list:
    """Generate every EDA figure into reports/figures (called by train.py)."""
    config.ensure_dirs()
    figures = {
        "eda_class_balance.png": class_balance(df),
        "eda_amount_distribution.png": amount_distribution(df),
        "eda_fraud_by_hour.png": fraud_rate_by(df, "hour", "Fraud rate by hour of day"),
        "eda_fraud_by_category.png": fraud_rate_by(df, "category", "Fraud rate by merchant category",
                                                   sort_by_rate=True),
        "eda_monthly_trend.png": monthly_trend(df),
        "eda_feature_signal.png": feature_signal(df),
        "eda_fraud_map.png": fraud_map(df),
    }
    paths = []
    for name, fig in figures.items():
        path = config.FIGURES_DIR / name
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
