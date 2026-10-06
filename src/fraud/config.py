"""
config.py — single source of truth for paths, constants and hyper-parameters.

Why a config module?
    Hard-coding "dataset/credit_card_transactions.csv" or "2020-01-01" in five
    different files makes the project fragile: change it in one place and forget
    another, and the pipeline silently does something different. Every other module
    imports its settings from here instead, so a change is made exactly once.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# __file__ is .../src/fraud/config.py, so three .parent hops reach the repo root.
# Building every path from the root means the code works no matter which folder
# you launch it from (terminal, VS Code, Jupyter, Streamlit, pytest ...).
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Raw input: the Kaggle CSV you download and drop into dataset/ (it is git-ignored
# because it is ~340 MB — far above GitHub's 100 MB file limit).
RAW_CSV = ROOT_DIR / "dataset" / "credit_card_transactions.csv"

# Intermediate artefacts produced by the pipeline (all git-ignored, all re-creatable).
DATA_DIR = ROOT_DIR / "data"
CLEAN_PARQUET = DATA_DIR / "transactions_clean.parquet"     # typed + sorted raw data
FEATURES_PARQUET = DATA_DIR / "transactions_features.parquet"  # raw data + 30 features

# Trained models (git-ignored) and human-readable results (committed so the README
# can show them on GitHub without anyone having to re-run the pipeline).
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# One seed shared by SMOTE, the models, K-Means and any sampling, so two runs on
# the same data produce identical numbers.
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Time-based split
# ---------------------------------------------------------------------------
# Fraud models are always deployed on *future* transactions, so we evaluate them
# the same way: train on the past, tune on the next period, test on the one after.
# A random split would let the model "see the future" (e.g. a fraud ring that is
# active in May would appear in both train and test) and inflate every metric.
#
#   train      : 2019-01-01 -> 2019-12-31   (~915K rows, a full year incl. holidays)
#   validation : 2020-01-01 -> 2020-03-31   (~173K rows, used for model selection +
#                                            choosing the decision threshold)
#   test       : 2020-04-01 -> end of data  (~199K rows, touched exactly once)
VALID_START = "2020-01-01"
TEST_START = "2020-04-01"

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
TARGET = "is_fraud"          # 1 = fraudulent, 0 = legitimate
TIME_COL = "trans_time"      # parsed datetime (created in data.py)
CARD_COL = "cc_num"          # the card / customer identifier

# Personally identifying columns. They carry no generalisable fraud signal (a
# name is not "risky") and keeping them in a model would be a privacy and fairness
# problem, so data.py drops them right after loading.
PII_COLS = ["first", "last", "street", "trans_num"]

# ---------------------------------------------------------------------------
# The 30 engineered model features (built in features.py)
# ---------------------------------------------------------------------------
# Numeric features go through impute -> scale; categorical ones through one-hot.
NUMERIC_FEATURES = [
    # -- transaction ---------------------------------------------------------
    "amt",                       # 1  raw amount in USD
    "log_amt",                   # 2  log1p(amount): tames the long right tail
    # -- customer ------------------------------------------------------------
    "age",                       # 3  card-holder age at the time of purchase
    "log_city_pop",              # 4  log of the card-holder's city population
    # -- time ----------------------------------------------------------------
    "hour",                      # 5  hour of day (0-23)
    "hour_sin",                  # 6  cyclical encoding so 23:00 is "close" to 00:00
    "hour_cos",                  # 7
    "day_of_week",               # 8  0 = Monday ... 6 = Sunday
    "is_weekend",                # 9
    "is_night",                  # 10 22:00-03:59, when most fraud happens
    # -- geography -----------------------------------------------------------
    "cust_merch_dist_km",        # 11 home -> merchant distance (haversine)
    "km_from_prev_txn",          # 12 distance between this and the previous merchant
    "speed_kmh_from_prev",       # 13 implied travel speed ("impossible travel")
    # -- velocity (how busy has this card been recently?) --------------------
    "secs_since_prev_txn",       # 14
    "txn_count_1h",              # 15
    "txn_count_24h",             # 16
    "txn_count_7d",              # 17
    "amt_sum_1h",                # 18
    "amt_sum_24h",               # 19
    "amt_sum_7d",                # 20
    # -- behavioural baseline (is this normal *for this card*?) --------------
    "card_txn_count_hist",       # 21 how many purchases the card made before this one
    "card_amt_mean_hist",        # 22 the card's historical average amount
    "card_amt_std_hist",         # 23 ... and its spread
    "amt_zscore_card",           # 24 how many std-devs above the card's normal
    "amt_to_card_mean_ratio",    # 25 amount / card's historical mean
    "amt_to_card_cat_mean_ratio",  # 26 amount / card's historical mean *in this category*
    "is_new_merchant",           # 27 first time this card buys from this merchant
    "is_new_category",           # 28 first time this card buys in this category
]
CATEGORICAL_FEATURES = [
    "category",                  # 29 merchant category (14 levels, e.g. shopping_net)
    "gender",                    # 30 card-holder gender (F / M)
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
assert len(FEATURES) == 30, "the project advertises exactly 30 engineered features"

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
# SMOTE target ratio: after resampling, frauds = 10% of the legitimate count.
# Going all the way to 50/50 (1.0) would create ~900K synthetic frauds — slow to
# train on and it pushes the model to over-predict fraud. 0.1 is a common sweet spot.
SMOTE_RATIO = 0.10
SMOTE_K_NEIGHBORS = 5

# Business cost model used to choose the decision threshold.
#   * Missing a fraud (false negative) costs the bank the full transaction amount.
#   * Flagging a good transaction (false positive) costs an analyst review + some
#     customer friction. $10 is a typical industry ballpark for that.
FALSE_POSITIVE_COST = 10.0

# Recall levels we report precision at — e.g. "at 80% recall, how many alerts are
# actually fraud?" is the question a fraud-ops team asks.
RECALL_TARGETS = [0.80, 0.90, 0.95]

# ---------------------------------------------------------------------------
# Customer segmentation
# ---------------------------------------------------------------------------
# K in K-Means. Silhouette scores are low for every K (0.22-0.30, see
# reports/figures/segments.png): customers in this synthetic data do not form sharp
# clusters. K = 5 is a deliberate choice for INTERPRETABILITY (few enough segments to
# describe and act on), not the silhouette maximum.
N_SEGMENTS = 5


def ensure_dirs() -> None:
    """Create every output folder (no-op if they already exist)."""
    # parents=True creates intermediate folders; exist_ok=True avoids errors on reruns.
    for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
