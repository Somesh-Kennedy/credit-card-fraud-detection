"""
predict.py — score new transactions with the trained model.

Pipeline stage 9 (inference) of the workflow.

The catch with behavioural features: you cannot score a transaction in isolation.
"6 purchases in the last hour" or "14x this card's average" require the card's
HISTORY. So scoring works like a real-time fraud engine:

    new transactions  +  each card's past transactions (from the cleaned dataset)
        -> build the 30 features over the combined timeline
        -> keep only the new rows
        -> model probability  ->  FRAUD / OK decision  ->  top reasons

Usage:
    python -m fraud.predict --input examples/sample_transactions.csv
    python -m fraud.predict --input my_batch.csv --output scored.csv --top-k 5

The input CSV uses the same columns as the raw Kaggle file (is_fraud optional).
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd

from fraud import config, data, explain, features


def load_bundle(path=None) -> dict:
    """Load the model bundle written by train.py (pipeline + threshold + metadata)."""
    path = path or config.MODELS_DIR / "fraud_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run `python -m fraud.train` first.")
    return joblib.load(path)


def score(new_txns: pd.DataFrame, history: pd.DataFrame | None = None,
          bundle: dict | None = None, top_k: int = 3) -> pd.DataFrame:
    """Score a batch of new transactions.

    new_txns : raw-format rows (as in the Kaggle CSV) or already-cleaned rows.
    history  : cleaned past transactions used to compute behavioural features.
               Defaults to the full cleaned dataset (data/transactions_clean.parquet).
    Returns new_txns' key columns plus fraud_probability, decision and reasons.
    """
    bundle = bundle or load_bundle()

    # Accept raw CSV rows: if the raw timestamp column is present, run data.clean().
    new = data.clean(new_txns) if "trans_date_trans_time" in new_txns.columns else new_txns.copy()
    new["_is_new"] = True  # marker so we can pick these rows out again later

    if history is None:
        history = data.load_clean()
    # Only the history of the cards being scored matters, and only transactions
    # strictly BEFORE the batch starts — so if a batch row already exists in the
    # history file it is not counted twice.
    hist = history[history[config.CARD_COL].isin(new[config.CARD_COL])
                   & (history[config.TIME_COL] < new[config.TIME_COL].min())].copy()
    hist["_is_new"] = False

    # Build features on the combined, time-ordered timeline, then keep the new rows.
    combined = pd.concat([hist, new], ignore_index=True)
    feats = features.build_features(combined)
    scored = feats[feats["_is_new"]].copy()

    X = scored[config.FEATURES]
    scored["fraud_probability"] = bundle["pipeline"].predict_proba(X)[:, 1]
    scored["decision"] = (scored["fraud_probability"] >= bundle["threshold"]).map(
        {True: "FRAUD - review", False: "OK"})
    # Reasons are most useful for flagged rows, but computing for all keeps it simple.
    scored["top_reasons"] = [" | ".join(r) for r in explain.reason_codes(bundle["pipeline"], X, top_k)]

    keep = [config.CARD_COL, config.TIME_COL, "merchant", "category", "amt",
            "fraud_probability", "decision", "top_reasons"]
    if config.TARGET in scored.columns and scored[config.TARGET].notna().any():
        keep.insert(5, config.TARGET)  # show the true label when we have it (demo / backtest)
    return scored[keep].sort_values(config.TIME_COL).reset_index(drop=True)


def main(argv=None) -> None:
    """CLI: read a CSV, score it, print a summary and optionally save the result."""
    p = argparse.ArgumentParser(description="Score new transactions with the trained fraud model.")
    p.add_argument("--input", required=True, help="CSV of transactions in the raw Kaggle format")
    p.add_argument("--output", help="optional path to write the scored CSV")
    p.add_argument("--top-k", type=int, default=3, help="number of reasons per transaction")
    args = p.parse_args(argv)

    new = pd.read_csv(args.input, dtype=data.RAW_DTYPES)
    result = score(new, top_k=args.top_k)

    # Show probabilities as 0.9997 instead of 9.997e-01 for readability.
    with pd.option_context("display.max_colwidth", 120, "display.width", 250,
                           "display.float_format", "{:.4f}".format):
        print(result.to_string(index=False))
    print(f"\n{(result['decision'] != 'OK').sum()} of {len(result)} transactions flagged for review.")

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
