"""
train.py — run the whole pipeline end to end.

Usage (from the repo root, with the virtual environment active):

    python -m fraud.train                 # full run (~10 min on a laptop)
    python -m fraud.train --quick         # 25% of cards, for a fast smoke test (~2 min)
    python -m fraud.train --models lgbm_smote logreg_smote   # only some experiments
    python -m fraud.train --skip-eda --skip-segments          # skip the optional stages

What it does, in order:

    1. load + clean the raw CSV            (data.py)        -> data/transactions_clean.parquet
    2. engineer the 30 features            (features.py)    -> data/transactions_features.parquet
    3. EDA figures                         (eda.py)         -> reports/figures/eda_*.png
    4. time-based train / valid / test split
    5. train every candidate model on TRAIN, score VALID   (models.py)
    6. pick the best model by validation PR-AUC
    7. choose its decision threshold on VALID by minimising business cost (evaluate.py)
    8. evaluate ONCE on TEST with that threshold           -> reports/metrics.json + figures
    9. unsupervised Isolation Forest baseline              (anomaly.py)
   10. explainability: global SHAP importance              (explain.py)
   11. customer segmentation                               (segmentation.py)
   12. save the model bundle + scored test set for the dashboard / predict.py
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from fraud import anomaly, config, data, eda, evaluate, explain, features, segmentation
from fraud.models import MODEL_SPECS, build_pipeline


def log(msg: str) -> None:
    """Timestamped console logging so long runs show visible progress."""
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    """Command-line options (see module docstring for examples)."""
    p = argparse.ArgumentParser(description="Train and evaluate the fraud detection pipeline.")
    p.add_argument("--quick", action="store_true",
                   help="use a random 25%% of cards for a fast end-to-end smoke test")
    p.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=list(MODEL_SPECS),
                   help="which experiments to run (default: all)")
    p.add_argument("--rebuild", action="store_true",
                   help="ignore cached Parquet files and rebuild from the raw CSV")
    p.add_argument("--skip-eda", action="store_true", help="do not regenerate EDA figures")
    p.add_argument("--skip-segments", action="store_true", help="skip customer segmentation")
    p.add_argument("--skip-anomaly", action="store_true", help="skip the Isolation Forest baseline")
    return p.parse_args(argv)


def load_features(rebuild: bool, quick: bool) -> pd.DataFrame:
    """Stages 1-2: cleaned data + engineered features, cached to Parquet."""
    if not rebuild and not quick and config.FEATURES_PARQUET.exists():
        log(f"Loading cached features from {config.FEATURES_PARQUET.name}")
        return pd.read_parquet(config.FEATURES_PARQUET)

    log("Loading + cleaning raw CSV ...")
    df = data.load_clean(use_cache=not rebuild)

    if quick:
        # Sample whole CARDS (not random rows): per-card history features need a
        # card's complete transaction sequence to be computed correctly.
        cards = df[config.CARD_COL].drop_duplicates().sample(frac=0.25, random_state=config.RANDOM_STATE)
        df = df[df[config.CARD_COL].isin(cards)]
        log(f"--quick: kept {len(cards):,} cards / {len(df):,} transactions")

    log("Engineering 30 features ...")
    df = features.build_features(df)
    if not quick:  # never overwrite the full cache with a partial dataset
        df.to_parquet(config.FEATURES_PARQUET, index=False)
    return df


def run(args: argparse.Namespace) -> dict:
    config.ensure_dirs()
    t_start = time.time()

    # ------------------------------------------------------------------ 1-2
    df = load_features(args.rebuild, args.quick)

    # ------------------------------------------------------------------ 3
    if not args.skip_eda:
        log("Saving EDA figures ...")
        eda.save_all(df)

    # ------------------------------------------------------------------ 4
    train, valid, test = data.time_split(df)
    for name, part in [("train", train), ("validation", valid), ("test", test)]:
        log(data.describe_split(name, part))

    X_train, y_train = train[config.FEATURES], train[config.TARGET]
    X_valid, y_valid = valid[config.FEATURES], valid[config.TARGET]
    X_test, y_test = test[config.FEATURES], test[config.TARGET]

    # Class weight for the "weighted" experiment: sqrt of the imbalance ratio.
    pos_weight = float(np.sqrt((y_train == 0).sum() / (y_train == 1).sum()))

    # ------------------------------------------------------------------ 5
    fitted, valid_scores, test_scores = {}, {}, {}
    valid_results, test_results = {}, {}
    for name in args.models:
        spec = MODEL_SPECS[name]
        log(f"Training {name:<14} - {spec.description}")
        pipe = build_pipeline(spec, pos_weight=pos_weight)

        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_seconds = time.time() - t0

        # predict_proba returns [P(legit), P(fraud)] per row; keep the fraud column.
        valid_scores[name] = pipe.predict_proba(X_valid)[:, 1]
        test_scores[name] = pipe.predict_proba(X_test)[:, 1]

        # Each model gets ITS OWN cost-optimal threshold, chosen on validation, so
        # the comparison is fair (a 0.5 cut-off suits some models far better than others).
        thr = evaluate.best_cost_threshold(y_valid, valid_scores[name], valid["amt"])
        valid_results[name] = {**evaluate.compute_metrics(y_valid, valid_scores[name], valid["amt"], thr),
                               "train_seconds": train_seconds}
        test_results[name] = {**evaluate.compute_metrics(y_test, test_scores[name], test["amt"], thr),
                              "train_seconds": train_seconds}
        fitted[name] = pipe
        m = valid_results[name]
        log(f"   valid PR-AUC {m['pr_auc']:.4f} | ROC-AUC {m['roc_auc']:.4f} | "
            f"precision {m['precision']:.3f} recall {m['recall']:.3f} | {train_seconds:.0f}s")

    # ------------------------------------------------------------------ 6
    # Model selection uses VALIDATION only — the test set must not influence any choice.
    best_name = max(valid_results, key=lambda n: valid_results[n]["pr_auc"])
    best = fitted[best_name]
    log(f"Best model on validation: {best_name}")

    # ------------------------------------------------------------------ 7
    threshold = valid_results[best_name]["threshold"]
    f1_threshold = evaluate.best_f1_threshold(y_valid, valid_scores[best_name])
    evaluate.plot_cost_curve(y_valid, valid_scores[best_name], valid["amt"], threshold)
    log(f"Cost-optimal threshold = {threshold:.4f} (F1-optimal would be {f1_threshold:.4f})")

    # ------------------------------------------------------------------ 8
    test_metrics = test_results[best_name]
    log(f"TEST  PR-AUC {test_metrics['pr_auc']:.4f} | ROC-AUC {test_metrics['roc_auc']:.4f} | "
        f"precision {test_metrics['precision']:.3f} | recall {test_metrics['recall']:.3f} | "
        f"fraud $ caught {test_metrics['fraud_amt_caught_pct']:.1%}")
    evaluate.plot_pr_curves(y_test, test_scores, title="Precision-recall curves (test set)")
    evaluate.plot_roc_curves(y_test, test_scores, title="ROC curves (test set)")
    evaluate.plot_confusion(test_metrics, title=f"Confusion matrix — {best_name} (test set)")
    evaluate.plot_score_distribution(y_test, test_scores[best_name], threshold)

    # ------------------------------------------------------------------ 9
    anomaly_metrics = None
    if not args.skip_anomaly:
        log("Fitting Isolation Forest (unsupervised baseline) ...")
        iforest = anomaly.fit_isolation_forest(train)
        a_scores = anomaly.anomaly_scores(iforest, test)
        # A ranking score has no natural threshold; report it at the same alert
        # volume as the best supervised model so the two are directly comparable.
        same_volume_thr = float(np.sort(a_scores)[::-1][max(test_metrics["alerts"] - 1, 0)])
        anomaly_metrics = evaluate.compute_metrics(y_test, a_scores, test["amt"], same_volume_thr)
        anomaly_metrics.pop("threshold")
        test_results["isolation_forest"] = {**anomaly_metrics, "train_seconds": np.nan}
        log(f"   Isolation Forest test PR-AUC {anomaly_metrics['pr_auc']:.4f} | "
            f"ROC-AUC {anomaly_metrics['roc_auc']:.4f}")

    # Save comparison tables (validation = used for selection, test = final report).
    evaluate.comparison_table(valid_results).to_csv(config.REPORTS_DIR / "model_comparison_valid.csv")
    evaluate.comparison_table(test_results).to_csv(config.REPORTS_DIR / "model_comparison_test.csv")

    # ------------------------------------------------------------------ 10
    log("Computing SHAP feature importance ...")
    # TreeSHAP on 10K random test rows gives a stable global ranking in ~1 minute
    # (it is exact but costly: every row walks all 600 trees).
    shap_sample = X_test.sample(min(10_000, len(X_test)), random_state=config.RANDOM_STATE)
    importance = explain.global_importance(best, shap_sample)
    importance.rename("mean_abs_shap").to_csv(config.REPORTS_DIR / "feature_importance.csv")
    explain.plot_global_importance(importance)

    # ------------------------------------------------------------------ 11
    segment_summary = None
    if not args.skip_segments:
        log("Segmenting customers ...")
        profiles, segment_summary, silhouette = segmentation.segment_customers(df)
        profiles.to_parquet(config.DATA_DIR / "card_segments.parquet")
        segment_summary.to_csv(config.REPORTS_DIR / "segments_summary.csv", index=False)
        segmentation.plot_segments(profiles, silhouette)

    # ------------------------------------------------------------------ 12
    # The bundle holds everything needed to score new data consistently.
    bundle = {
        "pipeline": best,
        "model_name": best_name,
        "threshold": threshold,
        "features": config.FEATURES,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_period": [str(train[config.TIME_COL].min()), str(train[config.TIME_COL].max())],
        "quick": args.quick,
    }
    joblib.dump(bundle, config.MODELS_DIR / "fraud_model.joblib")

    # Scored test set (+ every model's score) for the dashboard's interactive tabs.
    scored = test[[config.CARD_COL, config.TIME_COL, "merchant", "state", "lat", "long",
                   "merch_lat", "merch_long", config.TARGET, *config.FEATURES]].copy()
    scored["score"] = test_scores[best_name]
    for name, s in test_scores.items():
        scored[f"score_{name}"] = s
    scored.to_parquet(config.DATA_DIR / "test_scored.parquet", index=False)

    report = {
        "best_model": best_name,
        "threshold": threshold,
        "f1_threshold": f1_threshold,
        "false_positive_cost": config.FALSE_POSITIVE_COST,
        "smote_ratio": config.SMOTE_RATIO,
        "quick_run": args.quick,
        "splits": {name: {"rows": len(p), "frauds": int(p[config.TARGET].sum()),
                          "start": str(p[config.TIME_COL].min().date()),
                          "end": str(p[config.TIME_COL].max().date())}
                   for name, p in [("train", train), ("valid", valid), ("test", test)]},
        "validation": valid_results[best_name],
        "test": test_metrics,
        "isolation_forest_test": anomaly_metrics,
        "top_features": importance.head(10).round(4).to_dict(),
        "runtime_minutes": round((time.time() - t_start) / 60, 1),
    }
    with open(config.REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    log(f"Done in {report['runtime_minutes']} min. Results in {config.REPORTS_DIR}")
    return report


def main(argv=None) -> None:
    """Console-script entry point (`fraud-train` / `python -m fraud.train`)."""
    run(parse_args(argv))


if __name__ == "__main__":
    main()
