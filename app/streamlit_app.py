"""
streamlit_app.py — interactive fraud-detection dashboard.

Run it (after `python -m fraud.train` has produced the artefacts):

    streamlit run app/streamlit_app.py

Tabs:
    Overview          KPIs + how fraud varies over time, category and hour
    Model performance compare models, move the decision threshold, see the $ impact live
    Transaction explorer  pick any test transaction and see WHY it was scored that way
    Live scoring      invent a new transaction for a real card and score it in real time
    Geography         fraud rate by state + map of caught vs. missed fraud
    Customer segments K-Means segments and how often each gets targeted

Streamlit re-runs this whole script top-to-bottom on every widget interaction.
@st.cache_data / @st.cache_resource make sure the heavy loading happens once.
"""

from __future__ import annotations

import json
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve

from fraud import config, evaluate, explain, predict

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="💳", layout="wide")

FRAUD_RED, LEGIT_BLUE = "#dc2626", "#2563eb"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading results ...")
def load_reports():
    """Small JSON / CSV outputs of train.py."""
    with open(config.REPORTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    comparison = pd.read_csv(config.REPORTS_DIR / "model_comparison_test.csv", index_col=0)
    importance = pd.read_csv(config.REPORTS_DIR / "feature_importance.csv", index_col=0).iloc[:, 0]
    return metrics, comparison, importance


@st.cache_data(show_spinner="Loading scored test set ...")
def load_test_scored() -> pd.DataFrame:
    return pd.read_parquet(config.DATA_DIR / "test_scored.parquet")


@st.cache_data(show_spinner="Loading all transactions ...")
def load_overview_data() -> pd.DataFrame:
    """Only the columns the Overview / Geography tabs need (keeps memory low)."""
    cols = [config.TIME_COL, config.TARGET, "amt", "category", "hour", "state"]
    return pd.read_parquet(config.FEATURES_PARQUET, columns=cols)


@st.cache_data(show_spinner="Loading card history ...")
def load_history() -> pd.DataFrame:
    return pd.read_parquet(config.CLEAN_PARQUET)


@st.cache_resource(show_spinner="Loading model ...")
def load_model():
    # cache_resource (not cache_data): the model object should be shared, not copied.
    return predict.load_bundle()


# Stop gracefully with instructions if the pipeline has not been run yet.
required = [config.REPORTS_DIR / "metrics.json", config.DATA_DIR / "test_scored.parquet",
            config.MODELS_DIR / "fraud_model.joblib"]
if not all(p.exists() for p in required):
    st.error("Artefacts not found. Run `python -m fraud.train` first, then reload this page.")
    st.stop()

metrics, comparison, importance = load_reports()
test_df = load_test_scored()
bundle = load_model()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("💳 Credit Card Fraud Detection")
st.caption(
    f"Model **{metrics['best_model']}** trained on {metrics['splits']['train']['rows']:,} transactions "
    f"({metrics['splits']['train']['start']} → {metrics['splits']['train']['end']}), "
    f"evaluated on {metrics['splits']['test']['rows']:,} unseen future transactions "
    f"({metrics['splits']['test']['start']} → {metrics['splits']['test']['end']})."
)

tabs = st.tabs(["Overview", "Model performance", "Transaction explorer",
                "Live scoring", "Geography", "Customer segments"])

# ===========================================================================
# 1. Overview
# ===========================================================================
with tabs[0]:
    df = load_overview_data()
    fraud = df[df[config.TARGET] == 1]

    # Headline KPIs across the whole dataset.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Fraudulent", f"{len(fraud):,}", f"{len(fraud) / len(df):.2%} of all", delta_color="off")
    c3.metric("Fraud $ total", f"${fraud['amt'].sum() / 1e6:,.2f}M")
    c4.metric("Avg fraud vs legit ticket",
              f"${fraud['amt'].mean():,.0f}", f"legit avg ${df.loc[df[config.TARGET] == 0, 'amt'].mean():,.0f}",
              delta_color="off")

    # Monthly volume (bars) + fraud rate (line) on twin axes.
    monthly = df.groupby(df[config.TIME_COL].dt.to_period("M").dt.to_timestamp())[config.TARGET].agg(["size", "mean"])
    fig = go.Figure()
    fig.add_bar(x=monthly.index, y=monthly["size"], name="Transactions", marker_color=LEGIT_BLUE, opacity=0.4)
    fig.add_scatter(x=monthly.index, y=monthly["mean"], name="Fraud rate", yaxis="y2",
                    line=dict(color=FRAUD_RED, width=3))
    fig.update_layout(title="Monthly volume and fraud rate", height=380,
                      yaxis=dict(title="Transactions"),
                      yaxis2=dict(title="Fraud rate", overlaying="y", side="right", tickformat=".2%"),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    by_cat = df.groupby("category", observed=True)[config.TARGET].mean().sort_values()
    left.plotly_chart(px.bar(by_cat, orientation="h", title="Fraud rate by merchant category",
                             labels={"value": "Fraud rate", "category": ""},
                             color_discrete_sequence=[FRAUD_RED])
                      .update_layout(showlegend=False, xaxis_tickformat=".1%", height=420),
                      width="stretch")
    by_hour = df.groupby("hour")[config.TARGET].mean()
    right.plotly_chart(px.bar(by_hour, title="Fraud rate by hour of day",
                              labels={"value": "Fraud rate", "hour": "Hour"},
                              color_discrete_sequence=[FRAUD_RED])
                       .update_layout(showlegend=False, yaxis_tickformat=".1%", height=420),
                       width="stretch")

# ===========================================================================
# 2. Model performance
# ===========================================================================
with tabs[1]:
    st.subheader("Model comparison (test set)")
    st.caption("Each model uses its own cost-optimal threshold chosen on the validation set. "
               "PR-AUC is the headline metric for imbalanced data.")
    st.dataframe(
        comparison.style.format({
            "pr_auc": "{:.4f}", "roc_auc": "{:.4f}", "precision": "{:.3f}", "recall": "{:.3f}",
            "f1": "{:.3f}", "precision_at_90_recall": "{:.3f}", "fraud_amt_caught_pct": "{:.1%}",
            "net_savings": "${:,.0f}", "train_seconds": "{:.0f}s"}, na_rep="—")
        .highlight_max(subset=["pr_auc"], color="#bbf7d0"),
        width="stretch")

    st.subheader("Choose your own decision threshold")
    st.caption("Slide to trade off catching more fraud (recall) against more false alarms. "
               f"Default = cost-optimal threshold ({metrics['threshold']:.4f}).")
    y, s, amt = test_df[config.TARGET], test_df["score"], test_df["amt"]

    col_a, col_b = st.columns([1, 1])
    thr = col_a.slider("Threshold", 0.0, 1.0, float(metrics["threshold"]), 0.001, format="%.3f")
    fp_cost = col_b.number_input("Cost of reviewing a false alarm ($)", 1.0, 200.0,
                                 float(config.FALSE_POSITIVE_COST), 1.0)

    # Recompute every metric live for the chosen threshold.
    m = evaluate.compute_metrics(y, s, amt, thr)
    m["review_cost"] = m["fp"] * fp_cost
    m["net_savings"] = m["fraud_amt_caught"] - m["review_cost"]
    k = st.columns(5)
    k[0].metric("Precision", f"{m['precision']:.1%}")
    k[1].metric("Recall", f"{m['recall']:.1%}")
    k[2].metric("Alerts / day", f"{m['alerts'] / test_df[config.TIME_COL].dt.date.nunique():,.0f}")
    k[3].metric("Fraud $ caught", f"{m['fraud_amt_caught_pct']:.1%}")
    k[4].metric("Net savings", f"${m['net_savings']:,.0f}")

    left, right = st.columns(2)
    # PR curve with the current operating point marked.
    p, r, _ = precision_recall_curve(y, s)
    pr_fig = go.Figure(go.Scatter(x=r, y=p, mode="lines", line=dict(color=LEGIT_BLUE, width=3), name="PR curve"))
    pr_fig.add_scatter(x=[m["recall"]], y=[m["precision"]], mode="markers",
                       marker=dict(size=14, color=FRAUD_RED), name="current threshold")
    pr_fig.update_layout(title=f"Precision-recall (PR-AUC {m['pr_auc']:.4f})", xaxis_title="Recall",
                         yaxis_title="Precision", height=400)
    left.plotly_chart(pr_fig, width="stretch")

    # Confusion matrix as an annotated heatmap.
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    cm_fig = px.imshow(np.log10(cm + 1), x=["Pred legit", "Pred fraud"], y=["Actual legit", "Actual fraud"],
                       color_continuous_scale="Blues", title="Confusion matrix")
    cm_fig.update_traces(text=[[f"{v:,}" for v in row] for row in cm], texttemplate="%{text}",
                         textfont_size=18, hovertemplate="%{y} / %{x}: %{text}<extra></extra>")
    cm_fig.update_layout(coloraxis_showscale=False, height=400)
    right.plotly_chart(cm_fig, width="stretch")

    st.subheader("What drives the model? (global SHAP importance)")
    st.plotly_chart(px.bar(importance.head(20)[::-1], orientation="h",
                           labels={"value": "Mean |SHAP| (log-odds)", "index": ""},
                           color_discrete_sequence=[LEGIT_BLUE])
                    .update_layout(showlegend=False, height=560), width="stretch")

# ===========================================================================
# 3. Transaction explorer
# ===========================================================================
with tabs[2]:
    st.subheader("Why was this transaction scored this way?")
    threshold = bundle["threshold"]
    test_df["outcome"] = np.select(
        [(test_df["score"] >= threshold) & (test_df[config.TARGET] == 1),
         (test_df["score"] >= threshold) & (test_df[config.TARGET] == 0),
         (test_df["score"] < threshold) & (test_df[config.TARGET] == 1)],
        ["Caught fraud (TP)", "False alarm (FP)", "Missed fraud (FN)"], default="Correctly OK (TN)")

    choice = st.radio("Show", ["Caught fraud (TP)", "False alarm (FP)", "Missed fraud (FN)", "Correctly OK (TN)"],
                      horizontal=True)
    pool = test_df[test_df["outcome"] == choice].sort_values("score", ascending=False)
    st.caption(f"{len(pool):,} transactions in this group (highest score first).")

    if pool.empty:
        st.info("No transactions in this group.")
    else:
        view = pool.head(500)
        labels = [f"{t:%Y-%m-%d %H:%M} · card …{str(c)[-4:]} · {cat} · ${a:,.2f} · score {sc:.3f}"
                  for t, c, cat, a, sc in zip(view[config.TIME_COL], view[config.CARD_COL],
                                              view["category"], view["amt"], view["score"])]
        idx = st.selectbox("Transaction", range(len(view)), format_func=lambda i: labels[i])
        row = view.iloc[[idx]]

        contrib = explain.contributions(bundle["pipeline"], row[config.FEATURES]).iloc[0]
        bias = contrib.pop("bias")
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(12)

        left, right = st.columns([3, 2])
        # Waterfall: start at the model's baseline and add each feature's push.
        wf = go.Figure(go.Waterfall(
            orientation="h",
            y=["baseline"] + [f"{f} = {explain._fmt(row[f].iat[0])}" for f in top.index][::-1] + ["final score"],
            x=[bias] + list(top.values[::-1]) + [0],
            measure=["absolute"] + ["relative"] * len(top) + ["total"],
            increasing=dict(marker_color=FRAUD_RED), decreasing=dict(marker_color=LEGIT_BLUE)))
        wf.update_layout(title="Feature contributions (log-odds; red = towards fraud)", height=520)
        left.plotly_chart(wf, width="stretch")

        right.metric("Fraud probability", f"{row['score'].iat[0]:.4f}",
                     "FLAGGED" if row["score"].iat[0] >= threshold else "not flagged", delta_color="off")
        right.metric("Actual label", "FRAUD" if row[config.TARGET].iat[0] else "legit")
        right.dataframe(row[config.FEATURES].T.rename(columns={row.index[0]: "value"}).astype(str),
                        height=360, width="stretch")

# ===========================================================================
# 4. Live scoring
# ===========================================================================
with tabs[3]:
    st.subheader("Score a brand-new transaction")
    st.caption("Pick a real card, invent a purchase, and the model scores it in real time using that "
               "card's full spending history (velocity + behavioural features).")
    history = load_history()
    cards = history[config.CARD_COL].drop_duplicates().sort_values()

    c1, c2, c3 = st.columns(3)
    card = c1.selectbox("Card", cards, format_func=lambda c: f"…{str(c)[-4:]}")
    card_hist = history[history[config.CARD_COL] == card]
    last = card_hist.iloc[-1]
    c1.caption(f"{len(card_hist):,} past transactions · avg ${card_hist['amt'].mean():,.2f} · "
               f"{last['city']}, {last['state']}")

    amount = c2.number_input("Amount ($)", 1.0, 30_000.0, 950.0, 10.0)
    category = c2.selectbox("Category", sorted(history["category"].astype(str).unique()),
                            index=sorted(history["category"].astype(str).unique()).index("shopping_net"))
    when_date = c3.date_input("Date", value=last[config.TIME_COL].date())
    when_time = c3.time_input("Time", value=dtime(2, 30))
    distance = c3.slider("Merchant distance from home (km)", 0, 3000, 50)

    if st.button("Score transaction", type="primary"):
        when = datetime.combine(when_date, when_time)
        # Place the merchant `distance` km north-east of the card-holder's home
        # (1 degree of latitude ~ 111 km).
        offset = distance / 111 / np.sqrt(2)
        new = pd.DataFrame([{
            **last.to_dict(),
            config.TIME_COL: pd.Timestamp(when), "amt": amount, "category": category,
            "merchant": "Manual entry", "merch_lat": last["lat"] + offset,
            "merch_long": last["long"] + offset, config.TARGET: np.nan,
        }])
        result = predict.score(new, history=history, bundle=bundle, top_k=5).iloc[0]
        p = result["fraud_probability"]
        if p >= bundle["threshold"]:
            st.error(f"🚨 **FLAG FOR REVIEW** — fraud probability {p:.3f} (threshold {bundle['threshold']:.3f})")
        else:
            st.success(f"✅ **Approve** — fraud probability {p:.4f} (threshold {bundle['threshold']:.3f})")
        st.write("**Top reasons pushing towards fraud:**")
        for reason in result["top_reasons"].split(" | "):
            if reason:
                st.write(f"- `{reason}`")

# ===========================================================================
# 5. Geography
# ===========================================================================
with tabs[4]:
    df = load_overview_data()
    state_stats = df.groupby("state", observed=True).agg(
        transactions=(config.TARGET, "size"), frauds=(config.TARGET, "sum"),
        fraud_rate=(config.TARGET, "mean")).reset_index()
    # Small states have noisy rates; hide those with very few transactions.
    min_txn = st.slider("Minimum transactions per state", 0, 20_000, 2_000, 500)
    shown = state_stats[state_stats["transactions"] >= min_txn]
    st.plotly_chart(
        px.choropleth(shown, locations="state", locationmode="USA-states", scope="usa",
                      color="fraud_rate", color_continuous_scale="Reds",
                      hover_data={"transactions": ":,", "frauds": ":,", "fraud_rate": ":.2%"},
                      title="Fraud rate by card-holder state").update_layout(height=480),
        width="stretch")

    # Where did the test-set frauds happen, and did the model catch them?
    tf = test_df[test_df[config.TARGET] == 1].copy()
    tf["result"] = np.where(tf["score"] >= bundle["threshold"], "caught", "missed")
    st.plotly_chart(
        px.scatter_geo(tf, lat="merch_lat", lon="merch_long", color="result", scope="usa",
                       color_discrete_map={"caught": LEGIT_BLUE, "missed": FRAUD_RED},
                       hover_data={"amt": ":$,.2f", "category": True},
                       title=f"Test-set fraud locations ({(tf['result'] == 'caught').mean():.1%} caught)")
        .update_layout(height=480), width="stretch")

# ===========================================================================
# 6. Customer segments
# ===========================================================================
with tabs[5]:
    seg_path = config.DATA_DIR / "card_segments.parquet"
    if not seg_path.exists():
        st.info("Segmentation was skipped. Re-run training without --skip-segments.")
    else:
        profiles = pd.read_parquet(seg_path)
        summary = pd.read_csv(config.REPORTS_DIR / "segments_summary.csv")
        st.subheader("Customer segments (K-Means on card-level behaviour)")
        st.dataframe(summary.style.format({
            "txn_per_month": "{:.1f}", "avg_amt": "${:.2f}", "monthly_spend": "${:,.0f}",
            "online_share": "{:.1%}", "night_share": "{:.1%}", "weekend_share": "{:.1%}", "age": "{:.0f}",
            "log_city_pop": "{:.1f}", "pct_cards_compromised": "{:.1%}", "fraud_rate": "{:.2%}"}),
            width="stretch")
        profiles["segment_label"] = profiles["segment"].astype(str) + ": " + profiles["segment_name"]
        left, right = st.columns(2)
        left.plotly_chart(px.scatter(profiles, x="avg_amt", y="txn_per_month", color="segment_label",
                                     log_x=True, hover_data=["state", "age"],
                                     title="Ticket size vs. frequency").update_layout(height=450),
                          width="stretch")
        right.plotly_chart(px.bar(summary, x="segment_name", y="pct_cards_compromised",
                                  title="Share of cards that suffered fraud, by segment",
                                  labels={"pct_cards_compromised": "Cards compromised", "segment_name": ""},
                                  color_discrete_sequence=[FRAUD_RED])
                           .update_layout(yaxis_tickformat=".0%", height=450),
                           width="stretch")
