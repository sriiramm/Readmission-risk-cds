import sys
from pathlib import Path

# --- Project root setup (CRITICAL for Streamlit imports) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import plotly.express as px

from src.data_load import load_raw, basic_clean
from src.preprocess import make_splits
from src.config import MODELS_DIR, RANDOM_STATE
from src.evaluate import projected_savings

st.set_page_config(page_title="Readmission Risk CDS", layout="wide")

CAPACITY_REPORT_PATH = PROJECT_ROOT / "reports" / "capacity_policy_report.json"


@st.cache_data
def load_capacity_report():
    if CAPACITY_REPORT_PATH.exists():
        with open(CAPACITY_REPORT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_data():
    return basic_clean(load_raw())


@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "xgb_model.joblib")


def risk_band(p: float) -> str:
    if p < 0.2:
        return "Low"
    if p < 0.5:
        return "Medium"
    return "High"


def topk_metrics(probs, y, k):
    top_idx = np.argsort(-probs)[:k]
    selected_y = y[top_idx]
    tp = int((selected_y == 1).sum())
    fp = int((selected_y == 0).sum())
    return top_idx, tp, fp


def simulate_daily_ops(probs, y, daily_discharges, k, days=200, seed=42):
    """
    Monte Carlo simulation: sample daily_discharges each day from the test cohort,
    intervene on top-k within that day, return avg TP/FP per day.
    """
    rng = np.random.default_rng(seed)
    n = len(probs)

    tp_list, fp_list = [], []
    for _ in range(days):
        day_idx = rng.choice(n, size=int(daily_discharges), replace=False if daily_discharges <= n else True)
        day_probs = probs[day_idx]
        day_y = y[day_idx]

        _, tp, fp = topk_metrics(day_probs, day_y, min(int(k), len(day_probs)))
        tp_list.append(tp)
        fp_list.append(fp)

    return float(np.mean(tp_list)), float(np.mean(fp_list))


@st.cache_resource
def get_shap_explainer(model, background_X):
    return shap.TreeExplainer(model, data=background_X, feature_perturbation="interventional")


def main():
    st.title("30-Day Readmission Risk — Clinical Decision Support (Demo)")
    st.caption("Educational decision support only. Not medical advice.")

    df = load_data()
    splits = make_splits(df, target="y", random_state=RANDOM_STATE)
    model = load_model()

    st.subheader("Executive Summary: Operational + Financial Impact (Capacity Constrained)")

    # Business inputs
    col1, col2, col3 = st.columns(3)
    cost_readmit = col1.number_input("Cost per readmission ($)", value=12000, step=500)
    cost_intervene = col2.number_input("Cost per intervention ($)", value=300, step=50)
    effectiveness = col3.slider("Intervention effectiveness (risk reduction)", 0.05, 0.50, 0.25, 0.01)
    avoidable_fraction = st.slider("Avoidable fraction of readmissions (realism)", 0.10, 1.00, 0.60, 0.05)


    st.markdown("### Daily operations controls")
    cA, cB, cC = st.columns(3)
    daily_discharges = cA.number_input("Daily discharges (avg)", value=100, step=10)
    max_interventions = cB.slider("Max interventions per day (staff capacity)", 10, 150, 40, 5)
    sim_days = cC.slider("Simulation days (stability)", 50, 500, 200, 50)

    probs = model.predict_proba(splits.X_test)[:, 1]
    y = splits.y_test.values

    # Daily simulation -> expected TP/FP per day at capacity K
    avg_tp, avg_fp = simulate_daily_ops(
        probs, y,
        daily_discharges=int(daily_discharges),
        k=int(max_interventions),
        days=int(sim_days),
        seed=42
    )

    sav_daily = projected_savings(
        avg_tp, avg_fp,
        cost_readmit, cost_intervene,
        effectiveness,
        avoidable_fraction=avoidable_fraction
    )


    # Report info (optional)
    capacity_report = load_capacity_report()
    if capacity_report and "best" in capacity_report:
        st.info(
            f"Policy: intervene on Top-K highest-risk patients (capacity constrained). "
            f"Report-suggested best K = {capacity_report['best']['k']} (based on test cohort net savings)."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg TP/day (within Top-K)", f"{avg_tp:.1f}")
    c2.metric("Avg FP/day (within Top-K)", f"{avg_fp:.1f}")
    c3.metric("Prevented readmissions/day (expected)", f"{sav_daily['prevented_readmissions']:.1f}")
    c4.metric("Net savings/day (expected)", f"${sav_daily['net_savings']:,.0f}")
    precision_at_k = avg_tp / max(avg_tp + avg_fp, 1e-9)
    st.caption(f"Precision at K={max_interventions}: {precision_at_k:.1%} (TP / (TP + FP))")
    st.caption(f"Interventions/day (capacity): {max_interventions} | Avg contacted/day: {max_interventions}")


    annual_savings = sav_daily["net_savings"] * 365
    st.metric("Estimated annual net savings (scaled from daily)", f"${annual_savings:,.0f}")

    st.divider()

    # Option A: ROI curve
    st.markdown("### Capacity vs ROI (diminishing returns)")
    k_grid = list(range(10, 151, 10))
    rows = []
    for k in k_grid:
        tp_k, fp_k = simulate_daily_ops(
            probs, y,
            daily_discharges=int(daily_discharges),
            k=int(k),
            days=int(sim_days),
            seed=42
        )
        sav_k = projected_savings(
            tp_k, fp_k,
            cost_readmit, cost_intervene,
            effectiveness,
            avoidable_fraction=avoidable_fraction
        )

        rows.append({
            "K_interventions_per_day": k,
            "Avg_TP_per_day": tp_k,
            "Avg_FP_per_day": fp_k,
            "Annual_net_savings": sav_k["net_savings"] * 365,
        })

    roi_df = pd.DataFrame(rows)
    fig_roi = px.line(
        roi_df,
        x="K_interventions_per_day",
        y="Annual_net_savings",
        title="Annual Net Savings vs Daily Intervention Capacity (Top-K policy)"
    )
    st.plotly_chart(fig_roi, use_container_width=True)
    st.caption("As capacity increases, ROI typically improves then shows diminishing returns as false positives rise.")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Patient View", "Population View", "Fairness View"])

    with tab1:
        idx = st.selectbox("Select a test encounter (row index)", splits.X_test.index.tolist())
        row = splits.X_test.loc[[idx]]
        prob = float(model.predict_proba(row)[:, 1][0])

        st.metric("Predicted 30-day readmission risk", f"{prob:.3f}", risk_band(prob))
        st.dataframe(row)

        st.markdown("### Why is this patient high risk? (SHAP explanation)")
        bg = splits.X_train.sample(n=min(200, len(splits.X_train)), random_state=RANDOM_STATE)
        explainer = get_shap_explainer(model, bg)

        shap_vals = explainer.shap_values(row)
        shap_series = pd.Series(shap_vals[0], index=row.columns)
        top = shap_series.abs().sort_values(ascending=False).head(10).index

        shap_df = pd.DataFrame({
            "feature": top,
            "value": row.iloc[0][top].values,
            "shap": shap_series[top].values
        }).sort_values("shap")

        fig_shap = px.bar(
            shap_df,
            x="shap",
            y="feature",
            orientation="h",
            title="Top 10 feature contributions (SHAP) for this patient"
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with tab2:
        fig = px.histogram(probs, nbins=30, title="Predicted Risk Distribution (Test Set)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Fairness audit (under Top-K capacity decision policy)")
        st.caption("Compares selection rate / TPR / FPR across groups at the current capacity K.")

        # Global Top-K selection on the test set (snapshot)
        top_idx_global = np.argsort(-probs)[:int(max_interventions)]
        selected = np.zeros(len(splits.X_test), dtype=int)
        selected[top_idx_global] = 1

        test_raw = df.loc[splits.X_test.index].copy()
        test_raw["y_true"] = y
        test_raw["selected"] = selected

        candidate_cols = [c for c in ["race", "gender", "age"] if c in test_raw.columns]
        if not candidate_cols:
            st.warning("Fairness columns not found (race/gender/age). Ensure preprocessing preserves these fields.")
            st.stop()

        group_col = st.selectbox("Group by", candidate_cols)

        def group_metrics(df_g):
            sel_rate = df_g["selected"].mean()
            pos = df_g[df_g["y_true"] == 1]
            tpr = pos["selected"].mean() if len(pos) > 0 else np.nan
            neg = df_g[df_g["y_true"] == 0]
            fpr = neg["selected"].mean() if len(neg) > 0 else np.nan
            return pd.Series({
                "selection_rate": sel_rate,
                "tpr_selected_among_positives": tpr,
                "fpr_selected_among_negatives": fpr,
                "count": len(df_g),
            })

        metrics_by_group = (
            test_raw.dropna(subset=[group_col])
            .groupby(group_col)
            .apply(group_metrics)
            .reset_index()
            .sort_values("count", ascending=False)
        )

        st.dataframe(metrics_by_group)

        st.plotly_chart(px.bar(metrics_by_group, x=group_col, y="selection_rate", title="Selection Rate by Group"), use_container_width=True)
        st.plotly_chart(px.bar(metrics_by_group, x=group_col, y="tpr_selected_among_positives", title="TPR (Selected | True Readmission) by Group"), use_container_width=True)
        st.plotly_chart(px.bar(metrics_by_group, x=group_col, y="fpr_selected_among_negatives", title="FPR (Selected | No Readmission) by Group"), use_container_width=True)

        st.caption("Large gaps suggest the allocation policy may impact groups differently; pair with clinical review and mitigation.")

if __name__ == "__main__":
    main()

