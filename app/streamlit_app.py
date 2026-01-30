import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import json
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px


from src.data_load import load_raw, basic_clean
from src.preprocess import make_splits
from src.config import MODELS_DIR, RANDOM_STATE
from src.evaluate import projected_savings

st.set_page_config(page_title="Readmission Risk CDS", layout="wide")

@st.cache_data
def load_data():
    return basic_clean(load_raw())

@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "xgb_model.joblib")

def risk_band(p):
    if p < 0.2: return "Low"
    if p < 0.5: return "Medium"
    return "High"

def main():
    st.title("30-Day Readmission Risk — Clinical Decision Support (Demo)")
    st.caption("Educational decision support only. Not medical advice.")

    df = load_data()
    splits = make_splits(df, target="y", random_state=RANDOM_STATE)
    model = load_model()

    st.subheader("Executive Summary: Projected Financial Impact")

    col1, col2, col3 = st.columns(3)
    cost_readmit = col1.number_input("Cost per readmission ($)", value=12000, step=500)
    cost_intervene = col2.number_input("Cost per intervention ($)", value=300, step=50)
    effectiveness = col3.slider("Intervention effectiveness (risk reduction)", 0.05, 0.50, 0.25, 0.01)

    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)

    probs = model.predict_proba(splits.X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    y = splits.y_test.values

    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())

    sav = projected_savings(tp, fp, cost_readmit, cost_intervene, effectiveness)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TP flagged", f"{tp}")
    c2.metric("FP flagged", f"{fp}")
    c3.metric("Prevented readmissions (expected)", f"{sav['prevented_readmissions']:.1f}")
    c4.metric("Net savings (test cohort)", f"${sav['net_savings']:,.0f}")

    st.divider()

    tab1, tab2 = st.tabs(["Patient View", "Population View"])

    with tab1:
        idx = st.selectbox("Select a test encounter (row index)", splits.X_test.index.tolist())
        row = splits.X_test.loc[[idx]]
        prob = float(model.predict_proba(row)[:, 1][0])
        st.metric("Predicted 30-day readmission risk", f"{prob:.3f}", risk_band(prob))
        st.dataframe(row)

        st.info("Next step: add SHAP drivers here (patient-level explanation).")

    with tab2:
        fig = px.histogram(probs, nbins=30, title="Predicted Risk Distribution (Test Set)")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
