import threading
import requests
import streamlit as st
import uvicorn

from src.data_loader import load_data, get_data_summary
from src.filters import apply_sidebar_filters
from src.charts import (
    plot_tip_distribution,
    plot_total_bill_vs_tip,
    plot_avg_tip_by_day,
    plot_tip_boxplot_by_time,
)
import src.models.linear_regression as lr_model
import src.models.random_forest as rf_model
import src.models.deep_learning as dl_model
import os
from src.models.preprocessing import SAVED_DIR, prepare_data, ALL_FEATURES

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tips Dashboard V2", page_icon="💰", layout="wide")

# ── Start FastAPI in a background thread (once) ────────────────────────────────
API_URL = "http://127.0.0.1:8000"

def _start_api():
    from api.endpoints import app as fastapi_app
    uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="error")

if "api_started" not in st.session_state:
    t = threading.Thread(target=_start_api, daemon=True)
    t.start()
    st.session_state["api_started"] = True

# ── Train models if not yet saved ─────────────────────────────────────────────
@st.cache_resource
def ensure_models_trained(_df):
    if not os.path.exists(os.path.join(SAVED_DIR, "linear_regression.pkl")):
        lr_model.train(_df)
    if not os.path.exists(os.path.join(SAVED_DIR, "random_forest.pkl")):
        rf_model.train(_df)
    if not os.path.exists(os.path.join(SAVED_DIR, "deep_learning.pkl")):
        dl_model.train(_df)

# ── Load data ──────────────────────────────────────────────────────────────────
df_raw = load_data()
ensure_models_trained(df_raw)
df = apply_sidebar_filters(df_raw)

# ── Shared input widget helper ─────────────────────────────────────────────────
def render_input_form(prefix: str):
    c1, c2 = st.columns(2)
    total_bill = c1.slider("Total Bill ($)", 1.0, 60.0, 18.0, 0.5, key=f"{prefix}_bill")
    size = c2.slider("Party Size", 1, 6, 2, key=f"{prefix}_size")
    c3, c4, c5, c6 = st.columns(4)
    sex = c3.selectbox("Gender", ["Male", "Female"], key=f"{prefix}_sex")
    smoker = c4.selectbox("Smoker", ["No", "Yes"], key=f"{prefix}_smoker")
    day = c5.selectbox("Day", ["Fri", "Sat", "Sun", "Thur"], key=f"{prefix}_day")
    time = c6.selectbox("Time", ["Dinner", "Lunch"], key=f"{prefix}_time")
    return dict(total_bill=total_bill, size=size, sex=sex, smoker=smoker, day=day, time=time)

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("💰 Restaurant Tips Dashboard V2")
st.markdown("Explore tipping patterns and **predict tip amounts** using three ML models. Use the sidebar to filter the data interactively.")
st.divider()

# SECTION 1 — Overview
st.header("📊 Dataset Overview")
summary = get_data_summary(df_raw)
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", summary["n_rows"])
col2.metric("Columns", summary["n_cols"])
col3.metric("Records after filters", len(df))
st.subheader("Raw Data Preview")
st.dataframe(df, use_container_width=True)
with st.expander("Column types & missing values"):
    col_a, col_b = st.columns(2)
    col_a.write("**Data types**"); col_a.json(summary["dtypes"])
    col_b.write("**Missing values**"); col_b.json(summary["missing_values"])
st.divider()

# SECTION 2 — Statistics
st.header("🔍 Filtered Data — Basic Statistics")
st.dataframe(df.describe(), use_container_width=True)
st.divider()

# SECTION 3 — Visualizations
st.header("📈 Visualizations")
if df.empty:
    st.warning("No data matches the current filters.")
else:
    r1c1, r1c2 = st.columns(2)
    with r1c1: st.plotly_chart(plot_tip_distribution(df), use_container_width=True)
    with r1c2: st.plotly_chart(plot_total_bill_vs_tip(df), use_container_width=True)
    r2c1, r2c2 = st.columns(2)
    with r2c1: st.plotly_chart(plot_avg_tip_by_day(df), use_container_width=True)
    with r2c2: st.plotly_chart(plot_tip_boxplot_by_time(df), use_container_width=True)
st.divider()

# SECTION 4 — Key Insights
st.header("💡 Key Insights")
if not df.empty:
    avg_tip = df["tip"].mean()
    avg_pct = (df["tip"] / df["total_bill"] * 100).mean()
    best_day = df.groupby("day", observed=True)["tip"].mean().idxmax()
    top_tipper = df.loc[df["tip"].idxmax()]
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Average Tip", f"${avg_tip:.2f}")
    i2.metric("Average Tip %", f"{avg_pct:.1f}%")
    i3.metric("Best Tipping Day", str(best_day))
    i4.metric("Highest Single Tip", f"${top_tipper['tip']:.2f}")
st.divider()

# SECTION 5 — Linear Regression
st.header("🤖 Linear Regression — Tip Predictor")
with st.expander("ℹ️ Model metrics", expanded=False):
    try:
        m, sc, en = lr_model.load_model()
        _, X_te, _, y_te, _, _ = prepare_data(df_raw)
        metrics = lr_model.evaluate(m, X_te, y_te)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("MAE", f"${metrics['mae']}"); mc2.metric("RMSE", f"${metrics['rmse']}"); mc3.metric("R²", metrics['r2'])
    except Exception as e:
        st.warning(f"Could not load metrics: {e}")
inputs_lr = render_input_form("lr")
if st.button("🔮 Predict with Linear Regression", key="btn_lr"):
    try:
        resp = requests.post(f"{API_URL}/predict/linear", json=inputs_lr, timeout=5)
        result = resp.json()
        st.success(f"💰 Predicted Tip (via API): **${result['predicted_tip']:.2f}**")
    except Exception:
        tip = lr_model.predict(**inputs_lr)
        st.success(f"💰 Predicted Tip: **${tip:.2f}**")
st.divider()

# SECTION 6 — Random Forest
st.header("🌲 Random Forest — Tip Predictor")
with st.expander("ℹ️ Model metrics & feature importances", expanded=False):
    try:
        import plotly.express as px, pandas as pd
        m_rf, _, _ = rf_model.load_model()
        _, X_te, _, y_te, _, _ = prepare_data(df_raw)
        metrics_rf = rf_model.evaluate(m_rf, X_te, y_te)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("MAE", f"${metrics_rf['mae']}"); mc2.metric("RMSE", f"${metrics_rf['rmse']}"); mc3.metric("R²", metrics_rf['r2'])
        fi = pd.DataFrame({"Feature": ALL_FEATURES, "Importance": m_rf.feature_importances_}).sort_values("Importance", ascending=True)
        fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h", title="Feature Importances", color="Importance", color_continuous_scale="Greens")
        st.plotly_chart(fig_fi, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load metrics: {e}")
inputs_rf = render_input_form("rf")
if st.button("🔮 Predict with Random Forest", key="btn_rf"):
    try:
        resp = requests.post(f"{API_URL}/predict/forest", json=inputs_rf, timeout=5)
        result = resp.json()
        st.success(f"💰 Predicted Tip (via API): **${result['predicted_tip']:.2f}**")
    except Exception:
        tip = rf_model.predict(**inputs_rf)
        st.success(f"💰 Predicted Tip: **${tip:.2f}**")
st.divider()

# SECTION 7 — Deep Learning
st.header("🧠 Deep Learning (MLP) — Tip Predictor")
with st.expander("ℹ️ Model metrics & architecture", expanded=False):
    try:
        m_dl, _, _ = dl_model.load_model()
        _, X_te, _, y_te, _, _ = prepare_data(df_raw)
        metrics_dl = dl_model.evaluate(m_dl, X_te, y_te)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("MAE", f"${metrics_dl['mae']}"); mc2.metric("RMSE", f"${metrics_dl['rmse']}")
        mc3.metric("R²", metrics_dl['r2']); mc4.metric("Iterations", m_dl.n_iter_)
        st.caption("Architecture: Input → Dense(64, ReLU) → Dense(32, ReLU) → Output(tip)")
    except Exception as e:
        st.warning(f"Could not load metrics: {e}")
inputs_dl = render_input_form("dl")
if st.button("🔮 Predict with Deep Learning", key="btn_dl"):
    try:
        resp = requests.post(f"{API_URL}/predict/deep", json=inputs_dl, timeout=5)
        result = resp.json()
        st.success(f"💰 Predicted Tip (via API): **${result['predicted_tip']:.2f}**")
    except Exception:
        tip = dl_model.predict(**inputs_dl)
        st.success(f"💰 Predicted Tip: **${tip:.2f}**")
