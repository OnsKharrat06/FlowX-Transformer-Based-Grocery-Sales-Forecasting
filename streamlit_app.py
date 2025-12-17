import json
import tempfile
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm
import torch
from pathlib import Path
from datetime import datetime, timedelta



# -----------------------------
# Paths
# -----------------------------

JIT_PATH = Path("artifacts/flowx_jit.pt")
BUNDLE_PATH = Path("artifacts/model.pt")
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_INFO_PATH = CONFIG_DIR / "model.json"

META_PATH = CONFIG_DIR / "dashboard_meta.json"
KPIS_PATH = CONFIG_DIR / "kpis.json"
PROMPT_PATH = CONFIG_DIR / "prompt_template.txt"

SALES_OVER_TIME_PATH = DATA_DIR / "sales_over_time.csv"
AVP_PATH = DATA_DIR / "actual_vs_predicted.csv"

LATEST_FORECAST_PATH = ARTIFACTS_DIR / "latest_forecast.json"


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Admin Dashboard", layout="wide")


# -----------------------------
# Custom CSS Styles
# -----------------------------
st.markdown("""
<style>
.meta-card {
    background-color: #f9f9fb;
    padding: 14px;
    border-radius: 8px;
    border: 1px solid #e6e6eb;
}
.meta-title {
    font-size: 0.8rem;
    color: #666;
    margin-bottom: 4px;
}
.meta-value {
    font-size: 0.95rem;
    font-weight: 500;
    word-wrap: break-word;
    white-space: normal;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# File loaders (cached)
# -----------------------------
@st.cache_resource
def load_jit_model():
    m = torch.jit.load(str(JIT_PATH), map_location="cpu")
    m.eval()
    return m

@st.cache_resource
def load_bundle():
    # contains model_config, feature_engineer_scalers, holiday_embeddings
    return torch.load(str(BUNDLE_PATH), map_location="cpu",weights_only=False)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# -----------------------------
# Ollama + PDF helpers
# -----------------------------
def call_ollama(prompt: str, model: str = "llama3") -> str:
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["response"]

def text_to_pdf(report_text: str) -> bytes:
    styles = getSampleStyleSheet()
    story = []

    for line in report_text.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.3 * cm))
        elif line.startswith("##"):
            story.append(Paragraph(f"<b>{line[2:].strip()}</b>", styles["Heading2"]))
        else:
            story.append(Paragraph(line, styles["BodyText"]))

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        doc = SimpleDocTemplate(
            tmp.name,
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )
        doc.build(story)
        tmp.seek(0)
        return tmp.read()


# -----------------------------
# Placeholder forecast generator (swap later with real model)
# -----------------------------
def run_forecast_placeholder(store_id: int, family: str, horizon_days: int, promo: float, oil: str, holidays: bool) -> dict:
    """
    Later: replace the body with your real model inference.
    Must return an object matching the latest_forecast.json schema.
    """
    start = pd.Timestamp("2017-08-01")
    dates = pd.date_range(start, periods=horizon_days, freq="D")
    # simple placeholder series
    y = (np.random.rand(horizon_days) * 400 + 1200).round(2)

    total = float(y.sum())
    avg = float(y.mean())

    # placeholder weights
    weights = {"historical_patterns_weight": 0.64, "promotion_calendar_weight": 0.26, "external_factors_weight": 0.10}

    out = {
        "store_information": {"store_id": store_id, "store_name": f"Store {store_id}", "region": "N/A"},
        "forecast_period": {
            "start_date": str(dates.min().date()),
            "end_date": str(dates.max().date()),
            "horizon_days": horizon_days,
        },
        "scenario_assumptions": {
            "promotion_intensity": promo,
            "oil_price_assumption": oil,
            "holidays_included": holidays,
            "product_family": family,
        },
        "sales_forecast_summary": {
            "total_forecast_sales": round(total, 2),
            "average_daily_sales": round(avg, 2),
            "expected_trend": "stable",
        },
        "model_explainability": weights,
        "forecast_series": [{"date": str(d.date()), "forecasted_sales": float(v)} for d, v in zip(dates, y)],
        "admin_context": {
            "generated_by": "Retail Forecasting System",
            "generation_date": datetime.now().date().isoformat(),
            "notes": "Placeholder forecast. Replace by real model output later."
        }
    }
    return out


# -----------------------------
# Load config/data
# -----------------------------
meta = load_json(META_PATH)
kpis = load_json(KPIS_PATH)
model_info = load_json(MODEL_INFO_PATH)
prompt_template_default = load_text(PROMPT_PATH)

df_sales = load_csv(SALES_OVER_TIME_PATH)
df_sales["date"] = pd.to_datetime(df_sales["date"])

df_avp = load_csv(AVP_PATH)
df_avp["date"] = pd.to_datetime(df_avp["date"])


# -----------------------------
# Header
# -----------------------------
st.markdown(f"# {meta.get('app_title', 'Dashboard')}")
st.caption(meta.get("app_subtitle", ""))


# -----------------------------
# Tabs
# -----------------------------
tab_ops, tab_forecast, tab_model = st.tabs([
    "Operations Dashboard",
    "Forecasting and Planning",
    "Model Insights"
])


# =====================================================
# TAB 1: Operations Dashboard
# =====================================================
with tab_ops:
    st.subheader("Key Metrics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stores", f"{kpis.get('num_stores', '—')}")
    c2.metric("Total Sales", f"${int(kpis.get('total_sales', 0)):,}")
    c3.metric("Products", f"{kpis.get('num_products', '—')}")
    c4.metric("Promo Ratio", f"{int(float(kpis.get('promo_ratio', 0))*100)}%")

    

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Average daily sales", f"${int(kpis.get('avg_daily_sales', 0)):,}")
    c6.metric("Avg sales per store", f"${int(kpis.get('avg_sales_per_store', 0)):,}")
    c7.metric("Top product share", f"{int(float(kpis.get('top_product_share', 0))*100)}%")

    st.divider()
    
    # Sales over time: store selector
    st.subheader("Sales over time")
    stores_sales = sorted(df_sales["store_id"].unique().tolist())
    selected_store_sales = st.selectbox("Store", stores_sales, key="ops_sales_store")
    df_sales_store = df_sales[df_sales["store_id"] == selected_store_sales].sort_values("date")

    fig1 = px.line(df_sales_store, x="date", y="sales", title=f"Sales over time — Store {selected_store_sales}")
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    # Actual vs predicted: store selector
    st.subheader("Actual vs predicted")
    stores_avp = sorted(df_avp["store_id"].unique().tolist())
    selected_store_avp = st.selectbox("Store", stores_avp, key="ops_avp_store")
    df_avp_store = df_avp[df_avp["store_id"] == selected_store_avp].sort_values("date")

    fig2 = px.line(df_avp_store, x="date", y=["actual", "predicted"], title=f"Actual vs predicted — Store {selected_store_avp}")
    st.plotly_chart(fig2, use_container_width=True)


# =====================================================
# TAB 2: Forecasting and Planning (Forecast form restored)
# =====================================================

def run_forecast_from_ui(
    store_id: int,
    family: str,
    item_nbr: int,
    onpromotion: int,
    description: str,
    dcoilwtico: float,
):
    model = load_jit_model()
    bundle = load_bundle()

    cfg = bundle["model_config"]
    fe = bundle["feature_engineer_scalers"]
    holiday_embs = bundle["holiday_embeddings"]

    LOOKBACK = int(cfg["lookback_window"])
    HORIZON = int(cfg["forecast_horizon"])
    FUTURE_KNOWN_DIM = int(cfg["future_known_dim"])
    STATIC_DIM = int(cfg["static_dim"])

    # --- Build a minimal synthetic past window that depends on UI ---
    # (valid for scenario planning; later you can replace with real historical window)
    past = pd.DataFrame({
        "dcoilwtico": [dcoilwtico] * LOOKBACK,
        "transactions": [0.0] * LOOKBACK,
        "unit_sales": [0.0] * LOOKBACK,
        "store_nbr": [store_id] * LOOKBACK,
        "family": [family] * LOOKBACK,
        "item_nbr": [item_nbr] * LOOKBACK,
        "onpromotion": [onpromotion] * LOOKBACK,
        "perishable": [0] * LOOKBACK,
        "description": [description] * LOOKBACK,
    })

    # --- Scale numericals exactly like training (using saved centers/scales) ---
    def scale_col(col: str, values: np.ndarray) -> np.ndarray:
        # fallback if missing
        if col not in fe["scalers"]:
            return values.astype(np.float32)

        center = np.array(fe["scalers"][col]["center_"], dtype=np.float32).reshape(1,)
        scale = np.array(fe["scalers"][col]["scale_"], dtype=np.float32).reshape(1,)
        scale = np.where(scale == 0, 1.0, scale)
        return ((values.astype(np.float32) - center) / scale).astype(np.float32)

    # IMPORTANT: your training used numerical_features ['unit_sales','dcoilwtico','transactions']
    past_numerical_np = np.stack([
        scale_col("unit_sales", past["unit_sales"].values),
        scale_col("dcoilwtico", past["dcoilwtico"].values),
        scale_col("transactions", past["transactions"].values),
        np.sin(2 * np.pi * 0 / 7) * np.ones(LOOKBACK),  # dow_sin
        np.cos(2 * np.pi * 0 / 7) * np.ones(LOOKBACK),  # dow_cos
        np.sin(2 * np.pi * 8 / 12) * np.ones(LOOKBACK), # month_sin
        np.cos(2 * np.pi * 8 / 12) * np.ones(LOOKBACK), # month_cos
        np.sin(2 * np.pi * 200 / 365) * np.ones(LOOKBACK), # doy_sin
        np.cos(2 * np.pi * 200 / 365) * np.ones(LOOKBACK), # doy_cos
    ], axis=-1)  # (LOOKBACK, 9)

    past_numerical = torch.tensor(past_numerical_np, dtype=torch.float32).unsqueeze(0)

    # --- Encode categoricals using saved cat_maps ---
    def enc_col(col: str, values: np.ndarray) -> np.ndarray:
        cmap = fe["cat_maps"].get(col, {})
        # training stored keys as strings
        return np.array([cmap.get(str(v), 0) for v in values], dtype=np.int64)

    # IMPORTANT: your training categorical list includes:
    # ['family','store_nbr','item_nbr','holiday_type','onpromotion','day_of_week','month','perishable']
    # We only have some in UI; the rest we set deterministically.
    # month/year fixed to Aug 2017 in your constraints; we will set month=8, day_of_week=0.
    day_of_week = 0
    month = 8

    past_categorical_np = np.stack([
        enc_col("family", past["family"].values),
        enc_col("store_nbr", past["store_nbr"].values),
        enc_col("item_nbr", past["item_nbr"].values),
        enc_col("onpromotion", past["onpromotion"].values),
        enc_col("day_of_week", np.array([day_of_week]*LOOKBACK)),
        enc_col("month", np.array([month]*LOOKBACK)),
        enc_col("perishable", past["perishable"].values),
    ], axis=-1)  # (LOOKBACK, cat_dim)

    past_categorical = torch.tensor(past_categorical_np, dtype=torch.long).unsqueeze(0)

    # --- Holiday embeddings ---
    # holiday_embs keys are descriptions; if not found -> zeros
    any_emb = next(iter(holiday_embs.values()))
    EMB_DIM = int(np.array(any_emb).shape[0])
    default_emb = np.zeros((EMB_DIM,), dtype=np.float32)

    emb_vec = np.array(holiday_embs.get(str(description), default_emb), dtype=np.float32)
    past_holiday_embedding_np = np.stack([emb_vec]*LOOKBACK, axis=0)  # (LOOKBACK, EMB_DIM)
    future_holiday_embedding_np = np.stack([emb_vec]*HORIZON, axis=0) # (HORIZON, EMB_DIM)

    past_holiday_embedding = torch.tensor(past_holiday_embedding_np, dtype=torch.float32).unsqueeze(0)
    future_holiday_embedding = torch.tensor(future_holiday_embedding_np, dtype=torch.float32).unsqueeze(0)

    # --- Future known + static ---
    future_known = torch.zeros((1, HORIZON, FUTURE_KNOWN_DIM), dtype=torch.float32)
    static = torch.zeros((1, STATIC_DIM), dtype=torch.float32)

    # --- Run TorchScript model ---
    with torch.no_grad():
        # IMPORTANT: TorchScript trace uses POSITIONAL args (because we traced lambda)
        out = model(
            past_numerical,
            past_categorical,
            past_holiday_embedding,
            future_known,
            future_holiday_embedding,
            static
        )

    # out shape expected: (1, HORIZON, num_quantiles)
    out_np = out.squeeze(0).cpu().numpy()

    # Use median quantile if exists (you used preds[:,:,1] in predict)
    if out_np.ndim == 2:
        preds = out_np  # (H, ?) unlikely
        preds = preds[:, 0]
    else:
        q_idx = 1 if out_np.shape[-1] > 1 else 0
        preds = out_np[:, q_idx]  # (H,)

    # Build forecast series (dates just for display)
    start_date = datetime(2017, 9, 1)  # day after your dataset ends (Aug 2017)
    forecast_series = [
        {"date": (start_date + timedelta(days=i)).strftime("%Y-%m-%d"),
         "forecasted_sales": float(preds[i])}
        for i in range(HORIZON)
    ]

    return {
        "forecast_series": forecast_series,
        "ui_inputs": {
            "store_id": store_id,
            "family": family,
            "item_nbr": item_nbr,
            "onpromotion": onpromotion,
            "description": description,
            "dcoilwtico": dcoilwtico,
            "horizon_days": HORIZON,
        }
    }

with tab_forecast:
    st.subheader("Scenario planning")

    # -----------------------------
    # UI CONSTRAINTS (FINAL)
    # -----------------------------
    STORE_IDS = [44, 47, 45, 46, 3, 48, 8, 49, 50, 11]

    FAMILY_OPTIONS = [
        "POULTRY", "GROCERY I", "DAIRY", "CLEANING", "EGGS",
        "BEVERAGES", "BREAD/BAKERY", "PREPARED FOODS", "MEATS",
        "FROZEN FOODS", "LIQUOR,WINE,BEER", "HOME AND KITCHEN II",
        "HOME CARE", "PRODUCE", "PERSONAL CARE", "DELI",
        "GROCERY II", "SEAFOOD"
    ]

    HOLIDAY_TYPES = [
        "Work Day", "Holiday", "Event",
        "Additional", "Transfer", "Bridge"
    ]

    OIL_MIN = 26.19
    OIL_MAX = 54.48

    HORIZON_DAYS = 16  # 🔒 FIXED, NOT USER-EDITABLE

    left, right = st.columns([1, 2])

    # -----------------------------
    # INPUT FORM
    # -----------------------------
    with left:
        with st.form("forecast_form", clear_on_submit=False):

            store_id = st.selectbox("Store", STORE_IDS, key="fc_store")

            family = st.selectbox(
                "Product family",
                FAMILY_OPTIONS,
                key="fc_family"
            )

            item_nbr = st.number_input(
                "Item number (item_nbr)",
                min_value=1,
                step=1,
                value=1,
                key="fc_item_nbr"
            )

            onpromotion = st.selectbox(
                "On promotion",
                [0, 1],
                key="fc_onpromotion"
            )

            holiday_type = st.selectbox(
                "Holiday type",
                HOLIDAY_TYPES,
                key="fc_holiday_type"
            )

            description = st.text_input(
                "Holiday description",
                value="No Holiday",
                key="fc_description"
            )

            dcoilwtico = st.slider(
                "Oil price (dcoilwtico)",
                min_value=float(OIL_MIN),
                max_value=float(OIL_MAX),
                value=float((OIL_MIN + OIL_MAX) / 2),
                key="fc_dcoilwtico"
            )

            st.caption("Forecast horizon: **16 days** (fixed)")

            run_btn = st.form_submit_button("Run forecast")

    # -----------------------------
    # RUN FORECAST
    # -----------------------------
    if run_btn:
        forecast_obj = run_forecast_from_ui(
            store_id=int(store_id),
            family=str(family),
            item_nbr=int(item_nbr),
            onpromotion=int(onpromotion),
            description=str(description),
            dcoilwtico=float(dcoilwtico)
        )

        save_json(LATEST_FORECAST_PATH, forecast_obj)
        st.session_state["latest_forecast_obj"] = forecast_obj
        st.session_state["forecast_ready"] = True

    # -----------------------------
    # LOAD LAST FORECAST
    # -----------------------------
    if (
        "latest_forecast_obj" not in st.session_state
        and LATEST_FORECAST_PATH.exists()
    ):
        st.session_state["latest_forecast_obj"] = load_json(LATEST_FORECAST_PATH)
        st.session_state["forecast_ready"] = True

    # -----------------------------
    # DISPLAY OUTPUT + REPORT
    # -----------------------------
    if st.session_state.get("forecast_ready", False):
        forecast_obj = st.session_state["latest_forecast_obj"]
        series = pd.DataFrame(forecast_obj["forecast_series"])
        series["date"] = pd.to_datetime(series["date"])

        with right:
            st.subheader("Forecast output")
            fig_f = px.line(
                series,
                x="date",
                y="forecasted_sales",
                title="Forecasted sales (16 days)"
            )
            st.plotly_chart(fig_f, use_container_width=True)
            st.dataframe(series, use_container_width=True)

        st.divider()

        if "prompt_template" not in st.session_state:
            st.session_state["prompt_template"] = prompt_template_default

        with st.expander("Edit report prompt", expanded=False):
            st.session_state["prompt_template"] = st.text_area(
                "Prompt template",
                st.session_state["prompt_template"],
                height=280
            )

        st.subheader("Planning report")

        if st.button("Generate report"):
            with st.spinner("Generating report and PDF..."):
                with open(LATEST_FORECAST_PATH, "r", encoding="utf-8") as f:
                    report_input = json.load(f)

                final_prompt = st.session_state["prompt_template"].replace(
                    "{REPORT_INPUT_JSON}",
                    json.dumps(report_input, indent=2)
                )

                report_md = call_ollama(final_prompt)
                pdf_bytes = text_to_pdf(report_md)

                st.session_state["report_md"] = report_md
                st.session_state["report_pdf"] = pdf_bytes

        if "report_md" in st.session_state:
            st.markdown(st.session_state["report_md"])

        if "report_pdf" in st.session_state:
            st.download_button(
                "Download PDF",
                data=st.session_state["report_pdf"],
                file_name="planning_report.pdf",
                mime="application/pdf",
            )

    else:
        st.info("Run a forecast to generate predictions and a planning report.")


# =====================================================
# TAB 3: Model Insights
# =====================================================
# =====================================================
# TAB 3: Model Insights
# =====================================================
# =====================================================
# TAB 3: Model Insights (VISUAL VERSION)
# =====================================================
with tab_model:
    st.subheader("System & Model Overview")

    # -----------------------------
    # Top metadata row (wrapped text)
    # -----------------------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.caption("📦 Dataset")
        st.markdown(meta.get("dataset_name", "—"))

    with c2:
        st.caption("📅 Training period")
        st.markdown(meta.get("training_period", "—"))

    with c3:
        st.caption("⏱ Forecast horizon")
        st.markdown("16 days")

    with c4:
        st.caption("🌍 Dataset source")
        st.markdown(meta.get("dataset_source", "—"))

    st.divider()

    # -----------------------------
    # Dataset overview
    # -----------------------------
    st.subheader("📊 Dataset overview")
    st.markdown(meta.get("dataset_overview", "—"))

    st.divider()

    # -----------------------------
    # Model summary (compact)
    # -----------------------------
    model_block = model_info["model"]

    st.subheader("🧠 Model summary")

    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown(f"**Model name**")
        st.markdown(model_block["name"])

    with c2:
        st.markdown("**Model strategy**")
        st.markdown(model_block["strategy"])

    st.divider()

    # -----------------------------
    # Inputs — shown as columns
    # -----------------------------
    st.subheader("🔢 Model inputs")

    inputs = model_block["inputs"]

    col_num, col_cat, col_temp = st.columns(3)

    with col_num:
        st.markdown("**Numerical features**")
        df_num = pd.DataFrame(
            {"Feature": inputs["numerical_features"]}
        )
        st.dataframe(df_num, hide_index=True, use_container_width=True)

    with col_cat:
        st.markdown("**Categorical features**")
        df_cat = pd.DataFrame(
            {"Feature": inputs["categorical_features"]}
        )
        st.dataframe(df_cat, hide_index=True, use_container_width=True)

    with col_temp:
        st.markdown("**Temporal features**")
        df_temp = pd.DataFrame(
            {"Feature": inputs["temporal_features"]}
        )
        st.dataframe(df_temp, hide_index=True, use_container_width=True)

    st.divider()

    # -----------------------------
    # Text features (separate)
    # -----------------------------
    st.subheader("📝 Text features")

    tf = inputs["text_features"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Source", tf["source"])
    c2.metric("Embedding model", tf["embedding_model"])
    c3.metric("Embedding dim", tf["embedding_dimension"])

    st.divider()

    # -----------------------------
    # Architecture — structured view
    # -----------------------------
    st.subheader("🏗 Architecture overview")

    arch = model_block["architecture"]

    c1, c2, c3 = st.columns(3)

    c1.metric("Architecture", arch["type"])
    c2.metric("Attention", "Multi-Head")
    c3.metric("Positional encoding", "Sinusoidal")

    st.markdown("**Key architectural components**")

    arch_df = pd.DataFrame([
        {"Component": "Encoder layers", "Value": arch["encoder"]["layers"]},
        {"Component": "Decoder layers", "Value": arch["decoder"]["layers"]},
        {"Component": "Input window", "Value": arch["encoder"]["input_window"]},
        {"Component": "Forecast horizon", "Value": arch["decoder"]["forecast_horizon"]},
        {"Component": "Attention heads", "Value": arch["multi_head_attention"]["number_of_heads"]},
        {"Component": "Fusion module", "Value": arch["fusion_module"]},
    ])

    st.dataframe(arch_df, hide_index=True, use_container_width=True)

    st.divider()

    # -----------------------------
    # Output — quantiles as chart
    # -----------------------------
    

    # -----------------------------
    # Training configuration (clean list)
    # -----------------------------
    st.subheader("⚙ Training configuration")

    training = model_info["training"]

    col_opt, col_reg = st.columns(2)

    with col_opt:
        st.markdown("**Optimization**")
        for k, v in training["optimization"].items():
            st.markdown(f"- **{k.replace('_', ' ').title()}**: {v}")

    with col_reg:
        st.markdown("**Regularization**")
        for k, v in training["regularization"].items():
            st.markdown(f"- **{k.replace('_', ' ').title()}**: {v}")

    st.markdown(f"**Training goal:** {training['goal']}")

    st.divider()

    # -----------------------------
    # Explainability — visual + text
    # -----------------------------
    st.subheader("🔍 Explainability")

    explain = model_info.get("explainability", {})
    st.markdown(explain.get("description", "—"))

    st.markdown("**Methods used**")
    for m in explain.get("methods", []):
        st.markdown(f"- {m}")