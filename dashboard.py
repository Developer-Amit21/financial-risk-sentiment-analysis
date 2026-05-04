import os
import gdown
import streamlit as st
import shutil
from pathlib import Path

BASE_DIR = "project_assets"
FOLDER_ID = "1B39WeJHcArkK12_WsRO968ZVCbqT9mCP"

@st.cache_resource
def setup_project():
    if not os.path.exists(BASE_DIR):
        st.write("📥 Downloading project files...")

        gdown.download_folder(
            id=FOLDER_ID,
            output=BASE_DIR,
            quiet=False,
            use_cookies=False
        )

    # -----------------------
    # AUTO DETECT REAL ROOT
    # -----------------------
    contents = os.listdir(BASE_DIR)

    # If 'data' is not directly inside BASE_DIR → go one level deeper
    if "data" in contents:
        REAL_ROOT = BASE_DIR
    else:
        REAL_ROOT = os.path.join(BASE_DIR, contents[0])

    # -----------------------
    # VALIDATION
    # -----------------------
    required_path = os.path.join(REAL_ROOT, "data", "final_credible_news_output.csv")

    if not os.path.exists(required_path):
        st.error(f"❌ Missing file: {required_path}")
        st.stop()

    return Path(REAL_ROOT)


PROJECT_ROOT = setup_project()

# -----------------------
# PATHS (NOW CORRECT)
# -----------------------
DATA_DIR = PROJECT_ROOT / "data"

NEWS_OUTPUT_PATH = DATA_DIR / "final_credible_news_output.csv"
XGB_MODEL_PATH = PROJECT_ROOT / "final_xgb_model.pkl"


import requests
import pandas as pd
import joblib
import torch
import torch.nn.functional as F
import numpy as np
import plotly.express as px
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from textblob import TextBlob
from sklearn.metrics import confusion_matrix
from datetime import datetime
from pathlib import Path


# --- Page Setup ---
st.set_page_config(page_title="AlphaGuard Risk Dashboard", layout="wide")
st.title("AlphaGuard | Financial Risk Intelligence")

# Keep Streamlit inference on CPU. A 4 GB GPU is too small for the cached dashboard
# models plus notebook sessions, and CUDA OOM stops the whole app.
device = torch.device("cpu")

# --- Logic Functions ---
def bin3(x):
    return "Low" if x < 0.35 else "Mid" if x < 0.70 else "High"

NEWS_TYPE_KEYWORDS = {
    "Market Trends": ["stock", "stocks", "market", "bull", "bear", "index", "nasdaq", "dow", "s&p", "rally", "selloff", "volatility", "equity", "shares"],
    "Economic Policy": ["central bank", "fed", "federal reserve", "ecb", "boe", "interest rate", "inflation", "gdp", "monetary", "fiscal", "policy", "tariff", "sanction"],
    "Cryptocurrency": ["crypto", "cryptocurrency", "bitcoin", "ethereum", "blockchain", "token", "stablecoin", "defi", "nft", "exchange", "wallet"],
    "Corporate Earnings": ["earnings", "quarterly", "q1", "q2", "q3", "q4", "profit", "revenue", "eps", "guidance", "merger", "acquisition", "m&a", "buyout"],
    "Commodities": ["gold", "silver", "oil", "crude", "brent", "wti", "natural gas", "commodity", "agriculture", "wheat", "corn", "soybean", "copper", "export"],
}

def classify_news_type(text):
    t = str(text).lower()
    scores = {label: sum(1 for kw in keywords if kw in t) for label, keywords in NEWS_TYPE_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other Financial News"

def safe_float_range(series, fallback=(0.0, 1.0)):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty: return fallback
    return float(values.min()), float(values.max())

def safe_int_range(series, fallback=(0, 0)):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty: return fallback
    return int(np.floor(values.min())), int(np.ceil(values.max()))

def float_range_slider(container, label, series, step=0.01, fallback=(0.0, 1.0)):
    container = container if container else st.sidebar
    lo, hi = safe_float_range(series, fallback=fallback)
    if np.isclose(lo, hi):
        container.caption(f"{label}: fixed at {lo:.3f}")
        return lo, hi
    return container.slider(label, min_value=float(lo), max_value=float(hi), value=(float(lo), float(hi)), step=step)

def int_range_slider(container, label, series, fallback=(0, 0)):
    container = container if container else st.sidebar
    lo, hi = safe_int_range(series, fallback=fallback)
    if lo == hi:
        container.caption(f"{label}: fixed at {lo}")
        return lo, hi
    return container.slider(label, min_value=int(lo), max_value=int(hi), value=(int(lo), int(hi)), step=1)

# --- Asset Loading ---
@st.cache_resource
def load_all_assets():
    rob_tok = AutoTokenizer.from_pretrained("distilroberta-base")
    rob_mod = AutoModel.from_pretrained("distilroberta-base").to(device)
    fin_tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    fin_mod = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

    df = pd.read_csv(NEWS_OUTPUT_PATH)

    # CORRECTED: Robust Date Processing
    if 'Time' in df.columns:
        df['date_dt'] = pd.to_datetime(df['Time'], errors='coerce')
    else:
        df['date_dt'] = pd.to_datetime(datetime.now())

    xgb_model = joblib.load(XGB_MODEL_PATH)

    features = ["fin_pos", "fin_neg", "extraction_confidence", "event_risk", "subjectivity", "source_reliability"]
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    df["predicted_risk"] = xgb_model.predict(df[features].fillna(0))
    df["risk_actual"] = df["risk_score"].apply(lambda x: bin3(x) if pd.notna(x) else "Unknown")
    df["risk_pred"] = df["predicted_risk"].apply(bin3)
    df["topic_label"] = df["topic_label"].astype(str)

    if {"fin_pos", "fin_neg", "fin_neu"}.issubset(df.columns):
        df["sent_label"] = df[["fin_pos", "fin_neg", "fin_neu"]].idxmax(axis=1).str.replace("fin_", "", regex=False)
    else:
        df["sent_label"] = "unknown"

    text_for_type = df["Headlines"].fillna("").astype(str) + " " + df["Description"].fillna("").astype(str)
    df["news_type"] = text_for_type.apply(classify_news_type)

    return rob_tok, rob_mod, fin_tok, fin_mod, xgb_model, df

try:
    rob_tok, rob_mod, fin_tok, fin_mod, xgb_model, df = load_all_assets()
except Exception as exc:
    st.error(f"Critical Error: {exc}")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Intelligence Filters")

# NEW: Multi-Range Time Filter
time_range = st.sidebar.selectbox(
    "🕒 Select Time Range",
    ["All Time", "Today", "Last 24 Hours", "Last 48 Hours", "Last 7 Days"]
)

search = st.sidebar.text_input("Search Headlines", "")

topic_counts = df["topic_label"].dropna().astype(str).value_counts()
topic_options = ["All Topics"] + topic_counts.index.tolist()
selected_topic = st.sidebar.selectbox("Topic", topic_options, index=0)

news_type_counts = df["news_type"].value_counts()
news_type_options = ["All Types"] + [f"{name} ({count})" for name, count in news_type_counts.items()]
news_type_map = {f"{name} ({count})": name for name, count in news_type_counts.items()}
selected_type_display = st.sidebar.selectbox("Financial News Type", news_type_options, index=0)
selected_news_type = news_type_map.get(selected_type_display, "All Types")

min_cred = st.sidebar.slider("Min Credibility (Quick)", 0.0, 1.0, 0.4, 0.01)
risk_min, risk_max = float_range_slider(st.sidebar, "Risk Score Range", df["risk_score"])
src_min, src_max = float_range_slider(st.sidebar, "Source Reliability Range", df["source_reliability"])

if "red_flags_count" in df.columns:
    red_min, red_max = int_range_slider(st.sidebar, "Red Flag Count Range", df["red_flags_count"])
else:
    red_min, red_max = None, None

fin_box = st.sidebar.expander("Fin Sentiment Filters", expanded=False)
with fin_box:
    pos_min, pos_max = float_range_slider(fin_box, "fin_pos Range", df["fin_pos"])
    neg_min, neg_max = float_range_slider(fin_box, "fin_neg Range", df["fin_neg"])
    neu_min, neu_max = (float_range_slider(fin_box, "fin_neu Range", df["fin_neu"]) if "fin_neu" in df.columns else (0.0, 1.0))

# --- Apply Filter Logic ---
mask = pd.Series(True, index=df.index)

# NEW: Corrected Time Filtering
now = datetime.now()
if time_range == "Today":
    mask &= df["date_dt"].dt.date == now.date()
elif time_range == "Last 24 Hours":
    mask &= df["date_dt"] >= (now - pd.Timedelta(hours=24))
elif time_range == "Last 48 Hours":
    mask &= df["date_dt"] >= (now - pd.Timedelta(hours=48))
elif time_range == "Last 7 Days":
    mask &= df["date_dt"] >= (now - pd.Timedelta(days=7))

mask &= df["source_reliability"] >= min_cred
mask &= df["risk_score"].between(risk_min, risk_max)
mask &= df["source_reliability"].between(src_min, src_max)
mask &= df["fin_pos"].between(pos_min, pos_max)
mask &= df["fin_neg"].between(neg_min, neg_max)
if selected_topic != "All Topics": mask &= df["topic_label"].astype(str) == selected_topic
if selected_news_type != "All Types": mask &= df["news_type"] == selected_news_type
if search: mask &= df["Headlines"].astype(str).str.contains(search, case=False, na=False)

filtered_df = df[mask].copy()

# --- Tabs ---
t1, t2, t3 = st.tabs(["News Feed", "Performance Analytics", "AI Validator"])

with t1:
    if st.button("🚀 Fetch & Analyze All Sources"):
        with st.status("Gathering Multi-Source Intelligence...", expanded=True) as status:
            # 1. Scraping
            st.write("Scraping Reuters, CNBC, and Bloomberg...")
            raw_live_df = fetch_multi_source_finance()

            # 2. AI Analysis
            st.write("Running AI Risk & Sentiment Models...")
            analyzed_df = process_new_data(raw_live_df, rob_tok, rob_mod, fin_tok, fin_mod, xgb_model)

            # 3. Saving
            analyzed_df.to_csv(NEWS_OUTPUT_PATH, mode='a', header=False, index=False)
            status.update(label="Market Analysis Complete!", state="complete")
            st.rerun()

    # --- Display Logic ---
    for _, row in filtered_df.head(20).iterrows():
        # Display the Date clearly in the title or caption
        display_date = row['Time'] if pd.notna(row['Time']) else "Date Unknown"

        with st.expander(f"{row['Headlines']} | 📅 {display_date}"):
            st.write(f"**Description:** {row['Description']}")
            st.write(f"**Risk Level:** {bin3(row['risk_score'])} ({row['risk_score']:.2f})")
            st.caption(f"Source: {row.get('source_id', 'Unknown')} | AI Confidence: {row['extraction_confidence']:.2f}")
with t2:
    st.header("Project Evaluation (%)")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risk Distribution")
        res = pd.DataFrame({"Count": filtered_df["risk_actual"].value_counts(), 
                           "Percentage": (filtered_df["risk_actual"].value_counts(normalize=True)*100).map("{:.1f}%".format)})
        st.table(res)
    with c2:
        st.subheader("Model Accuracy (Risk)")
        labels = ["Low", "Mid", "High"]
        cm = confusion_matrix(filtered_df["risk_actual"], filtered_df["risk_pred"], labels=labels)
        fig = px.imshow(cm, text_auto=True, x=labels, y=labels, color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("News Distribution by Financial Type")
    type_dist = filtered_df["news_type"].value_counts().reset_index()
    st.plotly_chart(px.bar(type_dist, x='news_type', y='count', color='news_type'), use_container_width=True)

    # ... Other original charts preserved ...
    st.subheader("Parameter Distributions")
    dist_cols = ["risk_score", "predicted_risk", "source_reliability", "fin_pos", "fin_neg", "subjectivity", "event_risk"]
    dist_cols = [c for c in dist_cols if c in filtered_df.columns]
    if dist_cols:
        melted = filtered_df[dist_cols].melt(var_name="parameter", value_name="value").dropna()
        st.plotly_chart(px.histogram(melted, x="value", facet_col="parameter", facet_col_wrap=3, nbins=40), use_container_width=True)

with t3:
    st.subheader("Real-time Headline Check")
    h_in = st.text_input("Enter Headline:")
    if st.button("Analyze") and h_in:
        with torch.no_grad():
            e = rob_mod(**rob_tok(h_in, return_tensors="pt").to(device)).last_hidden_state[:,0,:].cpu().numpy()
            conf = (np.linalg.norm(e) - 5.0) / 15.0
            p = F.softmax(fin_mod(**fin_tok(h_in, return_tensors="pt").to(device)).logits, dim=-1).cpu().numpy()[0]
            fv = pd.DataFrame([[p[0], p[1], conf, 0.3, TextBlob(h_in).sentiment.subjectivity, 0.9]], 
                              columns=["fin_pos","fin_neg","extraction_confidence","event_risk","subjectivity","source_reliability"])
            risk = float(xgb_model.predict(fv)[0])
            st.write(f"Predicted Risk: **{risk:.2f}** | Risk Class: **{bin3(risk)}** | Confidence: **{conf:.2f}**")

