from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nexus.analytics import (
    run_arima,
    run_association_rules,
    run_classification,
    run_clustering,
    run_regression,
)
from nexus.data import add_derived_fields, clean_headers, make_demo_data


APP_DIR = Path(__file__).parent
PLOT_TEMPLATE = "plotly_dark"

st.set_page_config(
    page_title="NEXUS | Business Intelligence OS",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Visual system ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
    .stApp { background: radial-gradient(ellipse at 78% -18%, rgba(40, 113, 133, .18), transparent 37%), #080D18; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0D1625 0%, #0A101B 100%); border-right: 1px solid rgba(155,185,210,.10); }
    [data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
    .nexus-brand { display:flex; align-items:center; gap:11px; margin: 4px 0 22px 0; }
    .nexus-mark { width:38px; height:38px; border-radius:13px; display:flex; align-items:center; justify-content:center; color:#07131C; font-size:21px; font-weight:900; background:linear-gradient(135deg,#87F0D4,#5EA7F5); box-shadow:0 8px 28px rgba(94,167,245,.22); }
    .nexus-word { color:#F1F6FC; font-size:18px; font-weight:800; letter-spacing:3px; line-height:1.0; }
    .nexus-sub { color:#7790A7; font-size:9px; font-family:'DM Mono',monospace; letter-spacing:1.7px; margin-top:5px; }
    .eyebrow { color:#77D9D0; font-family:'DM Mono',monospace; font-size:10px; letter-spacing:2.1px; text-transform:uppercase; margin-bottom:9px; }
    .hero-title { color:#F1F6FD; font-size:clamp(30px,4vw,48px); font-weight:800; letter-spacing:-2px; line-height:1.06; margin:0; }
    .hero-copy { color:#98AABD; font-size:14px; max-width:760px; line-height:1.7; margin-top:13px; }
    .hero-shell { padding:27px 29px 26px; border:1px solid rgba(138,181,209,.15); border-radius:22px; background:linear-gradient(115deg,rgba(20,42,59,.88),rgba(13,22,38,.91) 60%,rgba(24,38,57,.7)); position:relative; overflow:hidden; margin-bottom:18px; }
    .hero-shell:after { content:''; position:absolute; width:280px; height:280px; right:-80px; top:-160px; border-radius:50%; border:1px solid rgba(119,217,208,.15); box-shadow:0 0 0 30px rgba(119,217,208,.025),0 0 0 70px rgba(119,217,208,.018); }
    .hero-tag { display:inline-flex; align-items:center; gap:7px; padding:6px 10px; border-radius:100px; border:1px solid rgba(119,217,208,.22); background:rgba(119,217,208,.08); color:#9CE9DE; font:10px 'DM Mono',monospace; letter-spacing:1px; margin-bottom:16px; }
    .surface { background:linear-gradient(145deg,rgba(19,31,49,.94),rgba(13,21,34,.96)); border:1px solid rgba(147,173,204,.13); border-radius:17px; padding:18px 19px; }
    .surface-label { color:#8095AB; font:10px 'DM Mono',monospace; letter-spacing:1.2px; text-transform:uppercase; }
    .surface-value { color:#F2F7FC; font-size:27px; font-weight:800; letter-spacing:-1px; margin-top:8px; }
    .surface-note { color:#7F94AA; font-size:11px; margin-top:3px; line-height:1.45; }
    .section-kicker { color:#78D9D0; font:10px 'DM Mono',monospace; letter-spacing:1.8px; text-transform:uppercase; margin:14px 0 7px; }
    .section-title { color:#EAF2FB; font-size:21px; font-weight:750; letter-spacing:-.6px; margin:0 0 5px; }
    .section-copy { color:#8FA3B8; font-size:12px; line-height:1.6; margin-bottom:13px; }
    .flow-node { border:1px solid rgba(136,170,200,.16); border-radius:13px; padding:14px 15px; background:rgba(17,29,46,.85); min-height:96px; }
    .flow-num { color:#79D9D0; font:10px 'DM Mono',monospace; letter-spacing:1.3px; }
    .flow-head { color:#ECF4FC; font-size:14px; font-weight:800; margin:7px 0 5px; }
    .flow-desc { color:#8DA3B8; font-size:11px; line-height:1.5; }
    .dept-chip { display:inline-block; border:1px solid rgba(126,164,196,.2); background:rgba(126,164,196,.07); color:#B7C8D8; border-radius:7px; padding:4px 8px; font:10px 'DM Mono',monospace; letter-spacing:.4px; margin:3px 4px 3px 0; }
    .algorithm-card { border-radius:15px; padding:16px; background:linear-gradient(145deg,rgba(24,39,59,.9),rgba(14,24,39,.95)); border:1px solid rgba(132,173,199,.15); height:100%; }
    .algorithm-icon { font-size:19px; }
    .algorithm-title { color:#ECF4FC; font-weight:800; font-size:13px; margin:8px 0 4px; }
    .algorithm-body { color:#91A5B9; font-size:11px; line-height:1.55; }
    .status-pill { display:inline-block; padding:5px 9px; border-radius:99px; font:10px 'DM Mono',monospace; letter-spacing:.5px; background:rgba(111,215,192,.1); color:#8DE6D2; border:1px solid rgba(111,215,192,.2); }
    .muted { color:#879BB0; font-size:11px; }
    div[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(20,34,53,.92),rgba(13,22,36,.96)); border:1px solid rgba(145,177,205,.13); border-radius:14px; padding:14px 16px; }
    div[data-testid="stMetric"] label { color:#8CA1B7 !important; font-size:11px !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#F2F7FC !important; font-weight:800; }
    .stButton>button, .stDownloadButton>button { border-radius:10px; border:1px solid rgba(116,218,207,.28); background:linear-gradient(100deg,#8BE8D2,#6ABCE4); color:#081521; font-weight:800; padding:.55rem 1rem; }
    .stButton>button:hover, .stDownloadButton>button:hover { border:1px solid #B4F7E9; color:#06111C; filter:brightness(1.04); }
    div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input, .stTextArea textarea { background:#101C2D !important; border-color:rgba(142,173,202,.20) !important; }
    .stTabs [data-baseweb="tab-list"] { gap:7px; }
    .stTabs [data-baseweb="tab"] { background:rgba(25,42,62,.65); border-radius:9px 9px 0 0; padding:9px 14px; }
    .stTabs [aria-selected="true"] { background:rgba(110,210,200,.12) !important; }
    hr { border-color:rgba(140,170,200,.13); }
    footer {visibility:hidden;}
    #MainMenu {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Shared helpers ----------
@st.cache_data(show_spinner=False)
def demo_data():
    return make_demo_data()


def hero(eyebrow: str, title: str, copy: str, tag: str = "NEXUS INTELLIGENCE PLATFORM"):
    st.markdown(
        f"""<div class="hero-shell"><div class="hero-tag">✳ &nbsp;{tag}</div>
        <div class="eyebrow">{eyebrow}</div><h1 class="hero-title">{title}</h1>
        <div class="hero-copy">{copy}</div></div>""",
        unsafe_allow_html=True,
    )


def section(title: str, copy: str | None = None, kicker: str = ""):
    if kicker:
        st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if copy:
        st.markdown(f'<div class="section-copy">{copy}</div>', unsafe_allow_html=True)


def surface(label: str, value: str, note: str = ""):
    st.markdown(
        f'<div class="surface"><div class="surface-label">{label}</div><div class="surface-value">{value}</div><div class="surface-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def flow_card(number: str, title: str, description: str):
    st.markdown(
        f'<div class="flow-node"><div class="flow-num">{number}</div><div class="flow-head">{title}</div><div class="flow-desc">{description}</div></div>',
        unsafe_allow_html=True,
    )


def nice_chart(fig, height: int = 310):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        font=dict(family="Manrope, sans-serif", color="#AFC0D2", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="rgba(150,175,200,.09)", zerolinecolor="rgba(150,175,200,.1)")
    fig.update_yaxes(gridcolor="rgba(150,175,200,.09)", zerolinecolor="rgba(150,175,200,.1)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def safe_num(series: pd.Series, default: float = 0.0) -> float:
    value = pd.to_numeric(series, errors="coerce").dropna()
    return float(value.mean()) if len(value) else default


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def as_records(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    return value


# ---------- Sidebar / data loading ----------
with st.sidebar:
    st.markdown(
        '<div class="nexus-brand"><div class="nexus-mark">✳</div><div><div class="nexus-word">NEXUS</div><div class="nexus-sub">BUSINESS INTELLIGENCE OS</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eyebrow">WORKSPACE</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["Overview", "Data Studio", "Forecast Lab", "Algorithm Lab", "Decision Hub", "Governance & Export"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown('<div class="eyebrow">DATA CONNECTIONS</div>', unsafe_allow_html=True)
    sales_upload = st.file_uploader("Operational sales CSV", type=["csv"], key="sales_upload", help="One row per date/product/region or equivalent observation.")
    tx_upload = st.file_uploader("Transaction baskets CSV (optional)", type=["csv"], key="tx_upload", help="One row per product inside a transaction. Needs transaction_id and product columns.")
    demo_mode = sales_upload is None
    st.markdown("---")
    if demo_mode:
        st.markdown('<span class="status-pill">● DEMO DATA ACTIVE</span>', unsafe_allow_html=True)
        st.caption("Synthetic business data is loaded so you can explore every module immediately.")
    else:
        st.markdown('<span class="status-pill">● CUSTOM DATA ACTIVE</span>', unsafe_allow_html=True)
        st.caption("Your uploaded files are processed in this app session.")

if sales_upload is not None:
    try:
        raw_sales = pd.read_csv(sales_upload)
        source_name = sales_upload.name
    except Exception as exc:
        st.error(f"Could not read the sales CSV: {exc}")
        raw_sales = pd.DataFrame()
        source_name = "Upload error"
else:
    raw_sales, demo_transactions = demo_data()
    source_name = "NEXUS synthetic demo dataset"

raw_sales = clean_headers(raw_sales) if not raw_sales.empty else raw_sales

if "mapping_signature" not in st.session_state:
    st.session_state.mapping_signature = None
if "column_mapping" not in st.session_state:
    st.session_state.column_mapping = {}

file_signature = hashlib.md5((source_name + str(list(raw_sales.columns))).encode("utf-8")).hexdigest()
if st.session_state.mapping_signature != file_signature:
    st.session_state.column_mapping = {}
    st.session_state.mapping_signature = file_signature

# Apply saved user-selected mappings without removing original columns.
mapping = st.session_state.column_mapping
rename_map = {source: target for target, source in mapping.items() if source and source in raw_sales.columns and source != target}
working_sales = raw_sales.copy()
for target, source in mapping.items():
    if source and source in working_sales.columns and source != target:
        if target in working_sales.columns:
            working_sales = working_sales.drop(columns=[target])
        working_sales = working_sales.rename(columns={source: target})
sales_df = add_derived_fields(working_sales)

if tx_upload is not None:
    try:
        tx_df = clean_headers(pd.read_csv(tx_upload))
    except Exception as exc:
        st.error(f"Could not read the transaction CSV: {exc}")
        tx_df = pd.DataFrame()
else:
    if sales_upload is None:
        tx_df = demo_transactions.copy()
    else:
        tx_df = pd.DataFrame()

if "decision_log" not in st.session_state:
    st.session_state.decision_log = []
if "latest_forecast" not in st.session_state:
    st.session_state.latest_forecast = pd.DataFrame()
if "latest_algorithm_results" not in st.session_state:
    st.session_state.latest_algorithm_results = {}


# ---------- Pages ----------
if page == "Overview":
    total_units = int(sales_df["sales_units"].fillna(0).sum()) if "sales_units" in sales_df else 0
    product_count = sales_df["product"].nunique() if "product" in sales_df else 0
    latest_date = sales_df["date"].max().strftime("%d %b %Y") if "date" in sales_df and sales_df["date"].notna().any() else "Not mapped"
    gross_revenue = float(sales_df["revenue"].sum()) if "revenue" in sales_df else 0
    hero(
        "THE CONNECTED DECISION WORKSPACE",
        "One business. One decision engine.",
        "NEXUS connects Marketing demand signals, Supply Chain execution and Finance approval in one governed workflow—using machine learning as decision support, not as a substitute for accountable people.",
        "LIVE BUSINESS CONTROL ROOM  /  V1.0",
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Sales units in dataset", f"{total_units:,.0f}", help="Total units across all uploaded observations; not a forecast.")
    with c2: st.metric("Products tracked", f"{product_count:,}")
    with c3: st.metric("Gross revenue represented", f"{gross_revenue:,.0f}" if gross_revenue else "—")
    with c4: st.metric("Latest data date", latest_date)

    section("The decision loop", "Every hand-off is evidence-backed, visible and traceable.", "HOW NEXUS WORKS")
    cols = st.columns(4)
    with cols[0]: flow_card("01 / MARKETING", "Demand signal", "Submit sales expectations, campaign context and the assumptions behind the forecast.")
    with cols[1]: flow_card("02 / INTELLIGENCE", "Challenge & explain", "Regression and ARIMA test the demand story; classification, clustering and rules add context.")
    with cols[2]: flow_card("03 / SUPPLY CHAIN", "Plan & propose", "Translate validated demand into inventory coverage and an evidence-based procurement request.")
    with cols[3]: flow_card("04 / FINANCE", "Check & authorise", "Evaluate budget, unit economics, margin and downside exposure before an accountable approval.")

    left, right = st.columns([1.5, 1])
    with left:
        section("Demand pulse", "Weekly sales volume across the selected data source.", "OPERATING SIGNAL")
        if {"date", "sales_units"}.issubset(sales_df.columns):
            chart = sales_df.dropna(subset=["date"]).groupby(pd.Grouper(key="date", freq="W"))["sales_units"].sum().reset_index()
            fig = px.area(chart, x="date", y="sales_units", color_discrete_sequence=["#70D9CE"])
            fig.update_traces(line=dict(width=2), fillcolor="rgba(112,217,206,.12)")
            fig.update_layout(yaxis_title="Units", xaxis_title="")
            nice_chart(fig, 300)
        else:
            st.info("Map a date and sales-units column in Data Studio to activate the demand pulse.")
    with right:
        section("Algorithm stack", "Five methods. Five specific jobs.", "INTELLIGENCE LAYER")
        for icon, name, detail in [
            ("↗", "Regression", "Quantify demand drivers"),
            ("◷", "ARIMA", "Forecast time-series demand"),
            ("◇", "Classification", "Score operational risk"),
            ("◌", "Clustering", "Discover similar segments"),
            ("⇄", "Association rules", "Reveal product affinities"),
        ]:
            st.markdown(f'<div style="display:flex;align-items:center;gap:11px;padding:8px 0;border-bottom:1px solid rgba(145,175,200,.09)"><span style="color:#7EDFD4;font-size:18px;width:22px">{icon}</span><div style="flex:1"><div style="color:#E6F0F9;font-size:12px;font-weight:800">{name}</div><div class="muted">{detail}</div></div><span style="color:#6F879D;font:9px DM Mono,monospace">READY</span></div>', unsafe_allow_html=True)

    section("Department alignment", "DBM coordinates the process; each department keeps ownership of its business decisions.", "SHARED OPERATING MODEL")
    a, b, c, d = st.columns(4)
    with a: surface("MARKETING", "Demand", "Owns forecast assumptions and campaign context")
    with b: surface("SUPPLY CHAIN", "Feasibility", "Owns stock, lead-time and quantity recommendation")
    with c: surface("FINANCE", "Viability", "Owns budget and financial authorisation")
    with d: surface("DBM", "Coordination", "Owns workflow, data standards and auditability")

elif page == "Data Studio":
    hero("DATA FOUNDATION", "Bring your business data into focus.", "Load your own operational CSVs, map unfamiliar column names, inspect completeness and understand what each algorithm can—and cannot—do with the available data.", "DATA STUDIO  /  INPUTS & QUALITY")
    st.caption(f"Current source: **{source_name}** · {len(sales_df):,} rows · {len(sales_df.columns):,} columns")

    with st.expander("Column mapping — connect your CSV to NEXUS", expanded=not {"date", "sales_units", "product"}.issubset(sales_df.columns)):
        st.write("Choose which source columns represent the NEXUS business fields. Existing standard column names are detected automatically.")
        source_cols = ["— Not mapped —"] + raw_sales.columns.tolist()
        canonical_fields = [
            ("date", "Date / time", True), ("product", "Product / SKU", True),
            ("sales_units", "Sales units / demand", True), ("marketing_spend", "Marketing spend", False),
            ("price", "Selling price", False), ("inventory_on_hand", "Usable stock on hand", False),
            ("incoming_units", "Confirmed incoming stock", False), ("lead_time_days", "Supplier lead time (days)", False),
            ("unit_cost", "Unit procurement cost", False), ("category", "Product category", False),
            ("region", "Region / location", False), ("discount_pct", "Discount percentage", False),
            ("stockout_risk", "Existing risk label (optional)", False),
        ]
        defaults = {}
        for canonical, label, required in canonical_fields:
            current = mapping.get(canonical, canonical if canonical in raw_sales.columns else "— Not mapped —")
            defaults[canonical] = current if current in source_cols else "— Not mapped —"
        with st.form("column_mapping_form"):
            map_cols = st.columns(3)
            selected_map = {}
            for idx, (canonical, label, required) in enumerate(canonical_fields):
                with map_cols[idx % 3]:
                    selected_map[canonical] = st.selectbox(
                        f"{label}{' *' if required else ''}", source_cols,
                        index=source_cols.index(defaults[canonical]), key=f"map_{canonical}",
                    )
            submitted = st.form_submit_button("Apply column mapping")
            if submitted:
                chosen = [v for v in selected_map.values() if v != "— Not mapped —"]
                duplicates = sorted({v for v in chosen if chosen.count(v) > 1})
                missing_required = [label for canonical, label, required in canonical_fields if required and selected_map[canonical] == "— Not mapped —"]
                if duplicates:
                    st.error("A source column cannot be mapped to multiple fields: " + ", ".join(duplicates))
                elif missing_required:
                    st.error("Map the required fields first: " + ", ".join(missing_required))
                else:
                    st.session_state.column_mapping = {k: v for k, v in selected_map.items() if v != "— Not mapped —"}
                    st.session_state.mapping_signature = file_signature
                    st.success("Column mapping saved. NEXUS is refreshing the data model.")
                    st.rerun()

    top_a, top_b, top_c = st.columns(3)
    top_a.metric("Rows loaded", f"{len(sales_df):,}")
    top_b.metric("Missing cells", f"{int(sales_df.isna().sum().sum()):,}")
    top_c.metric("Duplicate rows", f"{int(sales_df.duplicated().sum()):,}")

    left, right = st.columns([1.15, .85])
    with left:
        section("Operational data preview", "The first rows after header normalisation and derived-field checks.", "DATASET")
        st.dataframe(sales_df.head(15), use_container_width=True, hide_index=True)
    with right:
        section("Data quality scan", "Missingness by field; resolve critical gaps before training.", "READINESS")
        missing = sales_df.isna().mean().mul(100).sort_values(ascending=False).head(12).reset_index()
        missing.columns = ["field", "missing_pct"]
        if len(missing):
            fig = px.bar(missing, x="missing_pct", y="field", orientation="h", color="missing_pct", color_continuous_scale=["#3E8C91", "#E8AA6A"])
            fig.update_layout(xaxis_title="Missing values (%)", yaxis_title="", coloraxis_showscale=False)
            nice_chart(fig, 300)

    section("Field coverage", "A minimum data contract for the NEXUS algorithms.", "READINESS CHECK")
    checks = [
        ("date", "Time-series forecasting", "ARIMA needs dated, consistently spaced observations."),
        ("sales_units", "Regression + ARIMA", "Numeric target representing units sold or demand."),
        ("marketing_spend", "Demand-driver regression", "Campaign expenditure or a comparable marketing input."),
        ("inventory_on_hand", "Procurement + stock risk", "Usable inventory at the relevant product/location grain."),
        ("unit_cost", "Finance validation", "Purchase cost for financial justification."),
        ("transaction_id", "Association rules", "A transaction/basket identifier in the separate basket file."),
    ]
    for field, purpose, description in checks:
        available = field in sales_df.columns or (field == "transaction_id" and "transaction_id" in tx_df.columns)
        status = "AVAILABLE" if available else "NEEDS DATA"
        st.markdown(f'<div style="display:flex;gap:12px;align-items:center;padding:8px 1px;border-bottom:1px solid rgba(145,175,200,.08)"><span class="status-pill" style="min-width:92px;text-align:center">{status}</span><div style="flex:1"><div style="color:#E5EFF8;font-size:12px;font-weight:700">{purpose}</div><div class="muted">{description}</div></div><code style="color:#7CCFC7">{field}</code></div>', unsafe_allow_html=True)

    st.markdown("#### Downloadable data templates")
    demo_sales, demo_tx = demo_data()
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("↓ Sales data template (CSV)", csv_bytes(demo_sales.head(250)), "nexus_sales_template.csv", "text/csv", use_container_width=True)
    with dl2:
        st.download_button("↓ Transaction basket template (CSV)", csv_bytes(demo_tx.head(500)), "nexus_transaction_template.csv", "text/csv", use_container_width=True)
    st.info("For the most reliable results, use one row per date × product × region for the operational dataset. Association Rule Mining needs a separate transaction-level file with one row per product inside each transaction.")

elif page == "Forecast Lab":
    hero("DEMAND INTELLIGENCE", "Challenge the forecast, not just the number.", "Compare Marketing’s weekly sales allocation with an independent ARIMA baseline. Make forecast assumptions, uncertainty and model diagnostics visible before Supply Chain commits inventory.", "FORECAST LAB  /  ARIMA + RECONCILIATION")
    if not {"date", "sales_units"}.issubset(sales_df.columns):
        st.warning("Forecasting needs mapped date and sales-units columns. Open Data Studio and map them first.")
    else:
        dated = sales_df.dropna(subset=["date", "sales_units"]).copy()
        product_options = ["All products"] + (sorted(dated["product"].dropna().astype(str).unique().tolist()) if "product" in dated else [])
        with st.container(border=True):
            controls = st.columns([1.2, 1, 1, 1, 1])
            with controls[0]: selected_product = st.selectbox("Forecast scope", product_options)
            with controls[1]: horizon = st.number_input("Forecast horizon (weeks)", 2, 16, 6)
            with controls[2]: p = st.number_input("AR order (p)", 0, 4, 2)
            with controls[3]: d = st.number_input("Differencing (d)", 0, 2, 1)
            with controls[4]: q = st.number_input("MA order (q)", 0, 4, 1)
        selected = dated if selected_product == "All products" or "product" not in dated else dated[dated["product"].astype(str) == selected_product]
        weekly = selected.groupby(pd.Grouper(key="date", freq="W-SUN"))["sales_units"].sum().sort_index()
        weekly = weekly[weekly.index.notna()]
        weekly = weekly.asfreq("W-SUN").fillna(0)

        st.markdown("#### Marketing submission · lecture scenario")
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1: week1_share = st.number_input("Week 1 share (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
        with mc2: week2_share = st.number_input("Week 2 share (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
        with mc3: reported_r2 = st.number_input("Marketing-reported R²", min_value=-1.0, max_value=1.0, value=0.70, step=0.05, format="%.2f")
        with mc4: reported_variation = st.number_input("Reported variation (units)", min_value=0, value=2000, step=100)
        if week1_share + week2_share > 100:
            st.error("Week 1 and Week 2 shares cannot exceed 100% together. Recheck the allocation.")
        elif len(weekly) < 18:
            st.warning(f"Only {len(weekly)} weekly observations are available. ARIMA needs at least 18 for this workspace; load a longer history.")
        else:
            try:
                with st.spinner("Fitting the time-series model…"):
                    result = run_arima(weekly, int(horizon), (int(p), int(d), int(q)))
                forecast = result["forecast"].copy()
                forecast["week_ending"] = [weekly.index.max() + pd.Timedelta(days=7 * (i + 1)) for i in range(len(forecast))]
                forecast["week_number"] = np.arange(1, len(forecast) + 1)
                st.session_state.latest_forecast = forecast
                total_baseline = float(forecast["forecast"].sum())
                marketing_w1 = total_baseline * week1_share / 100
                marketing_w2 = total_baseline * week2_share / 100
                baseline_w1_share = float(forecast.iloc[0]["forecast"] / total_baseline * 100) if total_baseline else 0
                baseline_w2_share = float(forecast.iloc[1]["forecast"] / total_baseline * 100) if total_baseline else 0
                variance1 = week1_share - baseline_w1_share
                variance2 = week2_share - baseline_w2_share

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("ARIMA baseline · horizon", f"{total_baseline:,.0f} units")
                m2.metric("ARIMA Week 1 share", f"{baseline_w1_share:.1f}%", f"Marketing {week1_share:.1f}% · Δ {variance1:+.1f} pp")
                m3.metric("ARIMA Week 2 share", f"{baseline_w2_share:.1f}%", f"Marketing {week2_share:.1f}% · Δ {variance2:+.1f} pp")
                m4.metric("Model AIC", f"{result['fitted_aic']:,.1f}", "Compare only across comparable model settings")
                if result.get("backtest"):
                    bt = result["backtest"]
                    st.markdown("#### Chronological holdout check")
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Backtest MAE", f"{bt['mae']:,.1f} units")
                    b2.metric("Backtest RMSE", f"{bt['rmse']:,.1f} units")
                    b3.metric("Backtest WAPE", f"{bt['wape_pct']:.1f}%", f"Naive baseline {bt['naive_wape_pct']:.1f}%")
                    st.caption(f"Holdout: last {bt['test_periods']} weekly observations. Use the holdout and baseline comparison to judge usefulness; AIC alone is not a business accuracy score.")

                section("Forecast vs historical demand", "The shaded range is the model-generated 95% interval, not the lecture’s ±2,000 figure.", "MODEL OUTPUT")
                hist_df = weekly.tail(30).rename("actual").reset_index()
                hist_df.columns = ["week_ending", "actual"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist_df["week_ending"], y=hist_df["actual"], name="Historical demand", line=dict(color="#78A9E8", width=2)))
                fig.add_trace(go.Scatter(x=forecast["week_ending"], y=forecast["upper_95"], name="95% upper", line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=forecast["week_ending"], y=forecast["lower_95"], name="95% model interval", fill="tonexty", fillcolor="rgba(119,217,208,.12)", line=dict(width=0)))
                fig.add_trace(go.Scatter(x=forecast["week_ending"], y=forecast["forecast"], name="ARIMA forecast", line=dict(color="#75E0D0", width=3, dash="dash"), mode="lines+markers"))
                nice_chart(fig, 350)

                section("Marketing allocation challenge", "Percentages are treated as shares of the total forecast horizon. Confirm this interpretation with Marketing before using it for planning.", "RECONCILIATION")
                comp = pd.DataFrame({
                    "Period": ["Week 1", "Week 2", "Remaining horizon"],
                    "Marketing allocation (%)": [week1_share, week2_share, 100 - week1_share - week2_share],
                    "ARIMA baseline share (%)": [baseline_w1_share, baseline_w2_share, max(0, 100 - baseline_w1_share - baseline_w2_share)],
                })
                fig2 = px.bar(comp, x="Period", y=["Marketing allocation (%)", "ARIMA baseline share (%)"], barmode="group", color_discrete_sequence=["#6EAEEB", "#73DDCF"])
                fig2.update_layout(yaxis_title="Share of horizon forecast (%)", xaxis_title="")
                nice_chart(fig2, 260)
                st.dataframe(forecast[["week_number", "week_ending", "forecast", "lower_95", "upper_95"]].round(1), use_container_width=True, hide_index=True)

                st.markdown("#### Forecast governance checks")
                if abs(variance1) >= 10 or abs(variance2) >= 10:
                    st.warning("Material allocation variance: Marketing and the statistical baseline differ by at least 10 percentage points in Week 1 or Week 2. Route for departmental review; do not auto-reject the forecast.")
                else:
                    st.success("No 10-percentage-point allocation variance detected in the first two weeks. This is a review signal, not approval.")
                st.info(f"**R² interpretation:** Marketing’s reported R² of {reported_r2:.2f} means the model explains {reported_r2*100:.0f}% of observed target variation in its evaluation context. It does not mean {reported_r2*100:.0f}% forecast accuracy. Require MAE, RMSE, WAPE, bias and time-based holdout results. **±{reported_variation:,}** is treated only as a reported variation until its statistical definition and coverage level are confirmed.")
                st.download_button("↓ Download ARIMA forecast CSV", csv_bytes(forecast), "nexus_arima_forecast.csv", "text/csv")
            except Exception as exc:
                st.error(f"ARIMA could not be fitted with these settings: {exc}")

elif page == "Algorithm Lab":
    hero("MACHINE LEARNING WORKBENCH", "Five algorithms. Real business jobs.", "Run the models on the uploaded dataset, inspect their outputs and connect the result to a decision. NEXUS keeps model evidence separate from business approval.", "ALGORITHM LAB  /  ANALYTICS ENGINE")
    tabs = st.tabs(["01 · Regression", "02 · Classification", "03 · Clustering", "04 · Association rules"])

    with tabs[0]:
        section("Regression · demand drivers", "Estimate a continuous outcome such as sales units or revenue from business drivers.", "MARKETING → SUPPLY CHAIN")
        numeric_targets = [c for c in sales_df.select_dtypes(include=np.number).columns if sales_df[c].nunique(dropna=True) > 2]
        if not numeric_targets:
            st.warning("No numeric target columns found. Map sales_units or another numeric measure in Data Studio.")
        else:
            default_target = "sales_units" if "sales_units" in numeric_targets else numeric_targets[0]
            rc1, rc2 = st.columns([1, 1.4])
            with rc1: reg_target = st.selectbox("Predict a numeric outcome", numeric_targets, index=numeric_targets.index(default_target), key="reg_target")
            with rc2: reg_model = st.selectbox("Regression method", ["Random Forest", "Linear Regression", "Ridge Regression"], key="reg_model")
            candidates = [c for c in sales_df.columns if c not in [reg_target, "date", "transaction_id"] and sales_df[c].nunique(dropna=True) > 1]
            sensible = [c for c in ["marketing_spend", "price", "discount_pct", "product", "category", "region", "campaign_active"] if c in candidates]
            reg_features = st.multiselect("Input features", candidates, default=sensible or candidates[:5], key="reg_features")
            if st.button("Run regression model", key="run_regression"):
                if not reg_features:
                    st.error("Choose at least one input feature.")
                else:
                    try:
                        output = run_regression(sales_df, reg_target, reg_features, reg_model)
                        st.session_state.latest_algorithm_results["regression"] = {k: v for k, v in output.items() if not isinstance(v, (pd.Series, pd.DataFrame))}
                        st.session_state.latest_algorithm_results["regression_features"] = output["features"]
                        st.session_state.latest_algorithm_results["regression_scatter"] = pd.DataFrame({"Actual": output["actual"], "Predicted": output["predicted"]})
                        st.success(f"Model trained on {output['rows']:,} usable rows · {output['split']} evaluation")
                        k1, k2, k3 = st.columns(3)
                        k1.metric("R²", f"{output['r2']:.3f}", help="Explained variation on this test set. Not forecast accuracy.")
                        k2.metric("MAE", f"{output['mae']:,.2f}", help="Average absolute error in the target's units.")
                        k3.metric("RMSE", f"{output['rmse']:,.2f}", help="Penalises larger prediction errors more strongly.")
                    except Exception as exc:
                        st.error(str(exc))
            result = st.session_state.latest_algorithm_results
            if "regression_scatter" in result:
                chart_data = result["regression_scatter"]
                fig = px.scatter(chart_data, x="Actual", y="Predicted", opacity=.65, color_discrete_sequence=["#73DCCF"])
                low = float(min(chart_data.min()))
                high = float(max(chart_data.max()))
                fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Perfect fit", line=dict(color="#E5AE72", dash="dash")))
                nice_chart(fig, 300)
                feat = result.get("regression_features", pd.DataFrame())
                if isinstance(feat, pd.DataFrame) and not feat.empty:
                    section("What influenced the model", "Feature importance or absolute coefficient magnitude; these are predictive signals, not proof of causality.")
                    figf = px.bar(feat.head(10).sort_values("importance"), x="importance", y="feature", orientation="h", color_discrete_sequence=["#70B7E8"])
                    nice_chart(figf, 260)

    with tabs[1]:
        section("Classification · risk categories", "Turn business observations into categories such as low, medium and high stockout risk.", "SUPPLY CHAIN + FINANCE")
        cat_targets = [c for c in sales_df.columns if 2 <= sales_df[c].nunique(dropna=True) <= 20 and (not pd.api.types.is_numeric_dtype(sales_df[c]) or c.endswith("_class"))]
        if "stockout_risk" in sales_df.columns and "stockout_risk" not in cat_targets:
            cat_targets.insert(0, "stockout_risk")
        if not cat_targets:
            st.warning("No suitable categorical target found. Map an existing label or provide a risk/category field.")
        else:
            default_cls = "stockout_risk" if "stockout_risk" in cat_targets else cat_targets[0]
            c1, c2 = st.columns(2)
            with c1: cls_target = st.selectbox("Class to predict", cat_targets, index=cat_targets.index(default_cls), key="cls_target")
            with c2: cls_model = st.selectbox("Classification method", ["Random Forest", "Logistic Regression"], key="cls_model")
            if cls_target == "stockout_risk":
                st.info("Demo-data note: the sample stockout-risk label is derived from current stock cover. For a genuinely predictive production model, train against a future stockout outcome and exclude any fields that directly reveal that outcome.")
            cls_candidates = [c for c in sales_df.columns if c != cls_target and c not in ["date", "transaction_id"] and sales_df[c].nunique(dropna=True) > 1]
            cls_default = [c for c in ["sales_units", "inventory_on_hand", "incoming_units", "lead_time_days", "marketing_spend", "price"] if c in cls_candidates]
            cls_features = st.multiselect("Predictor fields", cls_candidates, default=cls_default[:6] or cls_candidates[:5], key="cls_features")
            if st.button("Run classification model", key="run_classification"):
                try:
                    if not cls_features:
                        raise ValueError("Choose at least one predictor field.")
                    output = run_classification(sales_df, cls_target, cls_features, cls_model)
                    st.session_state.latest_algorithm_results["classification"] = {k: v for k, v in output.items() if not isinstance(v, (pd.DataFrame, pd.Series))}
                    st.session_state.latest_algorithm_results["classification_confusion"] = output["confusion"]
                    st.session_state.latest_algorithm_results["classification_report"] = output["report"]
                    st.session_state.latest_algorithm_results["classification_features"] = output["features"]
                    st.success(f"Classified {output['rows']:,} usable observations")
                    st.metric("Test accuracy", f"{output['accuracy']:.1%}", help="Share of test observations classified correctly. Review precision/recall by class as well.")
                except Exception as exc:
                    st.error(str(exc))
            result = st.session_state.latest_algorithm_results
            if "classification_confusion" in result:
                cm = result["classification_confusion"]
                fig = px.imshow(cm, text_auto=True, aspect="auto", color_continuous_scale=[[0, "#14253A"], [1, "#62CFC5"]], labels=dict(x="Predicted", y="Actual", color="Cases"))
                fig.update_layout(title="Confusion matrix")
                nice_chart(fig, 310)
                st.markdown("**Class-level precision / recall / F1**")
                st.dataframe(result["classification_report"].round(3), use_container_width=True)
                feat = result.get("classification_features", pd.DataFrame())
                if isinstance(feat, pd.DataFrame) and not feat.empty:
                    figf = px.bar(feat.head(10).sort_values("importance"), x="importance", y="feature", orientation="h", color_discrete_sequence=["#79B6E8"])
                    nice_chart(figf, 240)

    with tabs[2]:
        section("Clustering · discover natural segments", "Group products or observations by similarity without pre-labelled categories.", "MARKETING + SUPPLY CHAIN")
        numeric_features = [c for c in sales_df.select_dtypes(include=np.number).columns if sales_df[c].nunique(dropna=True) > 2]
        if len(numeric_features) < 2:
            st.warning("Clustering needs at least two numeric features.")
        else:
            default_cluster = [c for c in ["sales_units", "marketing_spend", "price", "inventory_on_hand", "lead_time_days"] if c in numeric_features]
            cl1, cl2 = st.columns([2, 1])
            with cl1: cluster_features = st.multiselect("Similarity variables", numeric_features, default=default_cluster[:4] or numeric_features[:3], key="cluster_features")
            with cl2: k_clusters = st.slider("Number of clusters", 2, 8, 4, key="cluster_count")
            if st.button("Discover segments", key="run_clustering"):
                try:
                    output = run_clustering(sales_df, cluster_features, k_clusters)
                    st.session_state.latest_algorithm_results["clustering_plot"] = output["plot"]
                    st.session_state.latest_algorithm_results["clustering_profile"] = output["profile"]
                    st.session_state.latest_algorithm_results["clustering_meta"] = {"inertia": output["inertia"], "rows": output["rows"], "features": output["features"]}
                    st.success(f"Found {k_clusters} segments across {output['rows']:,} observations")
                except Exception as exc:
                    st.error(str(exc))
            result = st.session_state.latest_algorithm_results
            if "clustering_plot" in result:
                plot = result["clustering_plot"]
                fig = px.scatter(plot, x="component_1", y="component_2", color="cluster", opacity=.72, color_discrete_sequence=["#6FD8CB", "#72AEEB", "#E3B174", "#B99BE8", "#E58CA4", "#91C78B"])
                fig.update_layout(title="PCA projection of clustered observations", xaxis_title="Principal component 1", yaxis_title="Principal component 2")
                nice_chart(fig, 340)
                st.markdown("**Segment profile (average original values)**")
                st.dataframe(result["clustering_profile"].round(2), use_container_width=True, hide_index=True)
                st.caption("Clusters are mathematical groupings, not business labels. Review their profiles with department owners before turning them into a policy.")

    with tabs[3]:
        section("Association Rule Mining · product affinity", "Find products that commonly appear in the same transaction. Use the evidence for bundles, cross-sell and joint inventory conversations.", "MARKETING + SUPPLY CHAIN + FINANCE")
        if tx_df.empty or not {"transaction_id", "product"}.issubset(tx_df.columns):
            st.warning("Upload a transaction basket CSV with two columns: transaction_id and product. Each row should represent one product in one basket.")
            st.code("transaction_id,product\nTX-001,SKU-101\nTX-001,SKU-422\nTX-002,SKU-205", language="csv")
        else:
            tx_count = tx_df["transaction_id"].nunique()
            x1, x2, x3 = st.columns(3)
            with x1: min_support = st.slider("Minimum support", .005, .25, .025, .005, format="%.3f")
            with x2: min_confidence = st.slider("Minimum confidence", .05, 1.0, .25, .05)
            with x3: min_lift = st.slider("Minimum lift", .5, 4.0, 1.0, .1)
            st.caption(f"{tx_count:,} unique transaction IDs · {len(tx_df):,} transaction-product rows")
            if st.button("Mine product associations", key="run_association"):
                try:
                    rules = run_association_rules(tx_df, "transaction_id", "product", min_support, min_confidence, min_lift)
                    st.session_state.latest_algorithm_results["association_rules"] = rules
                    st.session_state.latest_algorithm_results["association_meta"] = {"transactions": tx_count, "support": min_support, "confidence": min_confidence, "lift": min_lift}
                    if rules.empty:
                        st.info("No rules met the selected thresholds. Lower support, confidence or lift and run again.")
                    else:
                        st.success(f"Discovered {len(rules):,} directional association rules")
                except Exception as exc:
                    st.error(str(exc))
            rules = st.session_state.latest_algorithm_results.get("association_rules")
            if isinstance(rules, pd.DataFrame):
                if not rules.empty:
                    display_rules = rules.copy()
                    for col in ["support", "confidence"]: display_rules[col] = display_rules[col].map(lambda x: f"{x:.1%}")
                    display_rules["lift"] = display_rules["lift"].map(lambda x: f"{x:.2f}×")
                    st.dataframe(display_rules, use_container_width=True, hide_index=True)
                    st.download_button("↓ Download association rules", csv_bytes(rules), "nexus_association_rules.csv", "text/csv")
                else:
                    st.caption("No rules currently meet the thresholds.")
        st.info("Interpretation: support = share of baskets containing both items; confidence = share of baskets containing A that also contain B; lift = strength versus independent co-occurrence. A rule is not proof that buying A causes B to be purchased.")

elif page == "Decision Hub":
    hero("CROSS-FUNCTIONAL EXECUTION", "Turn a forecast into an accountable decision.", "A single decision record links Marketing’s demand signal, Supply Chain’s stock-aware procurement proposal, Finance’s budget and margin review, and DBM’s workflow coordination.", "DECISION HUB  /  MARKETING → SUPPLY CHAIN → FINANCE")

    has_core = {"product", "sales_units"}.issubset(sales_df.columns)
    if not has_core:
        st.warning("Decision Hub needs product and sales_units fields. Map them in Data Studio.")
    else:
        products = sorted(sales_df["product"].dropna().astype(str).unique().tolist())
        with st.form("procurement_form"):
            section("01 · Marketing demand submission", "Enter the demand signal that Supply Chain is being asked to plan against.")
            a, b, c = st.columns(3)
            with a:
                product_choice = st.selectbox("Product / SKU", products)
                planning_weeks = st.number_input("Planning horizon (weeks)", min_value=1, max_value=26, value=2)
            selected_product_rows = sales_df[sales_df["product"].astype(str) == product_choice]
            recent_demand = selected_product_rows
            if "date" in recent_demand and recent_demand["date"].notna().any():
                weekly_demand = recent_demand.dropna(subset=["date"]).groupby(pd.Grouper(key="date", freq="W-SUN"))["sales_units"].sum().tail(8)
                avg_weekly = float(weekly_demand.mean()) if len(weekly_demand) else safe_num(recent_demand["sales_units"]) * 7
            else:
                avg_weekly = safe_num(recent_demand["sales_units"]) * 7
            default_demand = int(max(0, round(avg_weekly * planning_weeks)))
            with b:
                forecast_units = st.number_input("Marketing forecast units", min_value=0, value=default_demand, step=10)
                forecast_confidence = st.selectbox("Forecast evidence status", ["Submitted — pending challenge", "Supported by historical data", "Material variance — explanation required"])
            with c:
                marketing_note = st.text_input("Marketing assumption / campaign note", placeholder="e.g. launch campaign + 15% promotion")
                forecast_owner = st.text_input("Forecast owner", value="Marketing")

            st.markdown("---")
            section("02 · Supply Chain procurement proposal", "Recommended quantity uses forecast demand, a planning buffer and usable supply—not just forecast sales.")
            latest_snapshot = selected_product_rows.copy()
            if "date" in latest_snapshot and latest_snapshot["date"].notna().any():
                latest_snapshot = latest_snapshot.dropna(subset=["date"]).sort_values("date").groupby("region").tail(1) if "region" in latest_snapshot else latest_snapshot.sort_values("date").tail(1)
            stock_default = int(latest_snapshot["inventory_on_hand"].fillna(0).sum()) if "inventory_on_hand" in latest_snapshot else 0
            incoming_default = int(latest_snapshot["incoming_units"].fillna(0).sum()) if "incoming_units" in latest_snapshot else 0
            cost_default = safe_num(selected_product_rows["unit_cost"], 0) if "unit_cost" in selected_product_rows else 0
            price_default = safe_num(selected_product_rows["price"], 0) if "price" in selected_product_rows else 0
            s1, s2, s3, s4 = st.columns(4)
            with s1: stock_on_hand = st.number_input("Usable stock on hand", min_value=0, value=max(0, stock_default), step=10)
            with s2: incoming_units = st.number_input("Confirmed incoming units", min_value=0, value=max(0, incoming_default), step=10)
            with s3: safety_buffer = st.number_input("Planning buffer (%)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
            with s4: supplier_lead = st.number_input("Supplier lead time (days)", min_value=0, max_value=365, value=int(safe_num(selected_product_rows["lead_time_days"], 14)) if "lead_time_days" in selected_product_rows else 14)

            st.markdown("---")
            section("03 · Finance review inputs", "Finance sees the unit economics and spend exposure before recording its authorised decision.")
            f1, f2, f3 = st.columns(3)
            with f1: unit_cost = st.number_input("Procurement unit cost", min_value=0.0, value=max(0.0, cost_default), step=1.0)
            with f2: selling_price = st.number_input("Expected selling price", min_value=0.0, value=max(0.0, price_default), step=1.0)
            with f3: available_budget = st.number_input("Available procurement budget", min_value=0.0, value=25000.0, step=1000.0)
            calculate = st.form_submit_button("Calculate procurement case", use_container_width=True)

        if calculate:
            planning_days = int(planning_weeks) * 7
            lead_time_top_up = max(0.0, (supplier_lead - planning_days) * (avg_weekly / 7))
            coverage_demand = forecast_units + lead_time_top_up
            buffered_demand = coverage_demand * (1 + safety_buffer / 100)
            suggested_qty = max(0, int(math.ceil(buffered_demand - stock_on_hand - incoming_units)))
            purchase_value = suggested_qty * unit_cost
            expected_revenue = forecast_units * selling_price
            expected_gross_profit = forecast_units * (selling_price - unit_cost)
            gross_margin = ((selling_price - unit_cost) / selling_price * 100) if selling_price > 0 else 0
            budget_gap = max(0, purchase_value - available_budget)
            recommendation = {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "product": product_choice,
                "planning_weeks": int(planning_weeks), "forecast_units": int(forecast_units),
                "forecast_evidence_status": forecast_confidence, "marketing_note": marketing_note,
                "forecast_owner": forecast_owner, "stock_on_hand": int(stock_on_hand),
                "incoming_units": int(incoming_units), "buffer_pct": float(safety_buffer),
                "lead_time_top_up_units": float(lead_time_top_up), "coverage_demand_units": float(coverage_demand),
                "supplier_lead_days": int(supplier_lead), "suggested_purchase_qty": suggested_qty,
                "unit_cost": float(unit_cost), "purchase_value": purchase_value,
                "selling_price": float(selling_price), "expected_revenue": expected_revenue,
                "expected_gross_profit": expected_gross_profit, "gross_margin_pct": gross_margin,
                "budget": float(available_budget), "budget_gap": budget_gap,
            }
            st.session_state.latest_procurement = recommendation

        recommendation = st.session_state.get("latest_procurement")
        if recommendation:
            st.markdown("---")
            section("Procurement decision case", "A calculated proposal—not an automatic purchase order or a financial approval.", "NEXUS RECOMMENDATION")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Suggested procurement", f"{recommendation['suggested_purchase_qty']:,} units")
            r2.metric("Purchase value", f"{recommendation['purchase_value']:,.2f}")
            r3.metric("Expected gross margin", f"{recommendation['gross_margin_pct']:.1f}%")
            r4.metric("Budget headroom", f"{recommendation['budget'] - recommendation['purchase_value']:,.2f}", delta=f"Gap {recommendation['budget_gap']:,.2f}" if recommendation['budget_gap'] else "Within submitted budget")
            check1, check2 = st.columns(2)
            with check1:
                st.markdown("**Supply Chain calculation**")
                st.code(
                    f"Forecast demand                 {recommendation['forecast_units']:>10,}\n"
                    f"Lead-time top-up                {recommendation['lead_time_top_up_units']:>10,.0f}\n"
                    f"Planning buffer ({recommendation['buffer_pct']:.0f}%)      {recommendation['coverage_demand_units'] * recommendation['buffer_pct'] / 100:>10,.0f}\n"
                    f"Less usable stock                {recommendation['stock_on_hand']:>10,}\n"
                    f"Less confirmed incoming          {recommendation['incoming_units']:>10,}\n"
                    f"Suggested purchase               {recommendation['suggested_purchase_qty']:>10,}", language="text")
            with check2:
                st.markdown("**Finance exposure**")
                st.write(f"Expected revenue at forecast demand: **{recommendation['expected_revenue']:,.2f}**")
                st.write(f"Expected gross profit before other costs: **{recommendation['expected_gross_profit']:,.2f}**")
                st.write(f"Procurement spend: **{recommendation['purchase_value']:,.2f}**")
                st.caption(f"Lead-time coverage adds {recommendation['lead_time_top_up_units']:,.0f} units beyond the selected planning horizon where supplier lead time is longer. Confirm incoming stock is due within the coverage window.")
                if recommendation["purchase_value"] > recommendation["budget"]:
                    st.error("Budget exception: request exceeds the submitted budget. Finance review or a revised quantity is required.")
                elif recommendation["gross_margin_pct"] <= 0:
                    st.warning("Unit economics are non-positive at the submitted price and cost. Recheck pricing/cost before approval.")
                else:
                    st.success("Within submitted budget and positive gross margin. This is eligibility for review—not automatic approval.")

            st.markdown("---")
            section("04 · DBM workflow and human approval", "Record the current departmental position. Each owner remains accountable for their own decision.")
            with st.form("governance_form"):
                g1, g2, g3, g4 = st.columns(4)
                with g1: marketing_status = st.selectbox("Marketing", ["Submitted", "Needs revision", "Withdrawn"], index=0)
                with g2: supply_status = st.selectbox("Supply Chain", ["Pending review", "Validated", "Returned for revision"], index=0)
                with g3: finance_status = st.selectbox("Finance", ["Pending", "Approved", "Conditional approval", "Rejected", "Returned for revision"], index=0)
                with g4: dbm_status = st.selectbox("DBM coordination", ["In coordination", "Ready for release", "Escalated"], index=0)
                approver_note = st.text_input("Decision note / conditions", placeholder="Record rationale, owner or conditions")
                record = st.form_submit_button("Record decision in audit log", use_container_width=True)
            if record:
                log_item = recommendation.copy()
                log_item.update({"marketing_status": marketing_status, "supply_chain_status": supply_status,
                                 "finance_status": finance_status, "dbm_status": dbm_status,
                                 "decision_note": approver_note, "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                st.session_state.decision_log.append(log_item)
                st.success("Decision recorded in this session's NEXUS audit log.")

    if st.session_state.decision_log:
        section("Decision history", "Session-level audit trail. Export it from Governance & Export.", "DBM CONTROL")
        log_df = pd.DataFrame(st.session_state.decision_log)
        st.dataframe(log_df[[c for c in ["recorded_at", "product", "suggested_purchase_qty", "purchase_value", "marketing_status", "supply_chain_status", "finance_status", "dbm_status", "decision_note"] if c in log_df]], use_container_width=True, hide_index=True)

elif page == "Governance & Export":
    hero("TRUST, CONTROL & TRANSPARENCY", "Intelligence with an audit trail.", "NEXUS provides evidence and workflow control while keeping forecast ownership, procurement recommendations and financial authorisation with the people accountable for them.", "GOVERNANCE  /  EXPORT CENTRE")
    section("Decision rights", "A connected process does not mean shared or ambiguous accountability.", "RACI SNAPSHOT")
    raci = pd.DataFrame([
        ["Submit sales forecast & campaign assumptions", "A/R", "C", "I", "C"],
        ["Validate forecast evidence & explain variance", "R", "C", "I", "A"],
        ["Assess stock, lead time & procurement quantity", "C", "A/R", "I", "C"],
        ["Evaluate budget, margin & cash impact", "C", "C", "A/R", "I"],
        ["Coordinate workflow, data standards & audit", "C", "C", "C", "A/R"],
        ["Approve purchase within delegated authority", "I", "R", "A", "C"],
        ["Monitor model performance & drift", "C", "C", "I", "A/R"],
    ], columns=["Business activity", "Marketing", "Supply Chain", "Finance", "DBM"])
    st.dataframe(raci, use_container_width=True, hide_index=True)
    st.caption("A = Accountable · R = Responsible · C = Consulted · I = Informed. Actual approval authority must follow the organisation’s delegation-of-authority policy.")

    section("Model governance rules", "Use these rules before operationalising any model output.", "CONTROL FRAMEWORK")
    governance = [
        ("Data lineage", "Every forecast and procurement recommendation should point to its source data, period, owner and version."),
        ("Forecast validation", "Use rolling-origin / walk-forward testing for time series; report MAE, RMSE, WAPE and forecast bias alongside R² where appropriate."),
        ("Uncertainty", "Do not interpret ±2,000 as a confidence interval until the measure, unit, sample and coverage level are documented."),
        ("Human approval", "Algorithms recommend and classify; authorised people approve procurement and financial commitments."),
        ("Explainability", "Expose key drivers, class probabilities where available, thresholds, training period and known limitations."),
        ("Monitoring", "Track forecast errors, class-level precision/recall, data drift, overrides and business outcomes."),
        ("Privacy & access", "Apply least-privilege access, data minimisation, retention policies and secure handling for customer/supplier data."),
        ("Change control", "Version datasets, features, model settings and approval rules; test changes before release."),
    ]
    for title, copy in governance:
        st.markdown(f'<div style="padding:11px 0;border-bottom:1px solid rgba(145,175,200,.09)"><div style="color:#E9F2FB;font-size:12px;font-weight:800">{title}</div><div class="muted" style="font-size:11px;margin-top:3px">{copy}</div></div>', unsafe_allow_html=True)

    section("Business KPI catalogue", "Set baselines before setting targets. Values below are definitions, not promised improvements.", "VALUE REALISATION")
    kpis = pd.DataFrame([
        ["Forecast WAPE", "Σ|Actual − Forecast| ÷ ΣActual", "Marketing / DBM", "Weekly"],
        ["Forecast bias", "Σ(Forecast − Actual) ÷ ΣActual", "Marketing", "Weekly"],
        ["Stockout rate", "Stockout product-location-days ÷ total product-location-days", "Supply Chain", "Weekly"],
        ["Inventory days of cover", "Usable inventory ÷ average daily demand", "Supply Chain", "Weekly"],
        ["Procurement approval cycle", "Approval timestamp − complete request timestamp", "Finance / DBM", "Monthly"],
        ["Budget variance", "Actual procurement spend − approved budget", "Finance", "Monthly"],
        ["Forecast override rate", "Manually overridden forecasts ÷ submitted forecasts", "DBM", "Monthly"],
        ["Model drift", "Change in input or outcome distributions / performance", "DBM", "Monthly"],
    ], columns=["KPI", "Definition", "Owner", "Frequency"])
    st.dataframe(kpis, use_container_width=True, hide_index=True)

    section("Open decisions before PRD", "Resolve these with the business owners before detailed product design begins.", "BRD → PRD GATE")
    open_questions = [
        "Are 40% and 30% shares of the total horizon, weekly growth rates or campaign-specific targets? What is the remaining forecast period?",
        "What exactly does ±2,000 represent, in which unit, at what level and with what statistical coverage?",
        "Which system is the source of truth for sales, inventory, purchase orders, supplier lead times and finance budgets?",
        "What is the organisation’s material forecast-variance threshold and escalation policy?",
        "Who can approve procurement at each monetary threshold, and who can authorise exceptions?",
        "How many years / weeks of clean historical sales and transaction-level basket data are available?",
        "Which product, region and replenishment horizon should be used for the first pilot?",
    ]
    for q in open_questions:
        st.markdown(f"- {q}")

    st.markdown("---")
    section("Export your NEXUS decision pack", "Download a portable snapshot of the source data and the decisions created in this session.", "EXPORT CENTRE")
    pack = io.BytesIO()
    with zipfile.ZipFile(pack, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("sales_data.csv", sales_df.to_csv(index=False))
        if not tx_df.empty:
            archive.writestr("transaction_data.csv", tx_df.to_csv(index=False))
        if st.session_state.decision_log:
            archive.writestr("procurement_decision_log.csv", pd.DataFrame(st.session_state.decision_log).to_csv(index=False))
        if not st.session_state.latest_forecast.empty:
            archive.writestr("latest_arima_forecast.csv", st.session_state.latest_forecast.to_csv(index=False))
        for key, filename in [("association_rules", "association_rules.csv"), ("regression_features", "regression_feature_importance.csv"), ("classification_report", "classification_report.csv"), ("clustering_profile", "clustering_profile.csv")]:
            value = st.session_state.latest_algorithm_results.get(key)
            if isinstance(value, pd.DataFrame):
                archive.writestr(filename, value.to_csv(index=False))
        readme = "NEXUS decision pack\n\nThis export is a session snapshot. Model outputs are decision support, not approvals. Validate source data, model performance and business assumptions before operational use.\n"
        archive.writestr("README.txt", readme)
    st.download_button("↓ Download complete NEXUS decision pack (.zip)", pack.getvalue(), "nexus_decision_pack.zip", "application/zip", use_container_width=True)
    st.caption("Session storage note: this demonstration keeps decisions in Streamlit session state. For a production multi-user deployment, connect a governed database and implement authentication, role-based access, durable audit logs and secrets management.")


# Footer on every page
st.markdown("<div style='margin-top:35px;padding:13px 0;border-top:1px solid rgba(145,175,200,.10);display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;color:#667D93;font:9px DM Mono,monospace;letter-spacing:1px'><span>NEXUS / BUSINESS INTELLIGENCE OS</span><span>DECISION SUPPORT · HUMAN ACCOUNTABILITY · VERSION 1.0</span></div>", unsafe_allow_html=True)
