"""
=============================================================
  KATTEGAT DIGITAL TWIN -- DAY 6
  Interactive Streamlit Dashboard
  Vessel : M/V Kattegat (IMO 9112765) -- DFDS
=============================================================

HOW TO RUN:
-----------
  1. Install dependencies (only once):
       pip3 install streamlit plotly

  2. Run the dashboard:
       cd /Users/mbp/Desktop/Kattegat_Data
       streamlit run kattegat_day6_dashboard.py

  3. Browser opens automatically at:
       http://localhost:8501

  The dashboard reads your CSV files produced by Days 1-5.
  Make sure these files are in the same folder:
    - kattegat_daily_fixed.csv
    - kattegat_cii_monthly.csv
    - kattegat_day3_forecast.csv
    - kattegat_optimizer_results.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

warnings.filterwarnings("ignore")

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title  = "Kattegat Digital Twin",
    page_icon   = "🚢",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# =============================================================
# CONSTANTS
# =============================================================
REAL_BASELINE    = 58.2
FOULING_RATE     = 1.46
FUEL_PRICE_USD   = 620
DRYDOCK_WORK_USD = 300000   # Corrected: Schultz et al.(2011) + bottom-up
OFFHIRE_DAY_USD  = 50000    # Corrected: DFDS Annual Report 2024
DRYDOCK_DAYS     = 30       # Corrected: confirmed 4 Mar – 3 Apr 2025
TOTAL_DD_USD     = DRYDOCK_WORK_USD + OFFHIRE_DAY_USD * DRYDOCK_DAYS
AVG_ME_TONS_DAY  = 8.5
DRYDOCK_END      = pd.to_datetime("2025-04-04")
STUDY_END        = pd.to_datetime("2026-05-02")   # optimiser start
DFDS_LIMIT_M     = 11   # months from study end = Apr 2027 (24m from last DD)
SOLAS_LIMIT_M    = 23   # months from study end = Apr 2028 (36m from last DD SOLAS Reg I/7)
MONTHS_ELAPSED   = 13   # months already elapsed since last DD by study end
GT               = 14379

# Rating colours
RATING_COLORS = {
    "A": "#27AE60", "B": "#2ECC71",
    "C": "#F1C40F", "D": "#E67E22",
    "E": "#E74C3C"
}

# =============================================================
# DATA LOADING
# =============================================================
DATA_FOLDER = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    daily    = pd.read_csv(
        os.path.join(DATA_FOLDER, "kattegat_daily_fixed.csv"),
        parse_dates=["date"])
    cii      = pd.read_csv(
        os.path.join(DATA_FOLDER, "kattegat_cii_monthly.csv"))
    forecast = pd.read_csv(
        os.path.join(DATA_FOLDER, "kattegat_day3_forecast.csv"),
        parse_dates=["date"])
    optim    = pd.read_csv(
        os.path.join(DATA_FOLDER, "kattegat_optimizer_results.csv"),
        parse_dates=["optimal_date"])
    return daily, cii, forecast, optim

try:
    daily, cii_df, forecast_df, optim_df = load_data()
    data_ok = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Filter operational days
daily_op = daily[daily["period"] != "Drydock"].copy()
daily_op = daily_op.sort_values("date").reset_index(drop=True)

pre  = daily_op[daily_op["period"] == "Pre-Drydock"]
post = daily_op[daily_op["period"] == "Post-Drydock"]

# Latest values
latest      = daily_op.iloc[-1]
latest_fi   = daily_op["hm_fouling_pct"].dropna().tail(30).mean()
latest_fi   = max(latest_fi, 0) if not np.isnan(latest_fi) else 8.0
latest_date = daily_op["date"].max()
latest_cii  = daily_op["daily_cii"].dropna().tail(30).mean()

# Optimal drydock
opt_month = optim_df.loc[optim_df["total_cost_usd"].idxmin()]
opt_date  = pd.to_datetime(opt_month["optimal_date"])

# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
             width=50)
    st.title("🚢 Kattegat Digital Twin")
    st.markdown("**M/V Kattegat (IMO 9112765)**")
    st.markdown("DFDS | Algeciras – Tanger Med")
    st.divider()

    st.subheader("⚙️ Parameters")
    fuel_price = st.slider(
        "Fuel price (USD/ton)",
        min_value=400, max_value=1000,
        value=FUEL_PRICE_USD, step=20)

    drydock_cost = st.slider(
        "Drydock cost (USD)",
        min_value=150000, max_value=500000,
        value=DRYDOCK_WORK_USD, step=5000)   # default corrected to 300,000

    offhire = st.slider(
        "Off-hire rate (USD/day)",
        min_value=20000, max_value=60000,
        value=OFFHIRE_DAY_USD, step=5000)   # default corrected to 50,000

    dd_days = st.slider(
        "Drydock duration (days)",
        min_value=7, max_value=30,
        value=DRYDOCK_DAYS, step=1)          # default corrected to 30 days

    total_dd_custom = drydock_cost + offhire * dd_days
    st.metric("Total drydock cost", f"${total_dd_custom:,.0f}")

    st.divider()
    st.caption(f"Data updated: {latest_date.date()}")
    st.caption("© Master Thesis 2025")

# =============================================================
# HERO BANNER — CINEMATIC MOVING SHIP
# =============================================================
import base64

# Load ship image
local_photo = os.path.join(DATA_FOLDER, "kattegat_ship.png")
if os.path.exists(local_photo):
    with open(local_photo, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ship_src = f"data:image/png;base64,{b64}"
else:
    ship_src = ""

latest_fi_hero = latest_fi
post_eff       = post["hfo_per_nm"].tail(30).mean()

st.markdown(f"""
<style>

/* ── Animations ──────────────────────────────────────── */

/* Ship panning: moves as background left→right giving
   illusion of forward sailing motion                    */
@keyframes shipSail {{
    0%   {{ background-position: 60% 55%; }}
    50%  {{ background-position: 40% 58%; }}
    100% {{ background-position: 60% 55%; }}
}}

/* Subtle zoom pulse — breathing effect */
@keyframes shipZoom {{
    0%,100% {{ background-size: 110% auto; }}
    50%     {{ background-size: 115% auto; }}
}}

/* Wave scroll */
@keyframes waveMove {{
    0%   {{ transform: translateX(0);    }}
    100% {{ transform: translateX(-50%); }}
}}
@keyframes waveMove2 {{
    0%   {{ transform: translateX(-50%); }}
    100% {{ transform: translateX(0);    }}
}}

/* Text fade in */
@keyframes fadeDown {{
    from {{ opacity:0; transform:translateY(-18px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
@keyframes fadeUp {{
    from {{ opacity:0; transform:translateY(18px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
@keyframes glow {{
    0%,100% {{ opacity:1; text-shadow: 0 0 10px rgba(93,235,181,0.4); }}
    50%     {{ opacity:0.8; text-shadow: 0 0 20px rgba(93,235,181,0.8); }}
}}

/* ── Hero container ──────────────────────────────────── */
.hero {{
    position           : relative;
    width              : 100%;
    height             : 360px;
    border-radius      : 16px;
    overflow           : hidden;
    margin-bottom      : 28px;
    box-shadow         : 0 16px 48px rgba(0,0,0,0.6);
    background-image   : url('{ship_src}');
    background-repeat  : no-repeat;
    background-size    : 112% auto;
    background-position: 55% 55%;
    animation          : shipSail 12s ease-in-out infinite,
                         shipZoom 12s ease-in-out infinite;
}}

/* Cinematic overlay: darker edges, clear centre */
.hero-overlay {{
    position   : absolute;
    inset      : 0;
    background : linear-gradient(
        180deg,
        rgba(4,14,30,0.78) 0%,
        rgba(4,14,30,0.20) 28%,
        rgba(4,14,30,0.05) 50%,
        rgba(4,14,30,0.20) 72%,
        rgba(4,14,30,0.82) 100%
    );
    z-index    : 2;
}}

/* Left and right vignette */
.hero-vignette {{
    position   : absolute;
    inset      : 0;
    background : linear-gradient(
        90deg,
        rgba(4,14,30,0.55) 0%,
        transparent 20%,
        transparent 80%,
        rgba(4,14,30,0.55) 100%
    );
    z-index    : 3;
}}

/* ── Wave overlay at bottom ──────────────────────────── */
.hero-waves {{
    position : absolute;
    bottom   : 52px;
    left     : 0; right: 0;
    height   : 40px;
    overflow : hidden;
    z-index  : 4;
    opacity  : 0.45;
}}
.wave-row {{
    position          : absolute;
    width             : 200%;
    height            : 100%;
    bottom            : 0;
    background-repeat : repeat-x;
    background-size   : 50% 100%;
}}
.w1 {{
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 40'%3E%3Cpath fill='white' fill-opacity='0.35' d='M0,20 C100,0 200,40 300,20 C400,0 500,40 600,20 C700,0 800,40 800,20 L800,40 L0,40Z'/%3E%3C/svg%3E");
    animation: waveMove 5s linear infinite;
    bottom: 8px;
}}
.w2 {{
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 40'%3E%3Cpath fill='white' fill-opacity='0.25' d='M0,12 C120,32 240,0 360,16 C480,32 600,0 720,16 C760,24 800,12 800,16 L800,40 L0,40Z'/%3E%3C/svg%3E");
    animation: waveMove2 7s linear infinite;
    bottom: 0;
}}

/* ── Title block ─────────────────────────────────────── */
.hero-title-block {{
    position  : absolute;
    top       : 22px; left: 26px;
    z-index   : 10;
    animation : fadeDown 0.8s ease both;
}}
.hero-main-title {{
    font-size    : 27px;
    font-weight  : 800;
    color        : #ffffff;
    margin       : 0;
    line-height  : 1.25;
    text-shadow  : 0 2px 14px rgba(0,0,0,0.95);
    letter-spacing: 0.3px;
}}
.hero-sub-title {{
    font-size  : 12.5px;
    color      : rgba(255,255,255,0.80);
    margin-top : 6px;
    text-shadow: 0 1px 6px rgba(0,0,0,0.8);
}}

/* ── Route badge ─────────────────────────────────────── */
.hero-badge {{
    position       : absolute;
    top: 22px; right: 22px;
    background     : rgba(255,255,255,0.10);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border         : 1px solid rgba(255,255,255,0.22);
    border-radius  : 12px;
    padding        : 10px 18px;
    text-align     : center;
    z-index        : 10;
    animation      : fadeDown 0.8s ease 0.15s both;
}}
.hero-badge-route {{
    font-size : 14px; font-weight: 700; color: #fff;
    text-shadow: 0 1px 5px rgba(0,0,0,0.7);
}}
.hero-badge-sub {{
    font-size: 11px; color: rgba(255,255,255,0.70); margin-top: 3px;
}}

/* ── Thesis tag ──────────────────────────────────────── */
.hero-thesis-tag {{
    position       : absolute;
    bottom         : 66px; left: 26px;
    background     : rgba(4,14,30,0.70);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border         : 1px solid rgba(93,235,181,0.40);
    border-radius  : 8px;
    padding        : 5px 13px;
    font-size      : 10.5px;
    color          : #5DEBB5;
    font-weight    : 600;
    letter-spacing : 0.4px;
    z-index        : 10;
    animation      : fadeUp 0.8s ease 0.4s both;
}}

/* ── Speed indicator (right side) ───────────────────── */
.hero-speed {{
    position       : absolute;
    bottom         : 66px; right: 22px;
    background     : rgba(4,14,30,0.70);
    backdrop-filter: blur(8px);
    border         : 1px solid rgba(255,255,255,0.15);
    border-radius  : 8px;
    padding        : 5px 14px;
    text-align     : center;
    z-index        : 10;
    animation      : fadeUp 0.8s ease 0.5s both;
}}
.hero-speed-val {{
    font-size: 16px; font-weight: 700; color: #fff;
}}
.hero-speed-lbl {{
    font-size: 9.5px; color: rgba(255,255,255,0.60);
    text-transform: uppercase; letter-spacing: 0.5px;
}}

/* ── KPI strip ───────────────────────────────────────── */
.hero-kpi-strip {{
    position       : absolute;
    bottom: 0; left: 0; right: 0;
    display        : flex;
    z-index        : 10;
    background     : rgba(4,14,30,0.80);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-top     : 1px solid rgba(255,255,255,0.08);
    animation      : fadeUp 0.8s ease 0.25s both;
}}
.hero-kpi-item {{
    flex       : 1;
    padding    : 12px 6px;
    text-align : center;
    border-right: 1px solid rgba(255,255,255,0.07);
}}
.hero-kpi-item:last-child {{ border-right: none; }}
.hero-kpi-val {{
    display    : block;
    font-size  : 18px;
    font-weight: 700;
    color      : #5DEBB5;
    line-height: 1.1;
    animation  : glow 3s ease-in-out infinite;
}}
.hero-kpi-label {{
    display       : block;
    font-size     : 9px;
    color         : rgba(255,255,255,0.55);
    margin-top    : 3px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}

</style>

<div class="hero">
  <div class="hero-overlay"></div>
  <div class="hero-vignette"></div>
  <div class="hero-waves">
    <div class="wave-row w1"></div>
    <div class="wave-row w2"></div>
  </div>

  <div class="hero-title-block">
    <p class="hero-main-title">🚢 M/V Kattegat — Hull Performance Digital Twin</p>
    <p class="hero-sub-title">
      IMO 9112765 &nbsp;·&nbsp; DFDS &nbsp;·&nbsp; Built 1996 &nbsp;·&nbsp;
      Ro-Pax Ferry &nbsp;·&nbsp; Algeciras – Tanger Med &nbsp;·&nbsp; Jan 2024 – May 2026
    </p>
  </div>

  <div class="hero-badge">
    <div class="hero-badge-route">⚓ Algeciras ↕ Tanger Med</div>
    <div class="hero-badge-sub">~20 nm · 8 crossings/day</div>
  </div>

  <div class="hero-thesis-tag">
    Master Thesis · Marine Engineering · Hybrid Physics-ML Digital Twin
  </div>

  <div class="hero-speed">
    <div class="hero-speed-val">~19 kn</div>
    <div class="hero-speed-lbl">Service Speed</div>
  </div>

  <div class="hero-kpi-strip">
    <div class="hero-kpi-item">
      <span class="hero-kpi-val">{latest_fi_hero:.1f}%</span>
      <span class="hero-kpi-label">Fouling Index</span>
    </div>
    <div class="hero-kpi-item">
      <span class="hero-kpi-val">{post_eff:.1f} kg/nm</span>
      <span class="hero-kpi-label">HFO Efficiency</span>
    </div>
    <div class="hero-kpi-item">
      <span class="hero-kpi-val">Rating B</span>
      <span class="hero-kpi-label">CII 2025</span>
    </div>
    <div class="hero-kpi-item">
      <span class="hero-kpi-val">Jun 2026</span>
      <span class="hero-kpi-label">Optimal Drydock</span>
    </div>
    <div class="hero-kpi-item">
      <span class="hero-kpi-val">+1.46%</span>
      <span class="hero-kpi-label">Fouling/Month</span>
    </div>
    <div class="hero-kpi-item">
      <span class="hero-kpi-val">8.4%</span>
      <span class="hero-kpi-label">Drydock Gain</span>
    </div>
    <div class="hero-kpi-item">
      <span class="hero-kpi-val">$90.8k</span>
      <span class="hero-kpi-label">Fouling Cost</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# =============================================================
# KPI ROW -- 6 METRICS
# =============================================================
k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    fi_delta = latest_fi - FOULING_RATE
    st.metric(
        "Hull Fouling Index",
        f"{latest_fi:.1f}%",
        delta=f"+{FOULING_RATE:.1f}%/month",
        delta_color="inverse")

with k2:
    pre_avg  = pre["hfo_per_nm"].mean()
    post_avg = post["hfo_per_nm"].mean()
    saving   = pre_avg - post_avg
    st.metric(
        "Fuel Efficiency (HFO/nm)",
        f"{post_avg:.1f} kg/nm",
        delta=f"{-saving:.1f} vs pre-drydock",
        delta_color="normal")

with k3:
    # Use monthly CII from cii_df (correct scale ~18-20 g/GT·nm)
    # Per-voyage daily_cii is ~65-70 g/GT·nm (voyage level, not annual)
    try:
        cii_latest = cii_df["cii_monthly"].dropna().iloc[-1]
    except:
        cii_latest = latest_cii if not np.isnan(latest_cii) else 18.5
    cii_ref = 2023 * (GT ** (-0.460))   # MEPC.353(78) Table 1 Ro-Pax
    cii_req = cii_ref * 0.91             # 9% reduction 2025
    # Rating boundaries applied against cii_req (already reduced)
    if   cii_latest < cii_req * 0.86: rating = "A"
    elif cii_latest < cii_req * 0.94: rating = "B"
    elif cii_latest < cii_req * 1.06: rating = "C"
    elif cii_latest < cii_req * 1.18: rating = "D"
    else:                              rating = "E"
    st.metric(
        "Current CII Rating",
        rating,
        delta=f"{cii_latest:.1f} g CO₂/GT·nm")

with k4:
    days_to_dd = (opt_date - latest_date).days
    st.metric(
        "Days to Optimal Drydock",
        f"{days_to_dd}",
        delta=f"{opt_date.strftime('%b %Y')}",
        delta_color="off")

with k5:
    total_waste = daily_op["extra_cost_usd"].sum() if "extra_cost_usd" in daily_op else 0
    daily_op_copy = daily_op.copy()
    daily_op_copy["extra_fuel"] = (
        daily_op_copy["hfo_per_nm"] - REAL_BASELINE
    ).clip(lower=0) * daily_op_copy["distance_nm"] / 1000
    daily_op_copy["extra_cost"] = daily_op_copy["extra_fuel"] * fuel_price
    total_waste_usd = daily_op_copy["extra_cost"].sum()
    st.metric(
        "Historical Fouling Cost",
        f"${total_waste_usd:,.0f}",
        delta="Jan 2024 – May 2026",
        delta_color="off")

with k6:
    opt_cost = optim_df["total_cost_usd"].min()
    st.metric(
        "Optimal Total Cost",
        f"${opt_cost/1e6:.2f}M",
        delta=f"Month {int(opt_month['months_from_now'])}",
        delta_color="off")

st.divider()

# =============================================================
# TAB LAYOUT
# =============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Hull Performance",
    "🤖 ML Forecast",
    "🌍 CII Compliance",
    "🔧 Drydock Optimizer",
    "📋 Data Summary"
])

# ── TAB 1: Hull Performance ───────────────────────────────────
with tab1:
    st.subheader("Hull Performance Monitoring — Holtrop-Mennen Digital Twin")

    col_left, col_right = st.columns([3, 1])

    with col_left:
        # Main fuel efficiency plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=pre["date"], y=pre["hfo_per_nm"],
            mode="markers", name="Pre-drydock",
            marker=dict(color="#E74C3C", size=4, opacity=0.5)))

        fig.add_trace(go.Scatter(
            x=post["date"], y=post["hfo_per_nm"],
            mode="markers", name="Post-drydock",
            marker=dict(color="#27AE60", size=4, opacity=0.5)))

        # Rolling average
        roll = daily_op["hfo_per_nm"].rolling(14, center=True, min_periods=5).mean()
        fig.add_trace(go.Scatter(
            x=daily_op["date"], y=roll,
            mode="lines", name="14-day average",
            line=dict(color="#2C3E50", width=2.5)))

        # Baseline
        fig.add_hline(
            y=REAL_BASELINE,
            line_dash="dash", line_color="#8E44AD",
            annotation_text=f"Clean hull baseline ({REAL_BASELINE} kg/nm)")

        # Drydock markers
        fig.add_vline(x=pd.Timestamp("2025-03-04").timestamp()*1000,
                      line_dash="dot",
                      line_color="#E67E22", annotation_text="Drydock start")
        fig.add_vline(x=pd.Timestamp("2025-04-04").timestamp()*1000,
                      line_dash="dot",
                      line_color="#27AE60", annotation_text="First voyage")

        fig.update_layout(
            title="HFO Consumption per Nautical Mile — Distance Normalised",
            xaxis_title="Date", yaxis_title="HFO (kg/nm)",
            height=400, legend=dict(orientation="h", y=-0.2),
            template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Fouling gauge
        fi_val = min(latest_fi, 50)
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = fi_val,
            delta = {"reference": 0, "suffix": "%",
                     "valueformat": ".1f"},
            number= {"suffix": "%", "valueformat": ".1f"},
            title = {"text": "Hull Fouling Index<br><sub>Current State</sub>"},
            gauge = {
                "axis"  : {"range": [-10, 40]},
                "bar"   : {"color": "#E74C3C" if fi_val > 15 else
                                    "#E67E22" if fi_val > 8  else "#27AE60"},
                "steps" : [
                    {"range": [-10, 5],  "color": "#D5F5E3"},
                    {"range": [5,  10],  "color": "#FCF3CF"},
                    {"range": [10, 20],  "color": "#FADBD8"},
                    {"range": [20, 40],  "color": "#E74C3C"},
                ],
                "threshold": {
                    "line" : {"color": "#C0392B", "width": 4},
                    "value": 20
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=50,b=10,l=10,r=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Hull condition summary
        condition = (
            "🟢 Good — below 10% threshold" if latest_fi < 10
            else "🟡 Moderate — above 10% threshold" if latest_fi < 20
            else "🔴 Severe — above 20% threshold"
        )
        st.info(f"**Hull Condition:** {condition}")
        st.metric("Clean hull baseline", f"{REAL_BASELINE} kg/nm")
        st.metric("Current efficiency",
                  f"{post['hfo_per_nm'].tail(30).mean():.1f} kg/nm")
        st.metric("Degradation rate", f"+{FOULING_RATE:.2f}%/month")

    # Fouling index timeline
    st.subheader("Fouling Index Timeline")
    fig2 = go.Figure()

    colors_fi = ["#E74C3C" if x > 0 else "#27AE60"
                 for x in daily_op["hm_fouling_pct"].fillna(0)]
    fig2.add_trace(go.Bar(
        x=daily_op["date"],
        y=daily_op["hm_fouling_pct"],
        marker_color=colors_fi,
        opacity=0.6, name="Fouling index"))

    fi_roll = daily_op["hm_fouling_pct"].rolling(14, center=True, min_periods=5).mean()
    fig2.add_trace(go.Scatter(
        x=daily_op["date"], y=fi_roll,
        mode="lines", name="14-day average",
        line=dict(color="#2C3E50", width=2.5)))

    fig2.add_hline(y=10, line_dash="dot", line_color="#E67E22",
                   annotation_text="10% threshold")
    fig2.add_hline(y=20, line_dash="dot", line_color="#E74C3C",
                   annotation_text="20% threshold")
    fig2.add_hline(y=0,  line_color="black", line_width=1)
    fig2.add_vline(x=pd.Timestamp("2025-04-04").timestamp()*1000,
                   line_dash="dot",
                   line_color="#27AE60")

    fig2.update_layout(
        title="Physics-Based Fouling Index | Holtrop-Mennen Calibrated Method",
        xaxis_title="Date", yaxis_title="Fouling Index (%)",
        height=350, template="plotly_white",
        legend=dict(orientation="h", y=-0.2))

    st.plotly_chart(fig2, use_container_width=True)

# ── TAB 2: ML Forecast ────────────────────────────────────────
with tab2:
    st.subheader("Machine Learning Fouling Forecast — Extra Trees Model")

    col_a, col_b = st.columns([2, 1])

    with col_a:
        fig3 = go.Figure()

        # Historical
        fi_roll2 = daily_op["hm_fouling_pct"].rolling(
            14, center=True, min_periods=5).mean()
        fig3.add_trace(go.Scatter(
            x=daily_op["date"], y=fi_roll2,
            mode="lines", name="Historical (14-day avg)",
            line=dict(color="#2C3E50", width=2)))

        # Forecast
        if len(forecast_df) > 0:
            fig3.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["ml_forecast"],
                mode="lines", name="ML Forecast",
                line=dict(color="#8E44AD", width=2.5, dash="dash")))

            if "forecast_upper" in forecast_df.columns:
                fig3.add_trace(go.Scatter(
                    x=pd.concat([forecast_df["date"],
                                 forecast_df["date"][::-1]]),
                    y=pd.concat([forecast_df["forecast_upper"],
                                 forecast_df["forecast_lower"][::-1]]),
                    fill="toself", fillcolor="rgba(142,68,173,0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="95% Confidence interval"))

        fig3.add_hline(y=10, line_dash="dot", line_color="#E67E22",
                       annotation_text="10% threshold")
        fig3.add_hline(y=20, line_dash="dot", line_color="#E74C3C",
                       annotation_text="20% threshold")
        fig3.add_vline(x=latest_date.timestamp()*1000,
                       line_dash="dot", line_color="black",
                       annotation_text="Forecast start")

        fig3.update_layout(
            title="6-Month Hull Fouling Forecast with 95% Confidence Interval",
            xaxis_title="Date", yaxis_title="Fouling Index (%)",
            height=400, template="plotly_white",
            legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.markdown("### Model Performance")
        st.metric("Best Model",      "Extra Trees")
        st.metric("R² (test set)",   "0.6433")
        st.metric("MAE",             "2.40%")
        st.metric("RMSE",            "3.54%")
        st.metric("Training period", "Jan 2024 – Sep 2025")
        st.metric("Test period",     "Oct 2025 – May 2026")

        st.markdown("### Feature Importance")
        features = {
            "hfo_roll7": 0.353,
            "hfo_diff7": 0.185,
            "hfo_roll14": 0.170,
            "is_post_dd": 0.113,
            "hfo_lag7":  0.052,
        }
        fig_imp = go.Figure(go.Bar(
            x=list(features.values()),
            y=list(features.keys()),
            orientation="h",
            marker_color=["#2980B9","#5DADE2","#85C1E9",
                          "#AED6F1","#D6EAF8"]))
        fig_imp.update_layout(
            height=250, margin=dict(t=10,b=10,l=10,r=10),
            template="plotly_white",
            xaxis_title="Importance")
        st.plotly_chart(fig_imp, use_container_width=True)

# ── TAB 3: CII Compliance ─────────────────────────────────────
with tab3:
    st.subheader("IMO CII Compliance — MEPC.354(78) Ro-Pax Formula")

    cii_df["date"] = pd.to_datetime(cii_df["date"])

    col_c1, col_c2 = st.columns([3, 1])

    with col_c1:
        fig_cii = go.Figure()

        colors_cii = [RATING_COLORS.get(r,"#95A5A6")
                      for r in cii_df["rating"]]
        fig_cii.add_trace(go.Bar(
            x=cii_df["date"],
            y=cii_df["cii_monthly"],
            marker_color=colors_cii,
            name="Monthly CII"))

        # Required line
        fig_cii.add_trace(go.Scatter(
            x=cii_df["date"],
            y=cii_df["cii_required"],
            mode="lines", name="Required CII",
            line=dict(color="#2C3E50", width=2, dash="dash")))

        fig_cii.add_vline(x=pd.Timestamp("2025-04-04").timestamp()*1000,
                          line_dash="dot",
                          line_color="#27AE60",
                          annotation_text="Post-drydock")

        fig_cii.update_layout(
            title="Monthly CII Rating | Green = Rating B/A | Dashed = Required CII",
            xaxis_title="Month",
            yaxis_title="CII (g CO₂/GT·nm)",
            height=400, template="plotly_white",
            legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_cii, use_container_width=True)

    with col_c2:
        st.markdown("### Annual Results")
        cii_df["year"] = cii_df["date"].dt.year
        annual = cii_df.groupby("year").agg(
            cii=("cii_monthly","mean"),
            rating=("rating","first")
        ).reset_index()

        for _, row in annual.iterrows():
            color = RATING_COLORS.get(row["rating"],"#95A5A6")
            st.markdown(
                f"**{int(row['year'])}**: "
                f"<span style='color:{color};font-size:20px;"
                f"font-weight:bold;'>Rating {row['rating']}</span> "
                f"— CII = {row['cii']:.2f}",
                unsafe_allow_html=True)

        st.divider()
        st.markdown("### CII Improvement")
        pre_cii  = cii_df[cii_df["period"]=="Pre-Drydock"]["cii_monthly"].mean()
        post_cii = cii_df[cii_df["period"]=="Post-Drydock"]["cii_monthly"].mean()
        if not (np.isnan(pre_cii) or np.isnan(post_cii)):
            pct = (pre_cii - post_cii) / pre_cii * 100
            st.metric("Pre-drydock CII",  f"{pre_cii:.2f}")
            st.metric("Post-drydock CII", f"{post_cii:.2f}",
                      delta=f"-{pct:.1f}% improvement",
                      delta_color="normal")

        st.divider()
        cii_ref = 2023 * (GT ** (-0.460))   # MEPC.353(78) Table 1 Ro-Pax
        st.markdown("### IMO Reference")
        st.caption(f"CII_ref (Ro-Pax): {cii_ref:.2f}")
        st.caption("Capacity = GT = 14,379")
        st.caption("Formula: a×GT^(-c)")
        st.caption("a=2023, c=0.460  — MEPC.353(78) Table 1")

# ── TAB 4: Drydock Optimizer ──────────────────────────────────
with tab4:
    st.subheader("Predictive Drydocking Optimization — Techno-Economic Model")

    # Recalculate with sidebar values
    # Regulatory horizon: SOLAS Reg I/7 — Ro-Pax max 36m between DDs
    # Last DD April 2025 → hard limit April 2028 = Month 23 from May 2026
    horizon  = SOLAS_LIMIT_M  # = 23 months
    months_r = np.arange(1, horizon+1)
    costs    = []
    for m in months_r:
        wb = sum([AVG_ME_TONS_DAY*(latest_fi+FOULING_RATE*i)/100
                  * fuel_price*30 for i in range(1,m+1)])
        ma = horizon - m
        wa = (sum([AVG_ME_TONS_DAY*(FOULING_RATE*i)/100*fuel_price*30
                   for i in range(1,ma+1)]) if ma > 0 else 0)

        costs.append(wb + total_dd_custom + wa)

    costs    = np.array(costs)
    opt_m    = months_r[np.argmin(costs)]
    opt_c    = costs[np.argmin(costs)]
    opt_fi   = latest_fi + FOULING_RATE * opt_m
    opt_date_custom = (STUDY_END + pd.DateOffset(months=int(opt_m))).strftime("%B %Y")

    col_d1, col_d2 = st.columns([3, 1])

    with col_d1:
        fig_opt = go.Figure()

        fig_opt.add_trace(go.Scatter(
            x=months_r, y=costs/1000,
            mode="lines", name="Total lifecycle cost",
            line=dict(color="#2C3E50", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(44,62,80,0.08)"))

        fig_opt.add_vline(
            x=opt_m, line_dash="dash",
            line_color="#8E44AD", line_width=2.5,
            annotation_text=f"Optimal: Month {opt_m} ({opt_date_custom})")

        # Regulatory boundaries
        fig_opt.add_vline(x=DFDS_LIMIT_M, line_dash="dot",
                          line_color="#E67E22", line_width=2,
                          annotation_text="DFDS 24m (Apr 2027)")
        fig_opt.add_vline(x=SOLAS_LIMIT_M, line_dash="dashdot",
                          line_color="#C0392B", line_width=2.5,
                          annotation_text="SOLAS limit (Apr 2028)")
        month_10 = (10.0 - latest_fi) / FOULING_RATE
        month_20 = (20.0 - latest_fi) / FOULING_RATE
        if month_10 > 0:
            fig_opt.add_vline(x=float(month_10), line_dash="dot",
                              line_color="#E67E22",
                              annotation_text="10% fouling")
        if month_20 > 0:
            fig_opt.add_vline(x=float(month_20), line_dash="dot",
                              line_color="#E74C3C",
                              annotation_text="20% fouling")

        # Mark regulatory scenario points
        for m_s, name_s, col_s in [
            (opt_m, f"Optimal (Dec 2026)", "#8E44AD"),
            (DFDS_LIMIT_M, "DFDS practice (Apr 2027)", "#E67E22"),
            (SOLAS_LIMIT_M, "SOLAS limit (Apr 2028)", "#C0392B")]:
            if m_s <= horizon:
                idx = m_s - 1
                fig_opt.add_trace(go.Scatter(
                    x=[m_s], y=[costs[idx]/1000],
                    mode="markers+text",
                    marker=dict(size=12, color=col_s),
                    text=[name_s],
                    textposition="top center",
                    name=name_s,
                    showlegend=True))

        fig_opt.update_layout(
            title=f"Optimal Drydocking Window — Month {opt_m} ({opt_date_custom})<br>"
                  f"<sub>Fuel: ${fuel_price}/ton | Drydock: ${total_dd_custom:,} | "
                  f"Minimum cost: ${opt_c:,.0f}</sub>",
            xaxis_title="Months from Now Until Drydocking",
            yaxis_title="Total Lifecycle Cost (k USD)",
            height=420, template="plotly_white",
            legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_opt, use_container_width=True)

    with col_d2:
        st.markdown("### 📋 Recommendation")
        st.success(
            f"**Optimal drydocking:** {opt_date_custom}\n\n"
            f"**Months from now:** {opt_m}\n\n"
            f"**Expected fouling:** {opt_fi:.1f}%\n\n"
            f"**Minimum cost:** ${opt_c:,.0f}")

        st.markdown("### 💰 Scenario Costs")
        for m_s, name_s in [
                (opt_m, f"★ Optimal (Month {opt_m})"),
                (DFDS_LIMIT_M, "DFDS 24m practice"),
                (SOLAS_LIMIT_M, "SOLAS 36m limit")]:
            if m_s <= horizon:
                c = costs[m_s-1]
                delta = c - opt_c
                col_x, col_y = st.columns([2,1])
                with col_x:
                    st.caption(f"{name_s}")
                with col_y:
                    st.caption(f"${c/1000:.0f}k")

        st.divider()
        st.markdown("### ⚠️ Thresholds")
        month_10 = (10.0 - latest_fi) / FOULING_RATE
        month_20 = (20.0 - latest_fi) / FOULING_RATE
        if month_10 > 0:
            dd_10 = (STUDY_END + pd.DateOffset(
                months=int(month_10))).strftime("%b %Y")
            st.warning(f"10% fouling: ~{dd_10}")
        if month_20 > 0:
            dd_20 = (STUDY_END + pd.DateOffset(
                months=int(month_20))).strftime("%b %Y")
            st.error(f"20% fouling: ~{dd_20}")
        st.divider()
        st.info(f"SOLAS Reg I/7: Max 36m\nbetween DDs\nHard limit: Apr 2028\n(Month {SOLAS_LIMIT_M})")

# ── TAB 5: Data Summary ───────────────────────────────────────
with tab5:
    st.subheader("Data Summary — M/V Kattegat Operational Dataset")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown("### Dataset Overview")
        stats = {
            "Total voyages":       "5,704",
            "Date range":          "Jan 2024 – May 2026",
            "Pre-drydock days":    f"{len(pre)}",
            "Post-drydock days":   f"{len(post)}",
            "Speed coverage":      "100% (fixed)",
            "Weather correction":  "10.4% avg",
            "Clean hull baseline": f"{REAL_BASELINE} kg/nm",
            "Fouling rate":        f"{FOULING_RATE}%/month",
        }
        for k, v in stats.items():
            col_ka, col_va = st.columns([2,1])
            with col_ka: st.caption(k)
            with col_va: st.caption(f"**{v}**")

        st.divider()
        st.markdown("### Vessel Specifications")
        specs = {
            "Name":         "M/V Kattegat",
            "IMO":          "9112765",
            "Operator":     "DFDS",
            "Built":        "1996",
            "Type":         "Ro-Pax ferry",
            "LOA":          "136.40 m",
            "Beam":         "24.60 m",
            "GT":           "14,379",
            "DWT":          "4,030 t",
            "Main engines": "2 × B&W 9L35MC",
            "Total MCR":    "11,700 kW",
            "Service speed":"19 knots",
            "Route":        "Algeciras–Tanger Med",
        }
        for k, v in specs.items():
            col_ks, col_vs = st.columns([2,1])
            with col_ks: st.caption(k)
            with col_vs: st.caption(f"**{v}**")

    with col_e2:
        st.markdown("### Pre vs Post Drydock Comparison")
        compare = pd.DataFrame({
            "Metric": [
                "Avg HFO/day (kg)",
                "Avg HFO/nm (kg/nm)",
                "Avg CO₂/day (kg)",
                "Avg voyages/day",
                "Avg distance/day (nm)",
                "Fouling index (%)",
            ],
            "Pre-Drydock": [
                f"{pre['me_kg'].mean():,.0f}",
                f"{pre['hfo_per_nm'].mean():.2f}",
                f"{pre['co2_total_kg'].mean():,.0f}",
                f"{pre['num_voyages'].mean():.1f}",
                f"{pre['distance_nm'].mean():.1f}",
                f"{pre['hm_fouling_pct'].mean():.1f}%",
            ],
            "Post-Drydock": [
                f"{post['me_kg'].mean():,.0f}",
                f"{post['hfo_per_nm'].mean():.2f}",
                f"{post['co2_total_kg'].mean():,.0f}",
                f"{post['num_voyages'].mean():.1f}",
                f"{post['distance_nm'].mean():.1f}",
                f"{post['hm_fouling_pct'].mean():.1f}%",
            ]
        })
        st.dataframe(compare, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### Drydock Event")
        st.info(
            "**Drydock:** March 4 – April 3, 2025\n\n"
            "**Location:** Gibraltar\n\n"
            "**First commercial voyage:** April 4, 2025\n\n"
            "**Hull improvement:** 8.4% fuel efficiency gain\n\n"
            "**CII improvement:** Pre 19.1 → Post 17.5 g CO₂/GT·nm"
        )

        st.divider()
        st.markdown("### Raw Data Preview")
        cols_show = ["date","period","me_kg","hfo_per_nm",
                     "hm_fouling_pct","daily_cii","num_voyages"]
        available = [c for c in cols_show if c in daily_op.columns]
        st.dataframe(
            daily_op[available].tail(20).sort_values(
                "date", ascending=False),
            use_container_width=True, hide_index=True)

# =============================================================
# FOOTER
# =============================================================
st.divider()
st.caption(
    "**Kattegat Digital Twin** | "
    "Master Thesis in Marine Engineering | "
    "Physics-Informed ML Framework | "
    "Holtrop-Mennen + Extra Trees + IMO CII | "
    f"Data: Jan 2024 – May 2026 | "
    f"Last updated: {latest_date.date()}"
)
