"""
=============================================================
  KATTEGAT DIGITAL TWIN — DAY 4 SCRIPT
  CII Monthly Calculation + 6-Month Forecast
  Vessel : M/V Kattegat (IMO 9112765) — DFDS
=============================================================

WHAT THIS SCRIPT DOES:
-----------------------
  Module 1 — Monthly CII Calculation:
    Calculates the actual CII for every month Jan 2024 → May 2026
    using weather-corrected fuel consumption data
    Assigns IMO rating A/B/C/D/E per month
    Compares pre vs post drydock CII performance

  Module 2 — Annual CII Compliance:
    Calculates annual CII for 2024 and 2025
    Compares against IMO required values
    Determines if Kattegat is compliant

  Module 3 — 6-Month CII Forecast:
    Uses ML fouling forecast from Day 3 to project
    future fuel consumption and CII trajectory
    Predicts which rating class by end of 2026

IMO CII FORMULA (MEPC.352(78) + MEPC.353(78) + MEPC.354(78)):
---------------------------------
  For Ro-Pax vessels:
    CII = CO2_total(g) / (GT × Distance_nm)
    CO2 = FOC_HFO × 3114 + FOC_MGO × 3206  [g CO2]

  Reference line: CII_ref = a × GT^(-c)
    a = 2023   (Ro-ro passenger — MEPC.353(78) Table 1)
    c = 0.460  (Ro-ro passenger — MEPC.353(78) Table 1)

  Annual reduction factors:
    2023: 5%  | 2024: 7%  | 2025: 9%
    2026: 11% | 2027: 13% | 2028: 15%

  Rating boundaries (MEPC.338(76)):
    A: < 0.86 × CII_req
    B: 0.86–0.94 × CII_req
    C: 0.94–1.06 × CII_req  ← minimum compliance
    D: 1.06–1.18 × CII_req
    E: > 1.18 × CII_req

HOW TO USE:
-----------
  1. Make sure kattegat_daily_fixed.csv is in your folder
  2. Change DATA_FOLDER below
  3. Terminal: python3 kattegat_day4.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# =============================================================
# !! CHANGE THIS TO YOUR FOLDER PATH !!
# =============================================================
DATA_FOLDER  = "/Users/mbp/Desktop/Kattegat_Data/"
DAILY_CSV    = os.path.join(DATA_FOLDER, "kattegat_daily_fixed.csv")
FORECAST_CSV = os.path.join(DATA_FOLDER, "kattegat_day3_forecast.csv")

# =============================================================
# KATTEGAT CONFIRMED SPECS
# =============================================================
GT  = 14379    # Gross Tonnage — confirmed
DWT = 4030     # DWT — confirmed

# CO2 emission factors (IMO MEPC.352(78))
HFO_CO2_FACTOR = 3114   # g CO2 per kg HFO
MGO_CO2_FACTOR = 3206   # g CO2 per kg MGO

# IMO CII coefficients for Ro-Pax (MEPC.353(78) Table 1)
# Source: 2022 Guidelines on CII Reference Lines (G2)
# Ship type: Ro-Ro passenger ship
# Capacity: GT (Gross Tonnage) — NOT DWT for this ship type
CII_A = 2023       # MEPC.353(78) Table 1 — Ro-ro passenger ship
CII_C = 0.460      # MEPC.353(78) Table 1 — Ro-ro passenger ship

# Rating boundaries (MEPC.354(78)) — Ro-ro passenger ship specific
D1, D2, D3, D4 = 0.76, 0.92, 1.14, 1.30

# Annual reduction factors
REDUCTION = {
    2023: 0.05, 2024: 0.07, 2025: 0.09,
    2026: 0.11, 2027: 0.13, 2028: 0.15
}

# Drydock dates
DRYDOCK_START = pd.to_datetime("2025-03-04")
DRYDOCK_END   = pd.to_datetime("2025-04-04")

# =============================================================
# HELPER FUNCTIONS
# =============================================================
def cii_required(year):
    """Required CII for a given year after reduction factor."""
    cii_ref = CII_A * (GT ** (-CII_C))
    red     = REDUCTION.get(year, 0.09)
    return cii_ref * (1 - red)

def assign_rating(cii_value, year):
    """
    Assigns IMO CII rating A–E based on value and year.
    Returns rating letter and numeric score.
    """
    if pd.isna(cii_value):
        return "N/A", np.nan
    req = cii_required(year)
    if   cii_value < D1 * req: return "A", 1
    elif cii_value < D2 * req: return "B", 2
    elif cii_value < D3 * req: return "C", 3
    elif cii_value < D4 * req: return "D", 4
    else:                      return "E", 5

# Rating colors
RATING_COLORS = {
    "A": "#27AE60", "B": "#2ECC71",
    "C": "#F39C12", "D": "#E67E22",
    "E": "#E74C3C", "N/A": "#95A5A6"
}

# =============================================================
# STEP 1 — LOAD DATA
# =============================================================
print("=" * 60)
print("  KATTEGAT DIGITAL TWIN — DAY 4: CII MODULE")
print("=" * 60)

daily = pd.read_csv(DAILY_CSV, parse_dates=["date"])
daily = daily[daily["period"] != "Drydock"].copy()
daily = daily.sort_values("date").reset_index(drop=True)

print(f"\n✓ Loaded {len(daily)} daily records")
print(f"  Date range : {daily['date'].min().date()} → "
      f"{daily['date'].max().date()}")

# Use weather-corrected fuel where available
daily["fuel_for_cii"] = np.where(
    daily["me_kg_corrected"].notna(),
    daily["me_kg_corrected"],
    daily["me_kg"]
)

# Recalculate CO2 using weather-corrected fuel
# ME = HFO, AUX + Boiler = MGO
daily["co2_g"] = (
    daily["fuel_for_cii"] * HFO_CO2_FACTOR +
    (daily["aux_kg"] + daily["boiler_kg"]) * MGO_CO2_FACTOR
)

# Daily CII (for reference)
daily["cii_daily"] = daily["co2_g"] / (GT * daily["distance_nm"])

# =============================================================
# STEP 2 — MONTHLY CII CALCULATION
# =============================================================
print("\n" + "=" * 60)
print("  MODULE 1: MONTHLY CII CALCULATION")
print("=" * 60)

daily["year"]  = daily["date"].dt.year
daily["month"] = daily["date"].dt.month
daily["year_month"] = daily["date"].dt.to_period("M")

# Aggregate by month
monthly = daily.groupby("year_month").agg(
    co2_g_total    = ("co2_g",         "sum"),
    distance_total = ("distance_nm",   "sum"),
    me_kg_total    = ("fuel_for_cii",  "sum"),
    num_days       = ("date",          "count"),
    period         = ("period",        "first"),
    year           = ("year",          "first"),
).reset_index()

# Calculate monthly CII
monthly["cii_monthly"] = (
    monthly["co2_g_total"] / (GT * monthly["distance_total"])
)

# Assign rating
monthly["rating"] = monthly.apply(
    lambda r: assign_rating(r["cii_monthly"], r["year"])[0], axis=1)
monthly["rating_num"] = monthly.apply(
    lambda r: assign_rating(r["cii_monthly"], r["year"])[1], axis=1)
monthly["cii_required"] = monthly["year"].apply(cii_required)

# Performance ratio vs required
monthly["perf_ratio"] = (
    monthly["cii_monthly"] / monthly["cii_required"]
)

# Date for plotting
monthly["date"] = monthly["year_month"].dt.to_timestamp()

print(f"\n  {'Month':<10} {'CII':>10} {'Required':>10} "
      f"{'Ratio':>8} {'Rating':>8} {'Period'}")
print("  " + "-" * 62)
for _, row in monthly.iterrows():
    marker = " ← drydock" if str(row["year_month"]) in ["2025-03","2025-04"] else ""
    print(f"  {str(row['year_month']):<10} "
          f"{row['cii_monthly']:>10.6f} "
          f"{row['cii_required']:>10.6f} "
          f"{row['perf_ratio']:>8.3f} "
          f"{row['rating']:>8} "
          f"{row['period']}{marker}")

# =============================================================
# STEP 3 — ANNUAL CII COMPLIANCE
# =============================================================
print("\n" + "=" * 60)
print("  MODULE 2: ANNUAL CII COMPLIANCE")
print("=" * 60)

annual = daily.groupby("year").agg(
    co2_g_total    = ("co2_g",         "sum"),
    distance_total = ("distance_nm",   "sum"),
    num_days       = ("date",          "count"),
).reset_index()

annual["cii_annual"]   = (
    annual["co2_g_total"] / (GT * annual["distance_total"])
)
annual["cii_required"] = annual["year"].apply(cii_required)
annual["rating"]       = annual.apply(
    lambda r: assign_rating(r["cii_annual"], r["year"])[0], axis=1)
annual["compliant"]    = annual["rating"].isin(["A","B","C"])
annual["perf_ratio"]   = annual["cii_annual"] / annual["cii_required"]

print(f"\n  Annual CII Results:")
print(f"\n  {'Year':<6} {'Annual CII':>12} {'Required':>12} "
      f"{'Rating':>8} {'Compliant':>10} {'Days':>6}")
print("  " + "-" * 60)
for _, row in annual.iterrows():
    comp_str = "✓ YES" if row["compliant"] else "✗ NO"
    print(f"  {int(row['year']):<6} "
          f"{row['cii_annual']:>12.6f} "
          f"{row['cii_required']:>12.6f} "
          f"{row['rating']:>8} "
          f"{comp_str:>10} "
          f"{int(row['num_days']):>6}")

# Pre vs post drydock comparison
pre_monthly  = monthly[monthly["period"] == "Pre-Drydock"]
post_monthly = monthly[monthly["period"] == "Post-Drydock"]

print(f"\n  Pre-drydock avg CII   : {pre_monthly['cii_monthly'].mean():.6f}")
print(f"  Post-drydock avg CII  : {post_monthly['cii_monthly'].mean():.6f}")
pct = ((pre_monthly['cii_monthly'].mean() -
        post_monthly['cii_monthly'].mean()) /
       pre_monthly['cii_monthly'].mean() * 100)
print(f"  CII improvement       : {pct:.1f}%")

# =============================================================
# STEP 4 — 6-MONTH CII FORECAST
# =============================================================
print("\n" + "=" * 60)
print("  MODULE 3: 6-MONTH CII FORECAST")
print("=" * 60)

# Load ML fouling forecast from Day 3
forecast_df = None
if os.path.exists(FORECAST_CSV):
    forecast_df = pd.read_csv(FORECAST_CSV, parse_dates=["date"])
    print(f"\n  ✓ ML forecast loaded: {len(forecast_df)} days")
else:
    print("\n  ⚠ No ML forecast file found — using trend extrapolation")

# Get last known values
last_date   = daily["date"].max()
last_cii    = daily["cii_daily"].rolling(30, min_periods=10).mean().iloc[-1]
last_dist   = daily["distance_nm"].mean()
last_me_kg  = daily["fuel_for_cii"].mean()
last_aux_kg = daily["aux_kg"].mean()
last_boi_kg = daily["boiler_kg"].mean()
BASELINE_KGNM = 58.2  # confirmed clean hull baseline

print(f"  Last date             : {last_date.date()}")
print(f"  Last 30-day avg CII   : {last_cii:.6f}")
print(f"  Avg distance/day      : {last_dist:.1f} nm")

# Build 6-month daily forecast
forecast_dates = pd.date_range(
    last_date + pd.Timedelta(days=1),
    periods=180, freq="D"
)
fc = pd.DataFrame({"date": forecast_dates})
fc["year"]  = fc["date"].dt.year
fc["month"] = fc["date"].dt.month

# If ML forecast available — use fouling index to estimate fuel
if forecast_df is not None:
    fc = pd.merge(
        fc,
        forecast_df[["date","ml_forecast",
                     "forecast_upper","forecast_lower"]],
        on="date", how="left"
    )
    fc["ml_forecast"] = fc["ml_forecast"].interpolate()

    # Convert fouling index to fuel consumption
    # me_kg = baseline × (1 + fouling_index/100) × avg_distance
    fc["me_kg_forecast"] = (
        BASELINE_KGNM * (1 + fc["ml_forecast"]/100)
        * last_dist
    )
    fc["me_kg_upper"] = (
        BASELINE_KGNM * (1 + fc["forecast_upper"]/100)
        * last_dist
    )
    fc["me_kg_lower"] = (
        BASELINE_KGNM * (1 + fc["forecast_lower"]/100)
        * last_dist
    )
else:
    # Simple trend extrapolation
    trend = daily["cii_daily"].diff(30).mean() / 30
    for i, row in fc.iterrows():
        days_ahead = (row["date"] - last_date).days
        fc.loc[i, "me_kg_forecast"] = last_me_kg * (1 + trend * days_ahead)
    fc["me_kg_upper"] = fc["me_kg_forecast"] * 1.1
    fc["me_kg_lower"] = fc["me_kg_forecast"] * 0.9

# Calculate forecast CII
fc["co2_g_forecast"] = (
    fc["me_kg_forecast"] * HFO_CO2_FACTOR +
    (last_aux_kg + last_boi_kg) * MGO_CO2_FACTOR
)
fc["co2_g_upper"] = (
    fc["me_kg_upper"] * HFO_CO2_FACTOR +
    (last_aux_kg + last_boi_kg) * MGO_CO2_FACTOR
)
fc["co2_g_lower"] = (
    fc["me_kg_lower"] * HFO_CO2_FACTOR +
    (last_aux_kg + last_boi_kg) * MGO_CO2_FACTOR
)
fc["cii_forecast"] = fc["co2_g_forecast"] / (GT * last_dist)
fc["cii_upper"]    = fc["co2_g_upper"]    / (GT * last_dist)
fc["cii_lower"]    = fc["co2_g_lower"]    / (GT * last_dist)

fc["rating"] = fc.apply(
    lambda r: assign_rating(r["cii_forecast"], r["year"])[0], axis=1)

# Monthly forecast summary
fc["year_month"] = fc["date"].dt.to_period("M")
fc_monthly = fc.groupby("year_month").agg(
    cii_forecast = ("cii_forecast", "mean"),
    cii_upper    = ("cii_upper",    "mean"),
    cii_lower    = ("cii_lower",    "mean"),
    year         = ("year",         "first"),
    rating       = ("rating",       "first"),
).reset_index()
fc_monthly["cii_required"] = fc_monthly["year"].apply(cii_required)
fc_monthly["date"] = fc_monthly["year_month"].dt.to_timestamp()

print(f"\n  6-Month CII Forecast:")
print(f"\n  {'Month':<10} {'CII Forecast':>13} "
      f"{'Required':>10} {'Rating':>8} {'Compliant':>10}")
print("  " + "-" * 58)
for _, row in fc_monthly.iterrows():
    comp = "✓" if row["rating"] in ["A","B","C"] else "✗"
    print(f"  {str(row['year_month']):<10} "
          f"{row['cii_forecast']:>13.6f} "
          f"{row['cii_required']:>10.6f} "
          f"{row['rating']:>8} "
          f"{comp:>10}")

# =============================================================
# STEP 5 — 4 THESIS PLOTS
# =============================================================
fig, axes = plt.subplots(4, 1, figsize=(16, 26))
fig.suptitle(
    "M/V Kattegat — IMO CII Analysis & Forecast\n"
    "Monthly Carbon Intensity Indicator | Jan 2024 – Nov 2026",
    fontsize=15, fontweight="bold", y=1.005)

C_PRE="#C0392B"; C_PST="#27AE60"; C_AVG="#2C3E50"
C_DD1="#E67E22"; C_DD2="#27AE60"; C_FC="#8E44AD"

def shade(ax):
    ax.axvspan(DRYDOCK_START, DRYDOCK_END,
               alpha=0.15, color="#95A5A6", label="Drydock period")
    ax.axvline(DRYDOCK_START, c=C_DD1, lw=1.5, ls="--",
               label="Drydock start")
    ax.axvline(DRYDOCK_END,   c=C_DD2, lw=1.5, ls="--",
               label="First voyage")

# ── Plot 1: Monthly CII with rating bands ────────────────────
ax = axes[0]

# Draw rating bands
for year in [2024, 2025, 2026]:
    req = cii_required(year)
    start = pd.to_datetime(f"{year}-01-01")
    end   = pd.to_datetime(f"{year}-12-31")
    ax.axhspan(0,          D1*req, xmin=0, xmax=1,
               alpha=0.04, color="#27AE60")
    ax.axhspan(D1*req,     D2*req, alpha=0.04, color="#2ECC71")
    ax.axhspan(D2*req,     D3*req, alpha=0.06, color="#F39C12")
    ax.axhspan(D3*req,     D4*req, alpha=0.08, color="#E67E22")
    ax.axhspan(D4*req,     0.020,  alpha=0.08, color="#E74C3C")

# Plot monthly CII bars
for _, row in monthly.iterrows():
    color = RATING_COLORS.get(row["rating"], "#95A5A6")
    ax.bar(row["date"], row["cii_monthly"],
           width=20, color=color, alpha=0.85, edgecolor="white")

# Required line
for year in [2024, 2025, 2026]:
    start = pd.to_datetime(f"{year}-01-01")
    end   = pd.to_datetime(f"{year}-12-31") if year < 2026 else fc["date"].max()
    req   = cii_required(year)
    ax.hlines(req, start, end, colors="#2C3E50", lw=2, ls="--")

# Legend patches
patches = [mpatches.Patch(color=RATING_COLORS[r], label=f"Rating {r}")
           for r in ["A","B","C","D","E"]]
patches.append(mpatches.Patch(color="#2C3E50", label="Required CII"))
shade(ax)
ax.legend(handles=patches + [
    mpatches.Patch(color="#95A5A6", alpha=0.5, label="Drydock")],
    fontsize=8, ncol=4, loc="upper right")
ax.set_ylabel("Monthly CII (g CO₂/GT·nm)", fontsize=11)
ax.set_title("① Monthly CII Rating — Actual Values (Jan 2024 – May 2026)\n"
             "Colour = IMO Rating  |  Dashed line = Annual required CII",
             fontsize=11, fontweight="bold", pad=8)
ax.grid(True, alpha=0.3, axis="y")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# ── Plot 2: CII performance ratio timeline ────────────────────
ax = axes[1]
pre_m  = monthly[monthly["period"]=="Pre-Drydock"]
post_m = monthly[monthly["period"]=="Post-Drydock"]

ax.scatter(pre_m["date"],  pre_m["perf_ratio"],
           c=C_PRE, s=60, zorder=5, label="Pre-drydock months")
ax.scatter(post_m["date"], post_m["perf_ratio"],
           c=C_PST, s=60, zorder=5, label="Post-drydock months")
ax.plot(monthly["date"],
        monthly["perf_ratio"].rolling(3, center=True, min_periods=1).mean(),
        c=C_AVG, lw=2, label="3-month rolling avg")

# Draw rating bands as horizontal lines
for label, val, color in [
    ("A/B boundary", D1, "#27AE60"),
    ("B/C boundary (min compliance)", D2, "#F39C12"),
    ("C/D boundary", D3, "#E67E22"),
    ("D/E boundary", D4, "#E74C3C"),
]:
    ax.axhline(val, c=color, lw=1.2, ls=":", label=label)

ax.axhline(1.0, c="black", lw=1.5, ls="-", label="Required = 1.0")
shade(ax)
ax.set_ylabel("CII Performance Ratio\n(actual / required)", fontsize=11)
ax.set_title(
    "② CII Performance Ratio vs Annual Requirement\n"
    "< 1.0 = better than required  |  > 1.0 = worse than required",
    fontsize=11, fontweight="bold", pad=8)
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# ── Plot 3: Annual CII comparison ────────────────────────────
ax = axes[2]
years_plot = annual["year"].tolist()
x = np.arange(len(years_plot))
w = 0.35
req_vals  = [cii_required(y) for y in years_plot]
act_vals  = annual["cii_annual"].tolist()
ratings   = annual["rating"].tolist()
colors_bar= [RATING_COLORS.get(r,"#95A5A6") for r in ratings]

bars1 = ax.bar(x - w/2, act_vals, w, color=colors_bar,
               alpha=0.85, edgecolor="white", label="Actual annual CII")
bars2 = ax.bar(x + w/2, req_vals, w, color="#2C3E50",
               alpha=0.6, edgecolor="white", label="Required annual CII")

for bar, rating, val in zip(bars1, ratings, act_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0001,
            f"Rating {rating}", ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            color=RATING_COLORS.get(rating,"#95A5A6"))
for bar, val in zip(bars2, req_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0001,
            f"{val:.5f}", ha="center", va="bottom",
            fontsize=8, color="#2C3E50")

ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in years_plot], fontsize=12)
ax.set_ylabel("Annual CII (g CO₂/GT·nm)", fontsize=11)
ax.set_title(
    "③ Annual CII Compliance — Actual vs Required\n"
    "IMO MEPC.354(78) — Reduction factors: 2024=7%, 2025=9%",
    fontsize=11, fontweight="bold", pad=8)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")

# ── Plot 4: 6-month CII forecast ─────────────────────────────
ax = axes[3]

# Historical last 6 months
hist_start = last_date - pd.Timedelta(days=180)
hist = daily[daily["date"] >= hist_start].copy()
hist_monthly = hist.groupby(hist["date"].dt.to_period("M")).agg(
    cii=("cii_daily","mean"), year=("year","first")).reset_index()
hist_monthly = hist_monthly.rename(columns={"date": "year_month"})
hist_monthly["date"] = hist_monthly["year_month"].dt.to_timestamp()
hist_monthly["rating"] = hist_monthly.apply(
    lambda r: assign_rating(r["cii"], r["year"])[0], axis=1)

for _, row in hist_monthly.iterrows():
    ax.bar(row["date"], row["cii"], width=20,
           color=RATING_COLORS.get(row["rating"],"#95A5A6"),
           alpha=0.7, edgecolor="white")

# Forecast bars
for _, row in fc_monthly.iterrows():
    ax.bar(row["date"], row["cii_forecast"], width=20,
           color=RATING_COLORS.get(row["rating"],"#95A5A6"),
           alpha=0.45, edgecolor=C_FC, linewidth=1.5)

# Confidence band
ax.fill_between(fc["date"],
                fc["cii_lower"], fc["cii_upper"],
                alpha=0.15, color=C_FC, label="95% confidence interval")
ax.plot(fc["date"], fc["cii_forecast"],
        c=C_FC, lw=2, ls="--", label="CII forecast")

# Required lines for 2025 and 2026
for year, ls in [(2025,"-"),(2026,"--")]:
    req = cii_required(year)
    s   = pd.to_datetime(f"{year}-01-01")
    e   = pd.to_datetime(f"{year}-12-31")
    ax.hlines(req, max(s, hist_start), min(e, fc["date"].max()),
              colors="#2C3E50", lw=1.5, ls=ls,
              label=f"Required {year} ({req:.5f})")

ax.axvline(last_date, c="black", lw=2, ls=":",
           label=f"Forecast start ({last_date.date()})")

# Rating annotations on forecast
for _, row in fc_monthly.iterrows():
    ax.text(row["date"], row["cii_forecast"]+0.0001,
            row["rating"], ha="center", va="bottom",
            fontsize=9, fontweight="bold",
            color=RATING_COLORS.get(row["rating"],"#95A5A6"))

ax.set_ylabel("CII (g CO₂/GT·nm)", fontsize=11)
ax.set_xlabel("Date", fontsize=11)
ax.set_title(
    "④ 6-Month CII Forecast with Rating Projection\n"
    "Solid bars = historical  |  Transparent bars = forecast",
    fontsize=11, fontweight="bold", pad=8)
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3, axis="y")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

for ax in axes:
    ax.tick_params(axis="x", rotation=15)

plt.tight_layout(h_pad=4.0)
plot_path = os.path.join(DATA_FOLDER, "kattegat_day4_cii.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\n✓ Plot saved : {plot_path}")
plt.show()

# =============================================================
# STEP 6 — SAVE CII CSV
# =============================================================
monthly_out = monthly[[
    "year_month","date","period","year",
    "cii_monthly","cii_required","perf_ratio","rating",
    "co2_g_total","distance_total","me_kg_total","num_days"
]]
monthly_path = os.path.join(DATA_FOLDER, "kattegat_cii_monthly.csv")
monthly_out.to_csv(monthly_path, index=False)

forecast_out = fc_monthly[[
    "year_month","date","year","cii_forecast",
    "cii_upper","cii_lower","cii_required","rating"
]]
forecast_path = os.path.join(DATA_FOLDER, "kattegat_cii_forecast.csv")
forecast_out.to_csv(forecast_path, index=False)

print(f"✓ Monthly CII CSV : {monthly_path}")
print(f"✓ Forecast CSV    : {forecast_path}")

# =============================================================
# STEP 7 — KEY THESIS NUMBERS
# =============================================================
print("\n" + "=" * 60)
print("  KEY THESIS NUMBERS — CHAPTER 5")
print("=" * 60)
print(f"\n  IMO CII Reference (Ro-Pax, GT={GT}):")
cii_ref = CII_A * (GT ** (-CII_C))
print(f"    CII_ref        : {cii_ref:.6f} g CO₂/GT·nm")
for y in [2024, 2025, 2026]:
    print(f"    Required {y}  : {cii_required(y):.6f} "
          f"(reduction {REDUCTION.get(y,0)*100:.0f}%)")

print(f"\n  Monthly CII Results:")
print(f"    Pre-drydock avg  : {pre_m['cii_monthly'].mean():.6f}")
print(f"    Post-drydock avg : {post_m['cii_monthly'].mean():.6f}")
print(f"    CII improvement  : {pct:.1f}%")

print(f"\n  Annual CII Compliance:")
for _, row in annual.iterrows():
    comp = "COMPLIANT" if row["compliant"] else "NON-COMPLIANT"
    print(f"    {int(row['year'])}: Rating {row['rating']} — {comp}")

print(f"\n  6-Month Forecast:")
for _, row in fc_monthly.iterrows():
    comp = "✓" if row["rating"] in ["A","B","C"] else "✗"
    print(f"    {str(row['year_month'])}: Rating {row['rating']} "
          f"CII={row['cii_forecast']:.6f} {comp}")

print("\n" + "=" * 60)
print("  DAY 4 COMPLETE ✓")
print("=" * 60)
print("""
Files saved:
  kattegat_day4_cii.png        → 4 thesis plots
  kattegat_cii_monthly.csv     → monthly CII ratings
  kattegat_cii_forecast.csv    → 6-month forecast

Next:
  Day 5 → Drydocking optimizer
  Day 6 → Streamlit dashboard
""")
