"""
=============================================================
  KATTEGAT DIGITAL TWIN — DAY 1 SCRIPT v5 (FINAL)
  Vessel  : M/V Kattegat (IMO 9112765) — DFDS
  Route   : Algeciras — Tanger Med
  Source  : 2026_mim.xlsx — Voyage-level operational database
=============================================================

DATA SOURCE: 2026 mim.xlsx
----------------------------
  5,843 voyage records — Jan 2024 → May 2026
  One row per crossing (Algeciras ↔ Tanger Med)
  Key columns used:
    Departure              → voyage date + time
    Total ME Consumption   → Main Engine fuel (tons/voyage)
    Total AUX Consumption  → Auxiliary fuel (tons/voyage)
    Total Boiler Consumption → Boiler fuel (tons/voyage)
    Total Distance         → Actual distance sailed (nm)
    Kg/Nm - ME             → ME fuel efficiency (already computed)
    Kg/Nm - ME Aux & Boiler → Total efficiency (already computed)
    Average sea passage speed → Speed (knots) — available for 35% rows

FILTERING APPLIED:
-------------------
  Keep  : Voyage (DFDS to DFDS) type only
  Keep  : Algeciras ↔ Tanger Med crossings only
  Remove: Shifting In a Same Port (dummy/port moves)
  Remove: Positioning, Laid Up entries
  Remove: ME consumption < 0.5 tons (incomplete rows)
  Remove: Distance < 10 nm (not real crossings)
  Remove: Drydock period Mar 4 → Apr 3, 2025

CONFIRMED CONSTANTS (from daily engineering reports):
------------------------------------------------------
  Engine RPM   : 155 (both PS and STB — constant)
  Engine Load  : 56–59% → avg 57.5%
  Shaft Power  : 11,700 kW × 57.5% = 6,727 kW
  These are hardcoded in the physics model (Day 2)

KEY EVENTS:
-----------
  Lay-up 1  : ~Apr 15–26, 2024  → Regular maintenance (hull NOT cleaned)
  DRYDOCK   : Mar 04 → Apr 03, 2025 → Hull cleaned + coating renewed
  Lay-up 2  : ~Apr 28–29, 2026  → Regular maintenance (hull NOT cleaned)

HOW TO USE:
-----------
  1. Put 2026_mim.xlsx in your folder
  2. Change FILE_PATH below
  3. Terminal: python3 kattegat_day1.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# =============================================================
# !! CHANGE THIS TO YOUR FILE PATH !!
# =============================================================
FILE_PATH = "/Users/mbp/Desktop/Kattegat_Data/2026_mim.xlsx"

# ── CONFIRMED DATES ───────────────────────────────────────────
DRYDOCK_START = pd.to_datetime("2025-03-04")
DRYDOCK_END   = pd.to_datetime("2025-04-04")

# ── KATTEGAT CONFIRMED SPECS ─────────────────────────────────
DWT          = 4030     # tons
MCR_KW       = 11700    # kW — 2 × B&W 9L35MC
ENGINE_RPM   = 155      # constant — confirmed
ENGINE_LOAD  = 57.5     # % — avg of 56–59 range confirmed
SHAFT_KW     = MCR_KW * ENGINE_LOAD / 100   # 6,727 kW

# ── CO2 FACTORS — IMO MEPC.354(78) ───────────────────────────
HFO_CO2 = 3.114    # kg CO2 / kg HFO (ME uses HFO)
MGO_CO2 = 3.206    # kg CO2 / kg MGO (AUX + Boiler use MGO)

# =============================================================
# STEP 1 — LOAD AND FILTER DATA
# =============================================================
print("=" * 60)
print("  KATTEGAT DIGITAL TWIN — DAY 1 DATA LOADER")
print("=" * 60)

print(f"\nLoading: {FILE_PATH}")
df = pd.read_excel(FILE_PATH)
print(f"✓ Raw records loaded : {len(df)}")

# ── Filter 1: Keep only real commercial crossings ─────────────
df = df[df["Voyage Report Type Name"] == "Voyage (DFDS to DFDS)"]

# ── Filter 2: Keep only Algeciras ↔ Tanger Med ───────────────
df = df[
    (df["Departure Port Name"].isin(["Algeciras", "Tanger Med"])) &
    (df["Arrival Port Name"].isin(["Algeciras", "Tanger Med"]))   &
    (df["Departure Port Name"] != df["Arrival Port Name"])
]

# ── Filter 3: Remove dummy/incomplete rows ────────────────────
df = df[df["Total ME Consumption"] >= 0.5]
df = df[df["Total Distance"] >= 10]

print(f"✓ After filtering     : {len(df)} real voyages")

# ── Parse dates ───────────────────────────────────────────────
df["Departure"] = pd.to_datetime(df["Departure"])
df["Arrival"]   = pd.to_datetime(df["Arrival"])
df = df.sort_values("Departure").reset_index(drop=True)

print(f"✓ Date range          : {df['Departure'].min().date()} → "
      f"{df['Departure'].max().date()}")

# =============================================================
# STEP 2 — RENAME AND COMPUTE COLUMNS
# =============================================================
df = df.rename(columns={
    "Departure":                   "dep_dt",
    "Arrival":                     "arr_dt",
    "Departure Port Name":         "from_port",
    "Arrival Port Name":           "to_port",
    "Total ME Consumption":        "me_tons",
    "Total AUX Consumption":       "aux_tons",
    "Total Boiler Consumption":    "boiler_tons",
    "Total Distance":              "distance_nm",
    "Kg/Nm - ME":                  "kgnm_me",
    "Kg/Nm - ME, Aux & Boiler":   "kgnm_total",
    "Average sea passage speed (knots)": "speed_kn",
    "Time at sea (hours)":         "time_at_sea_hr",
})

# Convert ME from tons to kg (1 ton = 1000 kg)
df["me_kg"]      = df["me_tons"]     * 1000
df["aux_kg"]     = df["aux_tons"]    * 1000
df["boiler_kg"]  = df["boiler_tons"] * 1000
df["total_fuel_kg"] = df["me_kg"] + df["aux_kg"] + df["boiler_kg"]

# CO2 per voyage
# ME uses HFO, AUX + Boiler use MGO
df["co2_me_kg"]    = df["me_kg"]     * HFO_CO2
df["co2_aux_kg"]   = (df["aux_kg"] + df["boiler_kg"]) * MGO_CO2
df["co2_total_kg"] = df["co2_me_kg"] + df["co2_aux_kg"]

# Date column (day only)
df["date"] = df["dep_dt"].dt.normalize()

# Filter outlier speed values (realistic range 14–25 kn)
df["speed_kn"] = np.where(
    (df["speed_kn"] < 14) | (df["speed_kn"] > 25),
    np.nan, df["speed_kn"]
)

# Filter outlier kgnm values (realistic 40–120 kg/nm)
df["kgnm_me"] = np.where(
    (df["kgnm_me"] < 40) | (df["kgnm_me"] > 120),
    np.nan, df["kgnm_me"]
)

# =============================================================
# STEP 3 — ASSIGN PERIODS
# =============================================================
def assign_period(d):
    if d < DRYDOCK_START:  return "Pre-Drydock"
    elif d < DRYDOCK_END:  return "Drydock"
    else:                  return "Post-Drydock"

df["period"] = df["dep_dt"].apply(assign_period)
df["days_since_drydock"] = (df["dep_dt"] - DRYDOCK_END).dt.days

# =============================================================
# STEP 4 — AGGREGATE TO DAILY SUMMARY
# =============================================================
daily = df[df["period"] != "Drydock"].groupby("date").agg(
    me_kg           = ("me_kg",         "sum"),
    aux_kg          = ("aux_kg",        "sum"),
    boiler_kg       = ("boiler_kg",     "sum"),
    total_fuel_kg   = ("total_fuel_kg", "sum"),
    co2_me_kg       = ("co2_me_kg",     "sum"),
    co2_total_kg    = ("co2_total_kg",  "sum"),
    distance_nm     = ("distance_nm",   "sum"),
    num_voyages     = ("me_kg",         "count"),
    avg_speed_kn    = ("speed_kn",      "mean"),
    avg_kgnm_me     = ("kgnm_me",       "mean"),
    period          = ("period",        "first"),
    days_since_drydock = ("days_since_drydock", "mean"),
).reset_index()

# Daily fuel efficiency kg/nm (total ME / total distance)
daily["hfo_per_nm"] = np.where(
    daily["distance_nm"] > 0,
    daily["me_kg"] / daily["distance_nm"],
    np.nan
)

# Daily CII = CO2(g) / (DWT × distance_nm)
daily["daily_cii"] = np.where(
    daily["distance_nm"] > 0,
    (daily["co2_total_kg"] * 1000) / (DWT * daily["distance_nm"]),
    np.nan
)

# 7-day rolling averages
daily = daily.sort_values("date").reset_index(drop=True)
daily["me_7day"]     = daily["me_kg"].rolling(7, center=True, min_periods=3).mean()
daily["kgnm_7day"]   = daily["hfo_per_nm"].rolling(7, center=True, min_periods=3).mean()
daily["speed_7day"]  = daily["avg_speed_kn"].rolling(7, center=True, min_periods=3).mean()

# =============================================================
# STEP 5 — CLEAN HULL BASELINE + FOULING INDEX
# =============================================================
post = daily[daily["period"] == "Post-Drydock"]
pre  = daily[daily["period"] == "Pre-Drydock"]

# Baseline = first 14 days after first commercial voyage (Apr 4 2025)
baseline_end  = DRYDOCK_END + pd.Timedelta(days=14)
baseline_data = post[post["date"] <= baseline_end]
baseline_kgnm = baseline_data["hfo_per_nm"].mean()

daily["fouling_index"] = (
    (daily["hfo_per_nm"] - baseline_kgnm) / baseline_kgnm * 100
)
# Cap extreme outliers
daily["fouling_index"] = np.where(
    daily["fouling_index"].abs() > 80, np.nan,
    daily["fouling_index"]
)

daily["fi_7day"] = daily["fouling_index"].rolling(7, center=True, min_periods=3).mean()

# Redefine pre/post after all columns computed
pre  = daily[daily["period"] == "Pre-Drydock"]
post = daily[daily["period"] == "Post-Drydock"]

# =============================================================
# STEP 6 — SUMMARY STATISTICS
# =============================================================
print("\n" + "=" * 60)
print("  SUMMARY STATISTICS")
print("=" * 60)
print(f"\n  Date range           : {daily['date'].min().date()} → "
      f"{daily['date'].max().date()}")
print(f"  Total operational days: {len(daily)}")

print(f"\n  Periods:")
print(f"    Pre-Drydock        : {len(pre):4d} days  "
      f"(Jan 01 2024 → Mar 03 2025)  Fouled hull")
print(f"    Drydock            :   31 days  "
      f"(Mar 04 → Apr 03 2025)        EXCLUDED")
print(f"    Post-Drydock       : {len(post):4d} days  "
      f"(Apr 04 2025 → present)       Clean hull")

print(f"\n  Lay-ups (gaps in data — hull NOT cleaned):")
print(f"    Lay-up 1           : ~Apr 15–26, 2024")
print(f"    Lay-up 2           : ~Apr 28–29, 2026")

print(f"\n  Confirmed constants:")
print(f"    Engine RPM         : {ENGINE_RPM}")
print(f"    Engine Load        : {ENGINE_LOAD}%  (range 56–59%)")
print(f"    Shaft Power        : {SHAFT_KW:,.0f} kW")

print(f"\n  Clean hull baseline  : {baseline_kgnm:.2f} kg/nm")
print(f"  (First 14 days post-drydock, Apr 4–17 2025, "
      f"{len(baseline_data)} days)")

def print_period(label, sub):
    if len(sub) == 0: return
    print(f"\n  ── {label} ─────────────────────────────────────")
    print(f"  Days of data         : {len(sub):>6d}")
    print(f"  Avg ME/day           : {sub['me_kg'].mean():>8,.0f} kg")
    print(f"  Avg AUX+Boiler/day   : {(sub['aux_kg']+sub['boiler_kg']).mean():>8,.0f} kg")
    print(f"  Avg CO2/day          : {sub['co2_total_kg'].mean():>8,.0f} kg")
    print(f"  Avg distance/day     : {sub['distance_nm'].mean():>8.1f} nm")
    print(f"  Avg voyages/day      : {sub['num_voyages'].mean():>8.1f}")
    print(f"  Avg HFO/nm           : {sub['hfo_per_nm'].mean():>8.2f} kg/nm")
    print(f"  Avg speed            : {sub['avg_speed_kn'].mean():>8.2f} kn")
    print(f"  Avg daily CII        : {sub['daily_cii'].mean():>8.2f}")

print_period("PRE-DRYDOCK  (Fouled Hull — 14 months)", pre)
print_period("POST-DRYDOCK (Clean Hull + Regrowth — 13 months)", post)

if len(pre) > 0 and len(post) > 0:
    pct   = (pre["hfo_per_nm"].mean()-post["hfo_per_nm"].mean())/pre["hfo_per_nm"].mean()*100
    kgday = pre["me_kg"].mean()-post["me_kg"].mean()
    base_pct = (pre["hfo_per_nm"].mean()-baseline_kgnm)/baseline_kgnm*100
    print(f"\n  ✓ Overall efficiency gain post-drydock  : {pct:.1f}%")
    print(f"  ✓ Pre-drydock vs clean baseline          : {base_pct:.1f}% fouling penalty")
    print(f"  ✓ Avg daily ME saving post-drydock       : {kgday:,.0f} kg/day")
    print(f"  ✓ Monthly saving (30 days)                : {kgday*30:,.0f} kg/month")

# =============================================================
# STEP 7 — 5 THESIS PLOTS
# =============================================================
fig, axes = plt.subplots(5, 1, figsize=(16, 28))
fig.suptitle(
    "M/V Kattegat — Hull Performance & CII Analysis\n"
    "Algeciras–Tanger Med  |  Jan 2024 – May 2026",
    fontsize=15, fontweight="bold", y=1.005)

C_PRE="#C0392B"; C_PST="#27AE60"
C_AVG="#2C3E50"; C_DD1="#E67E22"; C_DD2="#27AE60"

def shade(ax):
    ax.axvspan(DRYDOCK_START, DRYDOCK_END,
               alpha=0.15, color="#95A5A6",
               label="Drydock (04/03–04/04/25)")
    ax.axvline(DRYDOCK_START, c=C_DD1, lw=1.8, ls="--",
               label="Drydock start (04/03/25)")
    ax.axvline(DRYDOCK_END,   c=C_DD2, lw=1.8, ls="--",
               label="First voyage (04/04/25)")

# ── Plot 1: Daily ME Consumption ──────────────────────────────
ax = axes[0]
ax.scatter(pre["date"],  pre["me_kg"],
           c=C_PRE, alpha=0.5, s=15, label="Pre-drydock (fouled hull)")
ax.scatter(post["date"], post["me_kg"],
           c=C_PST, alpha=0.5, s=15, label="Post-drydock (clean hull)")
ax.plot(daily["date"], daily["me_7day"],
        c=C_AVG, lw=2, label="7-day rolling avg")
shade(ax)
ax.set_ylabel("ME Consumption (kg/day)", fontsize=11)
ax.set_title("① Daily Main Engine Fuel Consumption — HFO only",
             fontsize=11, fontweight="bold", pad=8)
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# ── Plot 2: HFO per Nautical Mile ────────────────────────────
ax = axes[1]
ax.scatter(pre["date"],  pre["hfo_per_nm"],
           c=C_PRE, alpha=0.5, s=15, label="Pre-drydock")
ax.scatter(post["date"], post["hfo_per_nm"],
           c=C_PST, alpha=0.5, s=15, label="Post-drydock")
ax.plot(daily["date"], daily["kgnm_7day"],
        c=C_AVG, lw=2, label="7-day rolling avg")
ax.axhline(baseline_kgnm, c="#8E44AD", lw=2, ls="--",
           label=f"Clean hull baseline ({baseline_kgnm:.1f} kg/nm)")
shade(ax)
ax.set_ylabel("HFO (kg/nm)", fontsize=11)
ax.set_title("② Fuel Efficiency — HFO per Nautical Mile  (distance-normalised)",
             fontsize=11, fontweight="bold", pad=8)
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# ── Plot 3: Fouling Index ─────────────────────────────────────
ax = axes[2]
ax.bar(pre["date"],  pre["fouling_index"],
       color="#E74C3C", alpha=0.5, width=0.9, label="Pre-drydock")
ax.bar(post["date"], post["fouling_index"],
       color=post["fouling_index"].apply(
           lambda x: "#E74C3C" if x > 0 else "#27AE60"),
       alpha=0.5, width=0.9, label="Post-drydock")
ax.plot(daily["date"], daily["fi_7day"],
        c=C_AVG, lw=2, label="7-day avg")
ax.axhline(0,  c="black",   lw=1.0)
ax.axhline(10, c="#E67E22", lw=1.2, ls=":", label="10% threshold")
ax.axhline(20, c=C_PRE,     lw=1.2, ls=":", label="20% threshold")
shade(ax)
ax.set_ylabel("Fouling Index (%)", fontsize=11)
ax.set_title(
    f"③ Hull Fouling Index  |  Baseline = {baseline_kgnm:.1f} kg/nm "
    f"(first 14 days post-drydock Apr 2025)",
    fontsize=11, fontweight="bold", pad=8)
ax.legend(fontsize=8, ncol=3); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# ── Plot 4: Post-Drydock Degradation Trajectory ──────────────
ax = axes[3]
post_d = daily[daily["days_since_drydock"] >= 0].copy()
ax.scatter(post_d["days_since_drydock"],
           post_d["fouling_index"],
           c=C_PST, alpha=0.5, s=15, label="Daily observations")

x = post_d["days_since_drydock"].values.astype(float)
y = post_d["fouling_index"].values.astype(float)
mask = ~(np.isnan(x) | np.isnan(y))
if mask.sum() > 10:
    z = np.polyfit(x[mask], y[mask], 1)
    p = np.poly1d(z)
    rate = z[0] * 30
    xl   = np.linspace(0, max(x[mask].max(), 365), 300)
    ax.plot(xl, p(xl), c="#8E44AD", lw=2, ls="--",
            label=f"Degradation trend: {rate:.2f}%/month")
    print(f"\n  ✓ Fouling rate post-drydock  : {rate:.2f}% per month")
    print(f"  ✓ Projected at 6 months       : {p(180):.1f}%")
    print(f"  ✓ Projected at 12 months      : {p(365):.1f}%")

ax.axhline(10, c="#E67E22", lw=1.2, ls=":", label="10% threshold")
ax.axhline(20, c=C_PRE,     lw=1.2, ls=":", label="20% threshold")
ax.set_xlabel("Days Since First Voyage After Drydock (04 Apr 2025)",
              fontsize=11)
ax.set_ylabel("Fouling Index (%)", fontsize=11)
ax.set_title("④ Hull Degradation Trajectory — Post-Drydock Fouling Regrowth",
             fontsize=11, fontweight="bold", pad=8)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── Plot 5: Daily CII ─────────────────────────────────────────
ax = axes[4]
cii7 = daily["daily_cii"].rolling(7, center=True, min_periods=3).mean()
ax.scatter(pre["date"],  pre["daily_cii"],
           c=C_PRE, alpha=0.5, s=15, label="Pre-drydock")
ax.scatter(post["date"], post["daily_cii"],
           c=C_PST, alpha=0.5, s=15, label="Post-drydock")
ax.plot(daily["date"], cii7, c=C_AVG, lw=2, label="7-day avg")
shade(ax)
ax.set_ylabel("CII (g CO₂ / DWT·nm)", fontsize=11)
ax.set_xlabel("Date", fontsize=11)
ax.set_title(
    "⑤ Daily Carbon Intensity Indicator (CII)\n"
    "HFO (ME) + MGO (AUX + Boiler) — IMO MEPC.354(78)",
    fontsize=11, fontweight="bold", pad=8)
ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

for ax in axes:
    ax.tick_params(axis="x", rotation=15)

plt.tight_layout(h_pad=4.0)

# Save plot
out_dir = os.path.dirname(FILE_PATH)
plot_path = os.path.join(out_dir, "kattegat_day1_analysis.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\n✓ Plot saved  : {plot_path}")
plt.show()

# =============================================================
# STEP 8 — SAVE TWO CSV FILES
# =============================================================
# Voyage-level CSV (raw per-voyage data)
voyage_cols = [
    "dep_dt", "arr_dt", "from_port", "to_port", "period",
    "me_kg", "aux_kg", "boiler_kg", "total_fuel_kg",
    "co2_me_kg", "co2_aux_kg", "co2_total_kg",
    "distance_nm", "speed_kn", "kgnm_me", "time_at_sea_hr",
    "days_since_drydock"
]
voyage_path = os.path.join(out_dir, "kattegat_voyages.csv")
df[voyage_cols].to_csv(voyage_path, index=False)
print(f"✓ Voyage CSV  : {voyage_path}")
print(f"  {len(df)} voyages × {len(voyage_cols)} columns")

# Daily summary CSV (aggregated — for ML and CII modules)
daily_cols = [
    "date", "period",
    "me_kg", "aux_kg", "boiler_kg", "total_fuel_kg",
    "co2_me_kg", "co2_total_kg",
    "distance_nm", "num_voyages",
    "avg_speed_kn", "avg_kgnm_me",
    "hfo_per_nm", "daily_cii",
    "days_since_drydock", "fouling_index"
]
daily_path = os.path.join(out_dir, "kattegat_daily.csv")
daily[daily_cols].to_csv(daily_path, index=False)
print(f"✓ Daily CSV   : {daily_path}")
print(f"  {len(daily)} days × {len(daily_cols)} columns")

print("\n" + "=" * 60)
print("  DAY 1 COMPLETE ✓")
print("=" * 60)
print(f"""
Two CSV files saved for next scripts:
  kattegat_voyages.csv  → 5,704 per-voyage records
                          → Used for ML model training (Day 3)
  kattegat_daily.csv    → {len(daily)} daily summaries
                          → Used for Holtrop-Mennen (Day 2)
                          → Used for CII module (Day 4)
                          → Used for drydocking optimizer (Day 5)

Confirmed fixed parameters for physics model:
  Engine RPM   : 155
  Engine Load  : 57.5%
  Shaft Power  : {SHAFT_KW:,.0f} kW

Next:
  Day 2 → Holtrop-Mennen physics baseline model
  Day 3 → Random Forest ML model (5,704 training samples)
  Day 4 → CII monthly forecast
  Day 5 → Drydocking optimizer
""")
