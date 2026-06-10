"""
=============================================================
  KATTEGAT DIGITAL TWIN -- DAY 5 SCRIPT
  Predictive Drydocking Optimization
  Vessel : M/V Kattegat (IMO 9112765) -- DFDS
=============================================================

WHAT THIS SCRIPT DOES:
-----------------------
  Builds a techno-economic decision model that recommends
  the optimal drydocking date by minimising total cost:

    Total Cost = Drydock Cost + Offhire Cost + Cumulative Fuel Waste

  Four analyses:
    1. Historical fouling cost quantification
       (what fouling actually cost Jan 2024 - May 2026)

    2. Three-scenario comparison
       (Early / Baseline / Late drydocking strategies)

    3. Optimal drydocking window finder
       (minimises total cost over 36-month horizon)

    4. Sensitivity analysis
       (how optimal date changes with fuel price)

COST ASSUMPTIONS (declared explicitly -- Chapter 4):
------------------------------------------------------
  Fuel price          : 620 USD/ton VLSFO (Mediterranean 2025)
  Drydock hull work   : 300,000 USD (hull cleaning + antifouling)
                        Source: Schultz et al. (2011), bottom-up derivation
  Off-hire rate       : 50,000 USD/day (net operating contribution foregone)
                        Source: DFDS Annual Report 2024
  Drydock duration    : 30 days (confirmed: 4 Mar – 3 Apr 2025)
  Total drydock cost  : 1,800,000 USD
  Source: DFDS Annual Report 2024; Schultz et al. (2011)

HOW TO USE:
-----------
  1. Make sure kattegat_daily_fixed.csv is in your folder
  2. Change DATA_FOLDER below
  3. Terminal: python3 kattegat_day5.py
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
# COST PARAMETERS (declared assumptions)
# =============================================================
FUEL_PRICE_USD   = 620      # USD/ton VLSFO Mediterranean 2025
DRYDOCK_WORK_USD = 300000   # USD hull cleaning + antifouling coat
                               # Source: Schultz et al.(2011); bottom-up derivation
OFFHIRE_DAY_USD  = 50000    # USD/day net operating contribution foregone
                               # Source: DFDS Annual Report 2024
DRYDOCK_DAYS     = 30       # days in drydock -- confirmed from vessel records
                               # 4 March 2025 -> 3 April 2025
TOTAL_DRYDOCK_USD= DRYDOCK_WORK_USD + OFFHIRE_DAY_USD * DRYDOCK_DAYS

# Ship parameters
REAL_BASELINE    = 58.2     # kg/nm clean hull confirmed
FOULING_RATE     = 1.46     # %/month from Day 3 ML model
AVG_DIST_DAY     = 148.0    # nm/day average
AVG_ME_TONS_DAY  = 8.5      # tons/day ME fuel average
DRYDOCK_END      = pd.to_datetime("2025-04-04")
STUDY_END        = pd.to_datetime("2026-05-02")  # optimiser reference point
DRYDOCK_START    = pd.to_datetime("2025-03-04")

print("=" * 60)
print("  KATTEGAT DIGITAL TWIN -- DAY 5: DRYDOCKING OPTIMIZER")
print("=" * 60)
print(f"\n  Cost assumptions:")
print(f"    Fuel price      : ${FUEL_PRICE_USD}/ton VLSFO")
print(f"    Drydock work    : ${DRYDOCK_WORK_USD:,}")
print(f"    Off-hire rate   : ${OFFHIRE_DAY_USD:,}/day")
print(f"    Drydock days    : {DRYDOCK_DAYS} days")
print(f"    TOTAL drydock   : ${TOTAL_DRYDOCK_USD:,}")

# =============================================================
# HELPER FUNCTIONS
# =============================================================
def fouling_cost_per_day(fouling_pct):
    """Extra daily fuel cost due to fouling (USD/day)."""
    extra_tons = AVG_ME_TONS_DAY * (fouling_pct / 100.0)
    return extra_tons * FUEL_PRICE_USD

def cumulative_fouling_cost(months, start_fi=0.0):
    """Total extra fuel cost over given months from start_fi."""
    total = 0.0
    for m in range(1, int(months) + 1):
        fi = start_fi + FOULING_RATE * m
        total += fouling_cost_per_day(fi) * 30
    return total

def total_lifecycle_cost(drydock_month, horizon=23, start_fi=0.0):
    """
    Total cost of a drydocking strategy over horizon months.
    drydock_month = when to drydock (months from now)
    Returns: (total_cost, fuel_waste_before, drydock_cost, fuel_waste_after)
    """
    # Fuel waste BEFORE drydocking
    waste_before = cumulative_fouling_cost(drydock_month, start_fi)

    # Drydock cost (work + offhire)
    dd_cost = TOTAL_DRYDOCK_USD

    # After drydocking: hull is clean, fouling restarts from 0
    months_after = horizon - drydock_month
    waste_after  = cumulative_fouling_cost(months_after, 0.0) if months_after > 0 else 0

    total = waste_before + dd_cost + waste_after
    return total, waste_before, dd_cost, waste_after

# =============================================================
# MODULE 1 -- HISTORICAL FOULING COST QUANTIFICATION
# =============================================================
print("\n" + "=" * 60)
print("  MODULE 1: HISTORICAL FOULING COST")
print("=" * 60)

daily = pd.read_csv(DAILY_CSV, parse_dates=["date"])
daily = daily[daily["period"] != "Drydock"].copy()
daily = daily.sort_values("date").reset_index(drop=True)

# Use weather-corrected fuel
daily["fuel_kg"] = np.where(
    daily["me_kg_corrected"].notna(),
    daily["me_kg_corrected"],
    daily["me_kg"]
)
daily["fuel_tons"] = daily["fuel_kg"] / 1000

# Theoretical clean hull fuel
daily["clean_fuel_tons"] = daily["distance_nm"] * REAL_BASELINE / 1000

# Extra fuel from fouling
daily["extra_fuel_tons"] = (daily["fuel_tons"] - daily["clean_fuel_tons"]).clip(lower=0)
daily["extra_cost_usd"]  = daily["extra_fuel_tons"] * FUEL_PRICE_USD

pre  = daily[daily["period"] == "Pre-Drydock"]
post = daily[daily["period"] == "Post-Drydock"]

total_extra_pre  = pre["extra_cost_usd"].sum()
total_extra_post = post["extra_cost_usd"].sum()
total_extra      = daily["extra_cost_usd"].sum()

print(f"\n  Historical fouling cost Jan 2024 -- May 2026:")
print(f"    Pre-drydock period  : ${total_extra_pre:>12,.0f}")
print(f"    Post-drydock period : ${total_extra_post:>12,.0f}")
print(f"    Total extra fuel    : ${total_extra:>12,.0f}")
print(f"    Daily avg (pre)     : ${pre['extra_cost_usd'].mean():>12,.0f}/day")
print(f"    Daily avg (post)    : ${post['extra_cost_usd'].mean():>12,.0f}/day")

# Cumulative
daily["cumul_extra_cost"] = daily["extra_cost_usd"].cumsum()

# =============================================================
# MODULE 2 -- THREE-SCENARIO COMPARISON
# =============================================================
print("\n" + "=" * 60)
print("  MODULE 2: THREE-SCENARIO COMPARISON")
print("=" * 60)

# Regulatory horizon: SOLAS Reg I/7 — Ro-Pax max 36m between DDs
# Last DD April 2025 + 36m = April 2028
# Study end May 2026 → months remaining = 23
horizon  = 23  # months from study end (SOLAS Reg I/7 hard limit)
last_fi  = post["hm_fouling_pct"].dropna().tail(30).mean()
last_fi  = max(last_fi, 0) if not np.isnan(last_fi) else 8.0

print(f"\n  Current fouling index    : {last_fi:.1f}%")
print(f"  Fouling rate             : {FOULING_RATE}%/month")
print(f"  Planning horizon         : {horizon} months")
print()

# Scenarios aligned to regulatory boundaries
# Month 7  = optimal (Dec 2026 = 20m from last DD)
# Month 11 = DFDS 24-month practice boundary (Apr 2027)
# Month 23 = SOLAS 36-month hard limit (Apr 2028)
scenarios = {
    "Optimal (Month 7 — Dec 2026)" : 7,
    "DFDS practice (Month 11 — Apr 2027)": 11,
    "SOLAS hard limit (Month 23 — Apr 2028)": 23,
}

print(f"  {'Scenario':<25} {'Drydock Month':>14} {'Fouling at DD':>14} "
      f"{'Waste Before':>14} {'DD Cost':>12} {'Total Cost':>12}")
print("  " + "-" * 92)

results = {}
for name, dd_month in scenarios.items():
    total, waste_b, dd_c, waste_a = total_lifecycle_cost(
        dd_month, horizon, last_fi)
    fi_at_dd = last_fi + FOULING_RATE * dd_month
    results[name] = {
        "month": dd_month, "fi_at_dd": fi_at_dd,
        "waste_before": waste_b, "dd_cost": dd_c,
        "waste_after": waste_a, "total": total
    }
    print(f"  {name:<25} {dd_month:>14} {fi_at_dd:>13.1f}% "
          f"${waste_b:>13,.0f} ${dd_c:>11,.0f} ${total:>11,.0f}")

# =============================================================
# MODULE 3 -- OPTIMAL DRYDOCKING WINDOW
# =============================================================
print("\n" + "=" * 60)
print("  MODULE 3: OPTIMAL DRYDOCKING WINDOW")
print("=" * 60)

months_range  = np.arange(1, horizon + 1)
total_costs   = []
waste_befores = []
waste_afters  = []
fi_at_dd_list = []

for m in months_range:
    total, wb, dd, wa = total_lifecycle_cost(m, horizon, last_fi)
    total_costs.append(total)
    waste_befores.append(wb)
    waste_afters.append(wa)
    fi_at_dd_list.append(last_fi + FOULING_RATE * m)

total_costs   = np.array(total_costs)
waste_befores = np.array(waste_befores)
waste_afters  = np.array(waste_afters)

optimal_month = months_range[np.argmin(total_costs)]
optimal_cost  = total_costs[np.argmin(total_costs)]
optimal_fi    = fi_at_dd_list[np.argmin(total_costs)]
optimal_date  = STUDY_END + pd.DateOffset(months=optimal_month)

# Also find where fouling crosses 10% and 20%
month_10pct = (10.0 - last_fi) / FOULING_RATE
month_20pct = (20.0 - last_fi) / FOULING_RATE

print(f"\n  Optimal drydocking window: Month {optimal_month}")
print(f"  Optimal drydocking date  : {optimal_date.strftime('%B %Y')}")
print(f"  Fouling at optimal DD    : {optimal_fi:.1f}%")
print(f"  Minimum total cost       : ${optimal_cost:,.0f}")
print(f"\n  10% fouling threshold    : Month {month_10pct:.1f}")
print(f"  20% fouling threshold    : Month {month_20pct:.1f}")
print(f"\n  Savings vs SOLAS limit   : "
      f"${results['SOLAS hard limit (Month 23 — Apr 2028)']['total']-optimal_cost:,.0f}")
print(f"  Savings vs DFDS practice : "
      f"${results['DFDS practice (Month 11 — Apr 2027)']['total']-optimal_cost:,.0f}")

# =============================================================
# MODULE 4 -- SENSITIVITY ANALYSIS
# =============================================================
print("\n" + "=" * 60)
print("  MODULE 4: SENSITIVITY TO FUEL PRICE")
print("=" * 60)

fuel_prices  = [400, 500, 600, 620, 700, 800, 900]
opt_months_fp= []

print(f"\n  {'Fuel Price':>12} {'Optimal Month':>14} "
      f"{'Optimal Date':>15} {'Total Cost':>12}")
print("  " + "-" * 56)

for fp in fuel_prices:
    costs_fp = []
    for m in months_range:
        # Recalculate with different fuel price
        waste_b = sum([AVG_ME_TONS_DAY*(last_fi+FOULING_RATE*i)/100*fp*30
                       for i in range(1, m+1)])
        months_a = horizon - m
        waste_a  = sum([AVG_ME_TONS_DAY*(FOULING_RATE*i)/100*fp*30
                        for i in range(1, months_a+1)]) if months_a > 0 else 0
        costs_fp.append(waste_b + TOTAL_DRYDOCK_USD + waste_a)
    opt_m  = months_range[np.argmin(costs_fp)]
    opt_c  = min(costs_fp)
    opt_d  = (STUDY_END + pd.DateOffset(months=opt_m)).strftime("%b %Y")
    opt_months_fp.append(opt_m)
    marker = " <-- current" if fp == FUEL_PRICE_USD else ""
    print(f"  ${fp:>10}/ton {opt_m:>14} {opt_d:>15} "
          f"${opt_c:>11,.0f}{marker}")

# =============================================================
# MODULE 5 -- 4 THESIS PLOTS
# =============================================================
fig, axes = plt.subplots(4, 1, figsize=(16, 26))
fig.suptitle(
    "M/V Kattegat -- Predictive Drydocking Optimization\n"
    "Techno-Economic Decision Model | Jan 2024 -- May 2026",
    fontsize=15, fontweight="bold", y=1.005)

C_PRE="#C0392B"; C_PST="#27AE60"; C_AVG="#2C3E50"
C_DD1="#E67E22"; C_DD2="#27AE60"; C_OPT="#8E44AD"

def shade(ax):
    ax.axvspan(DRYDOCK_START, DRYDOCK_END,
               alpha=0.12, color="#95A5A6", label="Drydock")
    ax.axvline(DRYDOCK_START, c=C_DD1, lw=1.5, ls="--")
    ax.axvline(DRYDOCK_END,   c=C_DD2, lw=1.5, ls="--",
               label="First voyage (Apr 2025)")

# ── Plot 1: Historical cumulative fouling cost ────────────────
ax = axes[0]
pre_d  = daily[daily["period"] == "Pre-Drydock"]
post_d = daily[daily["period"] == "Post-Drydock"]

ax.fill_between(pre_d["date"], pre_d["extra_cost_usd"],
                alpha=0.5, color=C_PRE, label="Daily extra cost (pre)")
ax.fill_between(post_d["date"], post_d["extra_cost_usd"],
                alpha=0.5, color=C_PST, label="Daily extra cost (post)")
ax.plot(daily["date"], daily["extra_cost_usd"].rolling(30, min_periods=5).mean(),
        c=C_AVG, lw=2, label="30-day rolling avg")

ax2 = ax.twinx()
ax2.plot(daily["date"], daily["cumul_extra_cost"] / 1e6,
         c="#8E44AD", lw=2.5, ls="--", label="Cumulative cost (M USD)")
ax2.set_ylabel("Cumulative Cost (M USD)", fontsize=10, color="#8E44AD")
ax2.tick_params(axis="y", colors="#8E44AD")

shade(ax)
ax.set_ylabel("Daily Extra Fuel Cost (USD)", fontsize=11)
ax.set_title(
    "① Historical Fouling Cost -- Jan 2024 to May 2026\n"
    f"Total extra fuel cost: ${total_extra:,.0f}  |  "
    f"Pre-drydock: ${total_extra_pre:,.0f}  |  "
    f"Post-drydock: ${total_extra_post:,.0f}",
    fontsize=11, fontweight="bold", pad=8)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, fontsize=8, ncol=3)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# ── Plot 2: Three-scenario total cost comparison ─────────────
ax = axes[1]
scenario_names  = list(results.keys())
scenario_months = [results[n]["month"] for n in scenario_names]
waste_b_vals = [results[n]["waste_before"]/1000 for n in scenario_names]
dd_cost_vals = [results[n]["dd_cost"]/1000 for n in scenario_names]
waste_a_vals = [results[n]["waste_after"]/1000 for n in scenario_names]
total_vals   = [results[n]["total"]/1000 for n in scenario_names]
x = np.arange(len(scenario_names))
w = 0.6

p1 = ax.bar(x, waste_b_vals, w, color=C_PRE, alpha=0.8,
            label="Fuel waste before drydock")
p2 = ax.bar(x, dd_cost_vals, w, bottom=waste_b_vals,
            color="#2980B9", alpha=0.8, label="Drydock cost (work + offhire)")
p3 = ax.bar(x, waste_a_vals, w,
            bottom=[wb+dd for wb,dd in zip(waste_b_vals, dd_cost_vals)],
            color=C_PST, alpha=0.8, label="Fuel waste after drydock")

# Total cost labels
for i, total in enumerate(total_vals):
    ax.text(i, total + 10, f"${total*1000:,.0f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(scenario_names, fontsize=10)
ax.set_ylabel("Cost (thousands USD)", fontsize=11)
ax.set_title(
    f"② Scenario Comparison — Regulatory Drydocking Boundaries\n"
    f"Fuel: ${FUEL_PRICE_USD}/ton | Drydock cost: ${TOTAL_DRYDOCK_USD:,} "
    f"(${DRYDOCK_WORK_USD:,} work + ${OFFHIRE_DAY_USD:,}/day x {DRYDOCK_DAYS}d) "
    f"| SOLAS horizon: 23m | DFDS practice: 11m",
    fontsize=10, fontweight="bold", pad=8)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# ── Plot 3: Optimal drydocking curve ─────────────────────────
ax = axes[2]
ax.plot(months_range, total_costs/1000, c=C_AVG, lw=2.5,
        label="Total cost vs drydock timing")
ax.fill_between(months_range, total_costs/1000,
                alpha=0.1, color=C_AVG)

# Mark optimal
ax.axvline(optimal_month, c=C_OPT, lw=2.5, ls="--",
           label=f"Optimal: Month {optimal_month} "
                 f"({optimal_date.strftime('%b %Y')})")
ax.scatter([optimal_month], [optimal_cost/1000],
           c=C_OPT, s=150, zorder=6)

# Mark scenarios
# Mark regulatory boundaries
ax.axvline(11, c="#E67E22", lw=1.5, ls=":", alpha=0.8,
           label="DFDS 24m practice (Month 11)")
ax.axvline(23, c="#C0392B", lw=2, ls="-.", alpha=0.9,
           label="SOLAS 36m hard limit (Month 23)")
for name, dd_m in scenarios.items():
    idx = dd_m - 1
    ax.scatter([dd_m], [total_costs[idx]/1000],
               s=80, zorder=5, alpha=0.8)
    ax.annotate(name.split("(")[0].strip(),
                xy=(dd_m, total_costs[idx]/1000),
                xytext=(dd_m+0.5, total_costs[idx]/1000+15),
                fontsize=8, alpha=0.8)

# Threshold lines
ax.axvline(month_10pct, c="#E67E22", lw=1.5, ls=":",
           label=f"10% fouling (Month {month_10pct:.0f})")
ax.axvline(month_20pct, c=C_PRE,    lw=1.5, ls=":",
           label=f"20% fouling (Month {month_20pct:.0f})")

ax.set_xlabel("Months from Now Until Drydocking", fontsize=11)
ax.set_ylabel("Total Lifecycle Cost (k USD)", fontsize=11)
ax.set_title(
    f"③ Optimal Drydocking Window — Month {optimal_month} "
    f"({optimal_date.strftime('%B %Y')})\n"
    f"Fouling at optimal drydock: {optimal_fi:.1f}%  |  "
    f"Minimum cost: ${optimal_cost:,.0f}",
    fontsize=11, fontweight="bold", pad=8)
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# ── Plot 4: Sensitivity to fuel price ────────────────────────
ax = axes[3]

colors_sens = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(fuel_prices)))

for i, fp in enumerate(fuel_prices):
    costs_fp = []
    for m in months_range:
        wb = sum([AVG_ME_TONS_DAY*(last_fi+FOULING_RATE*j)/100*fp*30
                  for j in range(1, m+1)])
        ma = horizon - m
        wa = sum([AVG_ME_TONS_DAY*(FOULING_RATE*j)/100*fp*30
                  for j in range(1, ma+1)]) if ma > 0 else 0
        costs_fp.append((wb + TOTAL_DRYDOCK_USD + wa)/1000)
    opt_m = months_range[np.argmin(costs_fp)]
    lw    = 3.0 if fp == FUEL_PRICE_USD else 1.5
    ls    = "-" if fp == FUEL_PRICE_USD else "--"
    label = f"${fp}/ton"
    if fp == FUEL_PRICE_USD:
        label += " (current)"
    ax.plot(months_range, costs_fp, c=colors_sens[i],
            lw=lw, ls=ls, alpha=0.85, label=label)
    ax.scatter([opt_m], [costs_fp[opt_m-1]],
               c=colors_sens[i], s=60, zorder=5)

ax.set_xlabel("Months from Now Until Drydocking", fontsize=11)
ax.set_ylabel("Total Lifecycle Cost (k USD)", fontsize=11)
ax.set_title(
    "④ Sensitivity Analysis -- Optimal Timing vs Fuel Price\n"
    "Dots mark optimal drydocking month for each fuel price scenario",
    fontsize=11, fontweight="bold", pad=8)
ax.legend(fontsize=8, ncol=4)
ax.grid(True, alpha=0.3)

for ax in axes[:1]:
    ax.tick_params(axis="x", rotation=15)

plt.tight_layout(h_pad=4.0)
plot_path = os.path.join(DATA_FOLDER, "kattegat_day5_optimizer.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\n✓ Plot saved : {plot_path}")
plt.show()

# =============================================================
# MODULE 6 -- SAVE CSV
# =============================================================
opt_df = pd.DataFrame({
    "months_from_now"  : months_range,
    "optimal_date"     : [(STUDY_END+pd.DateOffset(months=int(m))).date()
                          for m in months_range],
    "fouling_at_dd_pct": fi_at_dd_list,
    "waste_before_usd" : waste_befores,
    "drydock_cost_usd" : TOTAL_DRYDOCK_USD,
    "waste_after_usd"  : waste_afters,
    "total_cost_usd"   : total_costs,
})
opt_path = os.path.join(DATA_FOLDER, "kattegat_optimizer_results.csv")
opt_df.to_csv(opt_path, index=False)
print(f"✓ CSV saved  : {opt_path}")

# =============================================================
# KEY THESIS NUMBERS
# =============================================================
print("\n" + "=" * 60)
print("  KEY THESIS NUMBERS -- CHAPTER 5")
print("=" * 60)
print(f"\n  HISTORICAL FOULING COST (Jan 2024 -- May 2026):")
print(f"    Total extra fuel cost   : ${total_extra:,.0f}")
print(f"    Pre-drydock total       : ${total_extra_pre:,.0f}")
print(f"    Post-drydock total      : ${total_extra_post:,.0f}")
print(f"    Daily avg pre-drydock   : ${pre['extra_cost_usd'].mean():,.0f}/day")
print(f"\n  OPTIMAL DRYDOCKING DECISION:")
print(f"    Optimal month           : Month {optimal_month} from now")
print(f"    Optimal date            : {optimal_date.strftime('%B %Y')}")
print(f"    Fouling at optimal DD   : {optimal_fi:.1f}%")
print(f"    Minimum total cost      : ${optimal_cost:,.0f}")
print(f"\n  SCENARIO COMPARISON (36-month horizon):")
for name, res in results.items():
    saving = res['total'] - optimal_cost
    print(f"    {name:<25}: ${res['total']:>10,.0f}  "
          f"(+${saving:>8,.0f} vs optimal)")
print(f"\n  SENSITIVITY ANALYSIS:")
print(f"    At $400/ton: optimal month {opt_months_fp[0]}")
print(f"    At $620/ton: optimal month {opt_months_fp[fuel_prices.index(620)]}"
      f" (current)")
print(f"    At $900/ton: optimal month {opt_months_fp[-1]}")

print("\n" + "=" * 60)
print("  DAY 5 COMPLETE")
print("=" * 60)
print("""
Files saved:
  kattegat_day5_optimizer.png     -> 4 thesis plots
  kattegat_optimizer_results.csv  -> full optimization data

Next:
  Day 6 -> Streamlit dashboard
  Then  -> Write thesis chapters
""")
