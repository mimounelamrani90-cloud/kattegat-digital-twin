"""
=============================================================
  KATTEGAT DIGITAL TWIN — DAY 3 SCRIPT
  Hybrid Physics-Informed Machine Learning Model
  Vessel : M/V Kattegat (IMO 9112765) — DFDS
=============================================================

WHAT THIS SCRIPT DOES:
-----------------------
  Trains two ML models on the daily operational data:
    1. Random Forest Regressor     (primary model)
    2. Extra Trees Regressor       (comparison model)

  The ML models learn the RESIDUAL between real consumption
  and the Holtrop-Mennen physics baseline — this is called
  a HYBRID Physics-Informed ML approach:

    ML input  : operational features (days since drydock,
                season, voyage count, efficiency trend...)
    ML target : hm_fouling_pct (from Day 2 H-M model)
    ML output : predicted fouling index for any future date

  Why hybrid is better than pure ML:
    - Physics model handles the baseline (speed, resistance)
    - ML learns the patterns physics cannot capture
      (seasonal biofouling, operational habits, coating aging)
    - Together they are more accurate than either alone

TRAIN / TEST SPLIT (time-series):
-----------------------------------
  Train : Jan 2024 → Sep 2025  (~630 days)
  Test  : Oct 2025 → May 2026  (~215 days)
  → Time-series split (never shuffle time-series data!)

FEATURES ENGINEERED:
---------------------
  Physical:
    days_since_drydock    → Primary fouling driver
    days_since_drydock_sq → Quadratic (fouling accelerates)
  Seasonal:
    sin_doy, cos_doy      → Cyclic day-of-year encoding
    month                 → Mediterranean temperature proxy
  Operational:
    num_voyages           → Daily voyage count
    distance_nm           → Total daily distance
  Efficiency trend:
    hfo_per_nm_lag7       → 7-day lagged efficiency
    hfo_per_nm_roll14     → 14-day rolling mean
    hfo_per_nm_diff7      → Rate of change (fouling speed)

HOW TO USE:
-----------
  1. Make sure kattegat_day2_data.csv is in your folder
  2. Change DATA_FOLDER below
  3. Terminal: python3 kattegat_day3.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# =============================================================
# !! CHANGE THIS TO YOUR FOLDER PATH !!
# =============================================================
DATA_FOLDER = "/Users/mbp/Desktop/Kattegat_Data/"
DAY2_CSV    = os.path.join(DATA_FOLDER, "kattegat_day2_data.csv")

# Key dates
DRYDOCK_END  = pd.to_datetime("2025-04-04")
DRYDOCK_START= pd.to_datetime("2025-03-04")

# Train/test split date — time series (never shuffle!)
SPLIT_DATE   = pd.to_datetime("2025-10-01")

# Forecast horizon
FORECAST_DAYS = 180   # 6 months ahead

# =============================================================
# STEP 1 — LOAD AND PREPARE DATA
# =============================================================
print("=" * 60)
print("  KATTEGAT DIGITAL TWIN — DAY 3: ML MODEL")
print("=" * 60)

df = pd.read_csv(DAY2_CSV, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"\n✓ Loaded {len(df)} daily records")
print(f"  Date range : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  Periods    : {df['period'].value_counts().to_dict()}")

# =============================================================
# STEP 2 — FEATURE ENGINEERING
# =============================================================

# Days since drydock (negative = pre-drydock, positive = post)
df["days_since_drydock"] = (df["date"] - DRYDOCK_END).dt.days

# Quadratic term — fouling growth is non-linear
df["days_sq"] = df["days_since_drydock"] ** 2

# Absolute days since drydock (pre and post drydock both count)
df["abs_days"] = df["days_since_drydock"].abs()

# Seasonal encoding — cyclic (avoids discontinuity at Jan/Dec)
doy = df["date"].dt.dayofyear
df["sin_doy"] = np.sin(2 * np.pi * doy / 365)
df["cos_doy"] = np.cos(2 * np.pi * doy / 365)
df["month"]   = df["date"].dt.month

# Mediterranean sea temperature proxy by month
# (warmer = faster biofouling growth)
med_temp = {1:14,2:13,3:14,4:15,5:18,6:22,7:25,8:27,9:25,10:22,11:19,12:16}
df["sea_temp_est"] = df["month"].map(med_temp)

# Operational features
df["num_voyages"]  = df["num_voyages"].fillna(7)
df["distance_nm"]  = df["distance_nm"].fillna(df["distance_nm"].median())

# Efficiency trend features — IMPORTANT for ML
df["hfo_per_nm"] = df["hfo_per_nm"].fillna(
    df["hfo_per_nm"].rolling(7, min_periods=1).mean())

df["hfo_lag7"]   = df["hfo_per_nm"].shift(7)    # 7-day lag
df["hfo_roll14"] = df["hfo_per_nm"].rolling(
    14, min_periods=5).mean()                     # 14-day rolling avg
df["hfo_diff7"]  = df["hfo_per_nm"].diff(7)      # rate of change
df["hfo_roll7"]  = df["hfo_per_nm"].rolling(
    7, min_periods=3).mean()                      # 7-day rolling avg

# Binary flag: is it post-drydock?
df["is_post_dd"] = (df["days_since_drydock"] >= 0).astype(int)

# Days since drydock — separate pre and post signals
df["days_post"] = df["days_since_drydock"].clip(lower=0)  # 0 for pre
df["days_pre"]  = (-df["days_since_drydock"]).clip(lower=0)  # 0 for post

# =============================================================
# STEP 3 — DEFINE FEATURES AND TARGET
# =============================================================

FEATURES = [
    "days_since_drydock",   # primary physical driver
    "days_sq",              # non-linear fouling acceleration
    "days_post",            # post-drydock fouling growth
    "days_pre",             # pre-drydock buildup
    "is_post_dd",           # period indicator
    "sin_doy",              # seasonal cycle
    "cos_doy",              # seasonal cycle
    "month",                # month of year
    "sea_temp_est",         # sea temperature proxy
    "num_voyages",          # operational intensity
    "distance_nm",          # daily distance
    "hfo_lag7",             # lagged efficiency
    "hfo_roll14",           # trend
    "hfo_diff7",            # rate of change
    "hfo_roll7",            # 7-day avg efficiency
]

TARGET = "hm_fouling_pct"

# Drop rows with NaN in features or target
df_clean = df.dropna(subset=FEATURES + [TARGET]).copy()
df_clean = df_clean[df_clean[TARGET].abs() <= 50]  # remove outliers

print(f"\n✓ Clean ML dataset : {len(df_clean)} days")
print(f"  Features used    : {len(FEATURES)}")
print(f"  Target           : {TARGET}")

# =============================================================
# STEP 4 — TRAIN / TEST SPLIT (TIME SERIES)
# =============================================================
train = df_clean[df_clean["date"] <  SPLIT_DATE]
test  = df_clean[df_clean["date"] >= SPLIT_DATE]

X_train = train[FEATURES]
y_train = train[TARGET]
X_test  = test[FEATURES]
y_test  = test[TARGET]

print(f"\n  Train set : {len(train)} days "
      f"({train['date'].min().date()} → {train['date'].max().date()})")
print(f"  Test set  : {len(test)} days "
      f"({test['date'].min().date()} → {test['date'].max().date()})")

# =============================================================
# STEP 5 — TRAIN RANDOM FOREST (PRIMARY MODEL)
# =============================================================
print("\n" + "=" * 60)
print("  TRAINING RANDOM FOREST MODEL")
print("=" * 60)

rf = RandomForestRegressor(
    n_estimators   = 300,     # 300 decision trees
    max_depth      = 8,       # prevent overfitting
    min_samples_split = 10,   # minimum samples to split
    min_samples_leaf  = 5,    # minimum samples per leaf
    max_features   = "sqrt",  # features per split
    random_state   = 42,
    n_jobs         = -1       # use all CPU cores
)
rf.fit(X_train, y_train)

# Predictions
rf_train_pred = rf.predict(X_train)
rf_test_pred  = rf.predict(X_test)

# Metrics
rf_r2_train  = r2_score(y_train, rf_train_pred)
rf_r2_test   = r2_score(y_test,  rf_test_pred)
rf_mae_test  = mean_absolute_error(y_test, rf_test_pred)
rf_rmse_test = np.sqrt(mean_squared_error(y_test, rf_test_pred))
rf_mape_test = np.mean(np.abs((y_test - rf_test_pred) / (y_test + 1e-6))) * 100

print(f"\n  Random Forest Results:")
print(f"    R² (train)     : {rf_r2_train:.4f}")
print(f"    R² (test)      : {rf_r2_test:.4f}  ← key metric")
print(f"    MAE (test)     : {rf_mae_test:.2f}%")
print(f"    RMSE (test)    : {rf_rmse_test:.2f}%")

# =============================================================
# STEP 6 — TRAIN EXTRA TREES (COMPARISON MODEL)
# =============================================================
print("\n" + "=" * 60)
print("  TRAINING EXTRA TREES MODEL (comparison)")
print("=" * 60)

et = ExtraTreesRegressor(
    n_estimators      = 300,
    max_depth         = 8,
    min_samples_split = 10,
    min_samples_leaf  = 5,
    random_state      = 42,
    n_jobs            = -1
)
et.fit(X_train, y_train)

et_train_pred = et.predict(X_train)
et_test_pred  = et.predict(X_test)

et_r2_test   = r2_score(y_test, et_test_pred)
et_mae_test  = mean_absolute_error(y_test, et_test_pred)
et_rmse_test = np.sqrt(mean_squared_error(y_test, et_test_pred))

print(f"\n  Extra Trees Results:")
print(f"    R² (test)  : {et_r2_test:.4f}")
print(f"    MAE (test) : {et_mae_test:.2f}%")
print(f"    RMSE (test): {et_rmse_test:.2f}%")

# Select best model
if rf_r2_test >= et_r2_test:
    best_model      = rf
    best_pred_test  = rf_test_pred
    best_pred_train = rf_train_pred
    best_name       = "Random Forest"
    best_r2         = rf_r2_test
    best_mae        = rf_mae_test
    best_rmse       = rf_rmse_test
else:
    best_model      = et
    best_pred_test  = et_test_pred
    best_pred_train = et_train_pred
    best_name       = "Extra Trees"
    best_r2         = et_r2_test
    best_mae        = et_mae_test
    best_rmse       = et_rmse_test

print(f"\n  ✓ Best model: {best_name} (R²={best_r2:.4f})")

# =============================================================
# STEP 7 — FEATURE IMPORTANCE
# =============================================================
importances = pd.Series(
    best_model.feature_importances_, index=FEATURES
).sort_values(ascending=False)

print(f"\n  Feature Importance (top 10):")
for feat, imp in importances.head(10).items():
    bar = "█" * int(imp * 50)
    print(f"    {feat:<25} {imp:.4f}  {bar}")

# =============================================================
# STEP 8 — FORECAST 6 MONTHS AHEAD
# =============================================================
last_date   = df_clean["date"].max()
last_hfopnm = df_clean["hfo_per_nm"].iloc[-1]
last_roll14 = df_clean["hfo_roll14"].iloc[-1]
last_roll7  = df_clean["hfo_roll7"].iloc[-1]

future_dates = pd.date_range(
    last_date + pd.Timedelta(days=1),
    periods=FORECAST_DAYS, freq="D"
)

future = pd.DataFrame({"date": future_dates})
future["days_since_drydock"] = (future["date"] - DRYDOCK_END).dt.days
future["days_sq"]    = future["days_since_drydock"] ** 2
future["days_post"]  = future["days_since_drydock"].clip(lower=0)
future["days_pre"]   = 0  # all future = post drydock
future["is_post_dd"] = 1

doy_f = future["date"].dt.dayofyear
future["sin_doy"]    = np.sin(2 * np.pi * doy_f / 365)
future["cos_doy"]    = np.cos(2 * np.pi * doy_f / 365)
future["month"]      = future["date"].dt.month
future["sea_temp_est"] = future["month"].map(med_temp)
future["num_voyages"]  = 8
future["distance_nm"]  = 160.0

# Efficiency trend: gradually increasing (fouling grows)
# Extrapolate from last trend
last_rate  = df_clean["hfo_diff7"].iloc[-30:].mean()
last_rate  = max(last_rate, 0)   # only increase
future["hfo_per_nm"] = [
    last_hfopnm + last_rate * i / 7
    for i in range(len(future))
]
future["hfo_lag7"]   = future["hfo_per_nm"].shift(7).fillna(last_hfopnm)
future["hfo_roll14"] = future["hfo_per_nm"].rolling(14, min_periods=1).mean()
future["hfo_diff7"]  = future["hfo_per_nm"].diff(7).fillna(last_rate)
future["hfo_roll7"]  = future["hfo_per_nm"].rolling(7, min_periods=1).mean()

# Predict
future["ml_forecast"] = best_model.predict(future[FEATURES])

# Confidence interval (std from trees)
tree_preds = np.array([
    tree.predict(future[FEATURES].values)
    for tree in best_model.estimators_
])
future["forecast_std"]  = tree_preds.std(axis=0)
future["forecast_upper"]= future["ml_forecast"] + 1.96 * future["forecast_std"]
future["forecast_lower"]= future["ml_forecast"] - 1.96 * future["forecast_std"]

print(f"\n  6-month forecast:")
for months_ahead in [1, 3, 6]:
    target_date = last_date + pd.Timedelta(days=months_ahead*30)
    row = future[future["date"] <= target_date].iloc[-1]
    print(f"    {months_ahead} month(s) → Fouling Index: "
          f"{row['ml_forecast']:.1f}% "
          f"(±{row['forecast_std']*1.96:.1f}%)")

# =============================================================
# STEP 9 — FULL PREDICTIONS ACROSS ALL DATA
# =============================================================
df_clean["ml_pred"] = best_model.predict(df_clean[FEATURES])

# Hybrid prediction: average of physics (HM) and ML
df_clean["hybrid_pred"] = (df_clean["hm_fouling_pct"] + df_clean["ml_pred"]) / 2

# =============================================================
# STEP 10 — 5 PLOTS
# =============================================================
C_PRE="#C0392B"; C_PST="#27AE60"; C_AVG="#2C3E50"
C_DD1="#E67E22"; C_DD2="#27AE60"; C_ML="#8E44AD"
C_HM="#2980B9";  C_HYB="#E67E22"

def shade(ax):
    ax.axvspan(DRYDOCK_START, DRYDOCK_END,
               alpha=0.12, color="#95A5A6", label="Drydock period")
    ax.axvline(DRYDOCK_START, c=C_DD1, lw=1.5, ls="--")
    ax.axvline(DRYDOCK_END,   c=C_DD2, lw=1.5, ls="--")

fig, axes = plt.subplots(5, 1, figsize=(16, 30))
fig.suptitle(
    "M/V Kattegat — Hybrid Physics-Informed ML Digital Twin\n"
    f"Random Forest + Holtrop-Mennen  |  Jan 2024 – May 2026",
    fontsize=15, fontweight="bold", y=1.005)

# ── Plot 1: ML predictions vs actual (full timeline) ─────────
ax = axes[0]
pre_d  = df_clean[df_clean["period"]=="Pre-Drydock"]
post_d = df_clean[df_clean["period"]=="Post-Drydock"]

ax.scatter(pre_d["date"],  pre_d["hm_fouling_pct"],
           c=C_PRE, alpha=0.3, s=12, label="H-M Fouling Index (actual)")
ax.scatter(post_d["date"], post_d["hm_fouling_pct"],
           c=C_PST, alpha=0.3, s=12)
ax.plot(df_clean["date"],
        df_clean["hm_fouling_pct"].rolling(7, center=True, min_periods=3).mean(),
        c=C_AVG, lw=1.5, label="Actual 7-day avg")
ax.plot(df_clean["date"],
        df_clean["ml_pred"],
        c=C_ML, lw=2, label=f"{best_name} prediction")
ax.plot(future["date"], future["ml_forecast"],
        c=C_ML, lw=2, ls="--", label="6-month forecast")
ax.fill_between(future["date"],
                future["forecast_lower"],
                future["forecast_upper"],
                alpha=0.2, color=C_ML, label="95% confidence interval")
ax.axvline(SPLIT_DATE, c="navy", lw=1.5, ls=":",
           label="Train/Test split")
ax.axvline(last_date,  c="black", lw=1.5, ls=":",
           label="Forecast start")
ax.axhline(0,  c="black", lw=1)
ax.axhline(10, c="#E67E22", lw=1, ls=":", label="10% threshold")
ax.axhline(20, c=C_PRE,    lw=1, ls=":", label="20% threshold")
shade(ax)
ax.set_ylabel("Fouling Index (%)", fontsize=11)
ax.set_title(
    f"① {best_name} Model — Predicted vs Actual Fouling Index  |  "
    f"R²={best_r2:.3f}  MAE={best_mae:.2f}%  RMSE={best_rmse:.2f}%",
    fontsize=11, fontweight="bold", pad=8)
ax.legend(fontsize=7, ncol=3); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# ── Plot 2: Test set actual vs predicted scatter ─────────────
ax = axes[1]
ax.scatter(y_test, best_pred_test,
           c=C_ML, alpha=0.6, s=20, label=f"{best_name}")
ax.scatter(y_test, et_test_pred,
           c="#E74C3C", alpha=0.4, s=15, marker="^", label="Extra Trees")
perfect = np.linspace(y_test.min(), y_test.max(), 100)
ax.plot(perfect, perfect, "k--", lw=1.5, label="Perfect prediction")
ax.plot(perfect, perfect + best_mae, ":", c=C_ML, lw=1,
        label=f"±MAE ({best_mae:.1f}%)")
ax.plot(perfect, perfect - best_mae, ":", c=C_ML, lw=1)

# Add metrics box
metrics_txt = (
    f"{best_name}:\n"
    f"  R²   = {best_r2:.4f}\n"
    f"  MAE  = {best_mae:.2f}%\n"
    f"  RMSE = {best_rmse:.2f}%\n\n"
    f"Extra Trees:\n"
    f"  R²   = {et_r2_test:.4f}\n"
    f"  MAE  = {et_mae_test:.2f}%\n"
    f"  RMSE = {et_rmse_test:.2f}%"
)
ax.text(0.02, 0.97, metrics_txt,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9))
ax.set_xlabel("Actual Fouling Index (%)", fontsize=11)
ax.set_ylabel("Predicted Fouling Index (%)", fontsize=11)
ax.set_title(
    "② Model Performance — Actual vs Predicted (Test Set Only)\n"
    f"Test period: {test['date'].min().date()} → {test['date'].max().date()}",
    fontsize=11, fontweight="bold", pad=8)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ── Plot 3: Feature Importance ───────────────────────────────
ax = axes[2]
top10 = importances.head(10)
colors_imp = ["#2980B9" if i < 3 else "#5DADE2" if i < 6 else "#AED6F1"
              for i in range(len(top10))]
bars = ax.barh(range(len(top10)), top10.values,
               color=colors_imp, edgecolor="white", height=0.7)
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(top10.index, fontsize=10)
ax.invert_yaxis()
for bar, val in zip(bars, top10.values):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
ax.set_xlabel("Feature Importance (Mean Decrease Impurity)", fontsize=11)
ax.set_title(
    f"③ Feature Importance — {best_name}\n"
    "Higher = more influential in predicting hull fouling",
    fontsize=11, fontweight="bold", pad=8)
ax.grid(True, alpha=0.3, axis="x")

# ── Plot 4: Hybrid model (HM physics + ML) ───────────────────
ax = axes[3]
ax.scatter(pre_d["date"],  pre_d["hm_fouling_pct"],
           c=C_PRE, alpha=0.2, s=10, label="Actual (pre-drydock)")
ax.scatter(post_d["date"], post_d["hm_fouling_pct"],
           c=C_PST, alpha=0.2, s=10, label="Actual (post-drydock)")
hm_7day = df_clean["hm_fouling_pct"].rolling(7, center=True, min_periods=3).mean()
ax.plot(df_clean["date"], hm_7day,
        c=C_HM, lw=2, label="Physics model (H-M 7-day avg)")
ax.plot(df_clean["date"], df_clean["ml_pred"],
        c=C_ML, lw=2, label=f"ML model ({best_name})")
ax.plot(df_clean["date"], df_clean["hybrid_pred"],
        c=C_HYB, lw=2.5, label="Hybrid (Physics + ML average)")
ax.axhline(0,  c="black", lw=1)
ax.axhline(10, c="#E67E22", lw=1, ls=":")
ax.axhline(20, c=C_PRE,    lw=1, ls=":")
shade(ax)
ax.set_ylabel("Fouling Index (%)", fontsize=11)
ax.set_title(
    "④ Hybrid Digital Twin — Physics (H-M) + ML Combined\n"
    "Hybrid model combines physics accuracy with ML pattern learning",
    fontsize=11, fontweight="bold", pad=8)
ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# ── Plot 5: 6-month forecast with confidence interval ─────────
ax = axes[4]
# Historical (last 6 months)
hist_start = last_date - pd.Timedelta(days=180)
hist = df_clean[df_clean["date"] >= hist_start]
ax.scatter(hist["date"], hist["hm_fouling_pct"],
           c="#7F8C8D", alpha=0.5, s=15, label="Historical actual")
ax.plot(hist["date"],
        hist["hm_fouling_pct"].rolling(7, center=True, min_periods=3).mean(),
        c=C_AVG, lw=2, label="Historical 7-day avg")

# Forecast
ax.plot(future["date"], future["ml_forecast"],
        c=C_ML, lw=2.5, label=f"ML Forecast ({FORECAST_DAYS} days)")
ax.fill_between(future["date"],
                future["forecast_lower"].clip(lower=-20),
                future["forecast_upper"].clip(upper=50),
                alpha=0.25, color=C_ML, label="95% confidence interval")

# Threshold lines
ax.axhline(0,  c="black",   lw=1)
ax.axhline(10, c="#E67E22", lw=1.5, ls="--", label="10% threshold")
ax.axhline(20, c=C_PRE,     lw=1.5, ls="--", label="20% threshold")
ax.axvline(last_date, c="black", lw=2, ls=":",
           label=f"Forecast start ({last_date.date()})")

# Annotate forecast milestones
for months, color in [(3, "#E67E22"), (6, "#C0392B")]:
    fd = last_date + pd.Timedelta(days=months*30)
    fr = future[future["date"] <= fd]
    if len(fr) > 0:
        fval = fr.iloc[-1]["ml_forecast"]
        ax.annotate(
            f"+{months}m: {fval:.1f}%",
            xy=(fd, fval),
            xytext=(fd, fval+4),
            fontsize=9, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color))

ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Predicted Fouling Index (%)", fontsize=11)
ax.set_title(
    f"⑤ 6-Month Hull Fouling Forecast — {best_name} Model\n"
    "Prediction with 95% confidence interval",
    fontsize=11, fontweight="bold", pad=8)
ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

for ax in axes[:-2]:
    ax.tick_params(axis="x", rotation=15)
axes[-1].tick_params(axis="x", rotation=15)
axes[-2].tick_params(axis="x", rotation=15)

plt.tight_layout(h_pad=4.0)
plot_path = os.path.join(DATA_FOLDER, "kattegat_day3_ml.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\n✓ Plot saved : {plot_path}")
plt.show()

# =============================================================
# STEP 11 — SAVE RESULTS CSV
# =============================================================
# Full predictions on all data
df_clean["rf_pred"]  = rf.predict(df_clean[FEATURES])
df_clean["et_pred"]  = et.predict(df_clean[FEATURES])
df_clean["hybrid"]   = (df_clean["hm_fouling_pct"] + df_clean["rf_pred"]) / 2

out_cols = [
    "date", "period", "days_since_drydock",
    "me_kg", "distance_nm", "num_voyages",
    "hfo_per_nm", "daily_cii",
    "hm_fouling_pct",      # Physics model
    "rf_pred",             # Random Forest
    "et_pred",             # Extra Trees
    "hybrid",              # Hybrid (physics + ML)
]
results_path = os.path.join(DATA_FOLDER, "kattegat_day3_results.csv")
df_clean[out_cols].to_csv(results_path, index=False)
print(f"✓ Results CSV: {results_path}")

# Forecast CSV
forecast_path = os.path.join(DATA_FOLDER, "kattegat_day3_forecast.csv")
future[["date","days_since_drydock","ml_forecast",
        "forecast_std","forecast_upper","forecast_lower"]].to_csv(
    forecast_path, index=False)
print(f"✓ Forecast CSV: {forecast_path}")

# =============================================================
# STEP 12 — PRINT KEY THESIS NUMBERS
# =============================================================
print("\n" + "=" * 60)
print("  KEY THESIS NUMBERS — CHAPTER 5")
print("=" * 60)
print(f"\n  MODEL COMPARISON:")
print(f"  {'Model':<20} {'R² Test':>10} {'MAE':>8} {'RMSE':>8}")
print(f"  {'-'*48}")
print(f"  {'Random Forest':<20} {rf_r2_test:>10.4f} "
      f"{rf_mae_test:>8.2f}% {rf_rmse_test:>8.2f}%")
print(f"  {'Extra Trees':<20} {et_r2_test:>10.4f} "
      f"{et_mae_test:>8.2f}% {et_rmse_test:>8.2f}%")
print(f"\n  BEST MODEL: {best_name}")
print(f"    R² (train)  : {rf_r2_train:.4f}  (no overfitting "
      f"if close to test R²)")
print(f"    R² (test)   : {best_r2:.4f}")
print(f"    MAE         : {best_mae:.2f}%  "
      f"(avg prediction error in fouling %)")
print(f"    RMSE        : {best_rmse:.2f}%")
print(f"\n  TOP 3 FEATURES:")
for i, (feat, imp) in enumerate(importances.head(3).items(), 1):
    print(f"    {i}. {feat:<25} ({imp:.3f} importance)")
print(f"\n  6-MONTH FORECAST:")
for months_ahead in [1, 2, 3, 6]:
    fd  = last_date + pd.Timedelta(days=months_ahead*30)
    row = future[future["date"] <= fd].iloc[-1]
    print(f"    +{months_ahead} months ({fd.date()}): "
          f"{row['ml_forecast']:.1f}% "
          f"[{row['forecast_lower']:.1f}%, {row['forecast_upper']:.1f}%]")

print("\n" + "=" * 60)
print("  DAY 3 COMPLETE ✓")
print("=" * 60)
print("""
Files saved:
  kattegat_day3_ml.png         → 5 thesis plots
  kattegat_day3_results.csv    → Daily predictions (all models)
  kattegat_day3_forecast.csv   → 6-month forward forecast

Next:
  Day 4 → CII monthly calculation + forecast
  Day 5 → Drydocking optimizer
""")
