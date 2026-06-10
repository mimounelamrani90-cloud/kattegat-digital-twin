# Kattegat Digital Twin — Hull Performance Monitoring

Master Thesis in Marine Engineering — 2025

## Overview

A seven-layer hybrid Digital Twin framework for hull fouling
detection and drydocking optimisation, applied to M/V Kattegat
(DFDS, Algeciras–Tanger Med route).

## Requirements

Python 3.10+
pip install pandas numpy scikit-learn matplotlib streamlit plotly

## Run Order

python3 kattegat_day1.py    # Data acquisition and processing
python3 kattegat_day2.py    # Holtrop-Mennen physics baseline
python3 kattegat_day3.py    # Machine learning model
python3 kattegat_day4.py    # IMO CII compliance module
python3 kattegat_day5.py    # Drydocking optimiser
streamlit run kattegat_day6_dashboard.py  # Dashboard

## Key Results

- Pre-drydock fouling penalty: 12.1%
- ML model R²: 0.643, MAE: 2.40%
- Optimal drydocking: December 2026
- Minimum lifecycle cost: USD 2,309,202

## Data Note
The raw voyage database (2026_mim.xlsx) contains commercially
sensitive DFDS operational data and is not included in this
repository. The processed output file (kattegat_daily_fixed.csv)
is included for reproducibility.
