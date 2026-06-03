# Agentic AI — Dynamic Tariff Optimization for EV Charging Networks

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F79231?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-2ecc71?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

**Open Project 2026 · Society of Business · IIT Roorkee**

*A three-agent AI framework for real-time EV charging demand prediction, dynamic tariff optimization, and continuous performance monitoring*

</div>

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Datasets](#datasets)
- [Key Findings](#key-findings)
- [Agents](#agents)
  - [Agent 1 — Demand Prediction](#agent-1--demand-prediction-agent)
  - [Agent 2 — Tariff Pricing](#agent-2--tariff-pricing-agent)
  - [Agent 3 — Monitoring & Learning](#agent-3--monitoring--learning-agent)
- [Model Comparison](#model-comparison)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Visualizations](#visualizations)
- [Limitations & Assumptions](#limitations--assumptions)
- [Author](#author)

---

## Overview

Traditional EV charging networks apply fixed tariffs regardless of real-time demand, leading to chronic underutilization of most stations while a small subset faces recurring congestion.

This project builds an **agentic AI framework** with three specialized agents that work in sequence:

1. **Demand Prediction Agent** — forecasts station-level utilization 1 hour ahead using a Random Forest model with engineered lag features
2. **Tariff Pricing Agent** — translates demand forecasts into dynamic tariffs using rule-based surge and discount logic
3. **Monitoring & Learning Agent** — tracks 144 pricing episodes and evaluates system performance across three operational metrics

The framework is trained and evaluated on two real-world datasets: **ACN-Data** (14,848 EV charging sessions, JPL Caltech) and **UrbanEV ST-EVCDP** (2.1M+ rows, 247 stations, Shenzhen, China).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENTIC AI FRAMEWORK                         │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   DEMAND     │    │   TARIFF     │    │   MONITORING &   │  │
│  │ PREDICTION   │───▶│   PRICING    │───▶│    LEARNING      │  │
│  │    AGENT     │    │    AGENT     │    │     AGENT        │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│         │                   │                     │             │
│  • Predicted util    • Dynamic ₹/kWh       • Pricing efficiency │
│  • Congestion prob   • Surge / discount    • Response rate      │
│  • Expected kWh      • Revenue gain        • Wait reduction     │
│                                                                 │
│  ◀─────────────── Feedback Loop (144 episodes) ───────────────▶ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Datasets

### ACN-Data (Adaptive Charging Network)
| Property | Value |
|---|---|
| Source | Caltech / JPL Site |
| Raw sessions | 16,304 |
| After cleaning | 14,848 |
| Coverage | 2018–2019 |
| Key fields | connectionTime, kWhDelivered, stationID, clusterID |

### UrbanEV ST-EVCDP (Shenzhen)
| Property | Value |
|---|---|
| Source | Intelligent Systems Lab, China |
| Rows (after integration) | 2,133,833 |
| Stations | 247 |
| Timestamps | 8,640 (5-min intervals, 30 days) |
| Coverage | June 2022 |
| Source files | 9 CSVs merged into unified base table |

> **Note:** The two datasets are geographically and temporally incompatible and are used complementarily — UrbanEV drives demand modeling and pricing; ACN provides session-level behavioral insights.

---

## Key Findings

| # | Finding | Implication |
|---|---|---|
| 1 | **62% of station-hours** are below 30% utilization | Off-peak demand activation is the primary lever, not surge control |
| 2 | **Demand peaks at 1am**, lowest at 4pm | Fleet/residential overnight charging pattern — not commercial |
| 3 | **Surge concentrated in 10 stations** out of 247 | Spatial redistribution more effective than blanket surge pricing |
| 4 | **CBD stations less utilized** than non-CBD (0.243 vs 0.293) | Location alone does not predict congestion |
| 5 | **30-day network trend is flat** (std = 0.009) | No growth signal; pricing must activate existing idle capacity |
| 6 | **Station-level CV ≈ 0.60** across all periods | High volatility at station level justifies real-time dynamic pricing |
| 7 | **Weekday idle time averages 2.69 hrs** (ACN) | Idle time penalty pricing can improve charger turnover |

---

## Agents

### Agent 1 — Demand Prediction Agent

Predicts station-level utilization **1 hour ahead** using a Random Forest regressor with engineered lag and rolling features.

#### Feature Engineering

| Feature | Description | Lookback |
|---|---|---|
| `util_lag1` | Utilization 1 step ago | 5 minutes |
| `util_lag12` | Utilization 12 steps ago | 1 hour |
| `util_rolling3` | Rolling 3-step mean | 15 minutes |
| `hour`, `day_of_week` | Temporal context | — |
| `count`, `fast_count`, `slow_count` | Station capacity | — |

#### Agent Outputs

| Output | Description |
|---|---|
| Predicted Utilization Rate | Continuous forecast (0–1) per station per slot |
| Congestion Probability | Soft score based on distance from 80% threshold |
| Expected Charging Load (kWh) | Predicted energy demand per station-slot |

#### Performance

| Metric | Value |
|---|---|
| RMSE | 0.0369 |
| MAE | 0.0187 |
| R² Score | **0.9559** |
| Test samples | 425,581 |
| Congestion flagged | 3,693 slots (0.87%) |
| Avg expected load | 11.00 kWh per slot |

---

### Agent 2 — Tariff Pricing Agent

Translates demand forecasts into dynamic tariffs using a three-zone rule-based pricing engine with demand elasticity modeling.

#### Pricing Rules

| Zone | Condition | Tariff | Multiplier | Elasticity |
|---|---|---|---|---|
| Surge | Utilization ≥ 80% | ₹22.50/kWh | 1.5× | −12% sessions |
| Normal | 30% < Util < 80% | ₹12.75–₹22.50/kWh | Linear | No change |
| Discount | Utilization ≤ 30% | ₹12.75/kWh | 0.85× | +35% sessions |

#### Results

| Metric | Value |
|---|---|
| Revenue Gain | **+11.12%** over fixed ₹15/kWh baseline |
| Network utilization before | 28.01% |
| Network utilization after | 31.59% (+3.57pp) |
| Off-Peak Session Uplift | +35.00% |
| Surge utilization before | 87.6% |
| Surge utilization after | 77.1% (−10.5pp) |
| Dynamic Rev/kWh | ₹16.67 vs ₹15.00 baseline |

---

### Agent 3 — Monitoring & Learning Agent

Evaluates pricing decisions across **144 episodes** (1 episode = 1 hour of network-wide decisions) and tracks three operational metrics over time.

#### Metrics Tracked

| Metric | Value | Benchmark |
|---|---|---|
| Pricing Efficiency Score | ₹16.67/kWh | ₹15.00 baseline |
| Customer Response Rate | 1.13× | 1.00 (no response) |
| Avg Wait Time Reduction | 0.105 utilization units | 0 (no reduction) |

All 144 episodes recorded pricing efficiency above the ₹15 baseline, confirming consistent system performance.

---

## Model Comparison

Random Forest was benchmarked against XGBoost and LightGBM under both default and tuned hyperparameters — 5 configurations total.

| Model | RMSE | MAE | R² | Train Time |
|---|---|---|---|---|
| **Random Forest** | **0.0369** | **0.0187** | **0.9559** | 81.7s |
| XGBoost (default) | 0.0508 | 0.0298 | 0.9166 | 3.9s |
| LightGBM (default) | 0.0520 | 0.0303 | 0.9126 | 3.7s |
| XGBoost (tuned, 1000 trees) | 0.0500 | 0.0293 | 0.9190 | 13.3s |
| LightGBM (tuned, 1000 trees) | 0.0524 | 0.0306 | 0.9111 | 12.2s |

**Why Random Forest wins:** The dominant feature `util_lag1` (88.6% importance) is a smooth, strongly autocorrelated signal. Random Forest's averaging of deep unpruned trees captures this optimally. XGBoost and LightGBM's regularization (shrinkage, subsampling, leaf penalties) constrains their ability to fit dominant continuous signals — even with 1000 estimators and tuned hyperparameters, the 3.7+ R² point gap remained.

---

## Results Summary

```
╔══════════════════════════════════════════════════════════╗
║        AGENTIC AI DYNAMIC TARIFF SYSTEM — RESULTS       ║
╠══════════════════════════════════════════════════════════╣
║  DEMAND PREDICTION AGENT                                 ║
║  ├── Algorithm       Random Forest (5-model comparison)  ║
║  ├── R² Score        0.9559                              ║
║  ├── RMSE            0.0369                              ║
║  └── Congestion      3,693 slots flagged (0.87%)         ║
╠══════════════════════════════════════════════════════════╣
║  TARIFF PRICING AGENT                                    ║
║  ├── Revenue Gain    +11.12% vs fixed ₹15/kWh            ║
║  ├── Utilization     28.01% → 31.59% (+3.57pp)           ║
║  ├── Off-Peak Uplift +35.00% session volume              ║
║  └── Surge Relief    87.6% → 77.1% (−10.5pp)            ║
╠══════════════════════════════════════════════════════════╣
║  MONITORING & LEARNING AGENT                             ║
║  ├── Episodes        144 (6 days of hourly decisions)    ║
║  ├── Efficiency      ₹16.67/kWh vs ₹15.00 baseline       ║
║  ├── Response Rate   1.13× (13% more sessions)           ║
║  └── Wait Reduction  0.105 utilization units             ║
╚══════════════════════════════════════════════════════════╝
```

---

## Project Structure

```
OP26-EV-Charging-Analytics/
│
├── data/
│   ├── acn_processed.csv              # Cleaned ACN session data (14,848 rows)
│   ├── acn_sessions_converted.csv     # ACN data converted from JSON/XLSX
│   ├── urban_processed.csv            # Unified UrbanEV base table (2.1M rows)
│   ├── information.csv                # Station metadata (247 stations)
│   ├── price.csv                      # Raw price matrix
│   └── stations.csv                   # Station coordinates
│
├── notebooks/
│   ├── 01_data_loading.ipynb          # Data ingestion, cleaning, feature engineering
│   ├── 02_EDA.ipynb                   # Exploratory data analysis (10 plots)
│   ├── 03_demand_model.ipynb          # Demand Prediction Agent + model comparison
│   ├── 04_pricing_agent.ipynb         # Tariff Pricing Agent + revenue simulation
│   └── 05_monitoring_agent.ipynb      # Monitoring & Learning Agent + final metrics
│
├── outputs/
│   ├── plot1_hourly_demand.png        # Hourly demand curve with pricing thresholds
│   ├── plot2_30day_trend.png          # 30-day network demand trend
│   ├── plot3_station_utilization.png  # Station utilization distribution
│   ├── plot4_weekday_weekend.png      # Weekday vs weekend demand pattern
│   ├── plot5_volatility.png           # Volatility across demand periods
│   ├── plot6_weekday_weekend_acn.png  # ACN session behavior analysis
│   ├── plot7_actual_vs_predicted.png  # Model Diagnostics - actual vs predicted
│   ├── plot8_pricing_curve.png        # Dynamic pricing curve
│   ├── plot9_pricing_results.png      # Revenue and utilization results
│   ├── plot10_monitoring.png          # Monitoring agent dashboard
│   ├── plot_model_comparison.png      # RF vs XGBoost vs LightGBM comparison
│   ├── demand_predictions.csv         # Agent 1 output — predictions + congestion (large file can't upload on github)
│   ├── pricing_output.csv             # Agent 2 output — tariffs + revenue        (large file can't upload on github)
│   └── final_metrics.csv              # Consolidated metrics across all agents
│ 
├── presentation/
│   └── OP26_Manas.pdf
│ 
└── .gitignore
```

---

## Setup & Usage

### Prerequisites

```bash
Python 3.10+
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib jupyter
```

### Installation

```bash
git clone https://github.com/HighOnKeys/OP26-EV-Charging-Analytics.git
cd OP26-EV-Charging-Analytics
pip install -r requirements.txt
```

### Running the Notebooks

Run notebooks in sequential order — each notebook depends on outputs from the previous:

```bash
jupyter notebook
```

| Order | Notebook | Purpose |
|---|---|---|
| 1 | `01_data_loading.ipynb` | Data ingestion and preprocessing |
| 2 | `02_EDA.ipynb` | Exploratory analysis and visualization |
| 3 | `03_demand_model.ipynb` | Demand Prediction Agent |
| 4 | `04_pricing_agent.ipynb` | Tariff Pricing Agent |
| 5 | `05_monitoring_agent.ipynb` | Monitoring Agent + final metrics |

> **Important:** Run notebooks from the `notebooks/` directory so relative paths (`../data/`, `../outputs/`) resolve correctly.

---

## Visualizations

| Plot | Description |
|---|---|
| Hourly Demand Curve | Network utilization by hour with surge/discount thresholds |
| Station Utilization Distribution | Distribution and top 20 most utilized stations |
| Weekday vs Weekend | Intraday demand pattern split by day type |
| 30-Day Demand Trend | Network-wide utilization over the full 30-day period |
| Volatility by Period | Peak vs shoulder vs off-peak utilization variance |
| Model Comparison | RMSE, MAE, R² across all 5 model configurations |
| Actual vs Predicted | Random Forest diagnostic scatter plot |
| Dynamic Pricing Curve | Tariff as a function of predicted utilization |
| Revenue Results | Baseline vs dynamic revenue by demand zone |
| Monitoring Dashboard | All 3 monitoring metrics tracked across 144 episodes |

---

## Limitations & Assumptions

**Assumptions:**
- Demand elasticity multipliers (+35% discount response, −12% surge deterrence) are assumed — not estimated from controlled experiments. A/B testing required for empirical validation
- Grid cost fixed at ₹6/kWh — actual procurement costs vary by time, contract, and location
- Revenue proxy uses utilization as a stand-in for billed kWh — directionally correct but not exact billing figures

**Limitations:**
- ACN (California) and UrbanEV (Shenzhen) datasets are geographically incompatible — used complementarily, not merged into a single model
- Monitoring agent feedback loop is simulated — real deployment requires live outcome data to retrain agents continuously
- Utilization occasionally exceeds 100% in UrbanEV data — data quality issue flagged, not corrected
- 30-day UrbanEV window may not capture seasonal demand variation
- Lag features create a cold-start dependency for the first 12 rows per station per day

> All findings are correlational unless explicitly stated otherwise. Causal language is avoided throughout.

---

## Author

**Kumar Manas**
B.Tech. Production and Industrial Engineering · IIT Roorkee

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kumarmanas-iitroorkee/)
[![GitHub](https://img.shields.io/badge/GitHub-121011?style=flat&logo=github&logoColor=white)](https://github.com/HighOnKeys)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:krmanas0811@gmail.com)

---

<div align="center">
<i>Built for Open Project 2026 · Society of Business · IIT Roorkee</i>
</div>
