#  Revenue Forecasting + API Deployment 

This project forecasts weekly revenue for different agents, products, and regions based on historical data. It uses machine learning models (LightGBM, CatBoost, XGBoost) and deploys a prediction API using FastAPI.

---

##  Problem Statement

**Goal:** Predict weekly revenue for the next 90 days using the following input features:

- `Date`
- `Product Category`
- `Region`
- `Agent ID`
- `Marketing Spend`
- `Lead Count`

---

##  Dataset

The dataset (`dummy_revenue_forecasting_data.csv`) contains daily records of sales and agent activity. It is cleaned and aggregated to weekly level for better forecasting accuracy.

---

## Features Used for Training

The following engineered features are used in the model:

- Current Week: `marketing_spend`, `lead_count`
- Lagged Revenue: `revenue_lag_1`, `revenue_lag_2`, `revenue_lag_3`
- Rolling Revenue Mean: `revenue_roll_mean_3`
- Lagged Marketing & Leads: `marketing_lag_*`, `leads_lag_*`
- Time Features: `weekofyear`, `month`
- Frequency Encoding: `product_category_freq`, `region_freq`, `agent_id_freq`
- Revenue Means: `*_mean_rev` for each category

---

## 🧠 Model & Training

- Model: `StackingRegressor` (CatBoost, LightGBM, XGBoost)
- Log-transformed target (`log1p(revenue)`) for stability
- Training split: 80% train / 20% test (time-based)
- Evaluation metrics:
  - ✅ MAE: ~34,000
  - ✅ R² Score: 0.97
  - ✅ MAPE: 2.9%

---

## 🚀 FastAPI Deployment

### Endpoint: `/predict`

**Request Format:**
```json
{
  "date": "2025-07-01",
  "product_category": "FMCG",
  "region": "North",
  "agent_id": "A123",
  "marketing_spend": 12000,
  "lead_count": 350
}
