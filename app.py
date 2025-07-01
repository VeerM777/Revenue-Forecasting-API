from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Load model and features
model = joblib.load("final_weekly_model.pkl")
features = joblib.load("final_weekly_features.pkl")

# Create app
app = FastAPI(title="Revenue Forecasting API")

# Request schema
class RevenueInput(BaseModel):
    date: str  # "YYYY-MM-DD"
    product_category: str
    region: str
    agent_id: str
    marketing_spend: float
    lead_count: int

# Helper to get start of the week
def get_week_start(date):
    date = pd.to_datetime(date)
    return date - timedelta(days=date.weekday())

@app.post("/predict")
def predict_revenue(data: RevenueInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    input_df["date"] = pd.to_datetime(input_df["date"])
    input_df["week_start"] = input_df["date"].apply(get_week_start)

    # Create placeholder weekly aggregation
    weekly_df = input_df.groupby(["week_start", "product_category", "region", "agent_id"]).agg({
        "marketing_spend": "sum",
        "lead_count": "sum"
    }).reset_index()

    # Add engineered features
    weekly_df["marketing_per_lead"] = weekly_df["marketing_spend"] / (weekly_df["lead_count"] + 1e-5)
    weekly_df["week_num"] = weekly_df["week_start"].dt.isocalendar().week
    weekly_df["month"] = weekly_df["week_start"].dt.month
    weekly_df["region_freq"] = 0.05  # placeholder (can be precomputed from training data)
    weekly_df["product_cat_freq"] = 0.05

    # Lag & roll placeholders (set to 0 or impute later)
    for col in ['revenue_lag_1', 'revenue_lag_2', 'revenue_roll_mean_2', 'revenue_roll_mean_3',
                'marketing_lag_1', 'marketing_lag_2', 'leads_lag_1', 'leads_lag_2']:
        weekly_df[col] = 0.0

    # Select features
    X = weekly_df[features]

    # Predict weekly revenue
    pred_log = model.predict(X)
    pred_weekly = np.expm1(pred_log)[0]

    # Distribute evenly over 7 days (or you can use custom distribution logic)
    daily_pred = round(pred_weekly / 7, 2)

    return {
        "input_date": data.date,
        "predicted_revenue_for_that_day": daily_pred,
        "predicted_weekly_revenue": round(pred_weekly, 2)
    }
