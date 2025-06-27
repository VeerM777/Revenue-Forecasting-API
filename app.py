# ✅ FastAPI for Raw Input-Based Weekly Revenue Forecasting
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

# Load trained model and features
model = joblib.load(r"D:\forecating of sales\final_model2.pkl")
features = joblib.load(r"D:\forecating of sales\final_features2.pkl")

# Load dataset safely
data_path = r"D:\forecating of sales\dummy_revenue_forecasting_data.csv"
df = pd.read_csv(data_path)

# --- Clean & Prepare Data ---
df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
df = df[df['date'].notna()]
df = df[df['date'].dt.year == 2025]

# Weekly aggregation
df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)

weekly_df = df.groupby(['week', 'agent_id', 'product_category', 'region']).agg({
    'marketing_spend': 'sum',
    'lead_count': 'sum',
    'revenue': 'sum'
}).reset_index()

# Ensure all numeric
for col in ['revenue', 'marketing_spend', 'lead_count']:
    weekly_df[col] = pd.to_numeric(weekly_df[col], errors='coerce')
weekly_df.dropna(subset=['revenue', 'marketing_spend', 'lead_count'], inplace=True)

# Frequency & mean encodings
product_freq = weekly_df['product_category'].value_counts(normalize=True).to_dict()
region_freq = weekly_df['region'].value_counts(normalize=True).to_dict()
agent_freq = weekly_df['agent_id'].value_counts(normalize=True).to_dict()

product_mean = weekly_df.groupby('product_category')['revenue'].mean().to_dict()
region_mean = weekly_df.groupby('region')['revenue'].mean().to_dict()
agent_mean = weekly_df.groupby('agent_id')['revenue'].mean().to_dict()

# --- FastAPI Setup ---
app = FastAPI()

class RevenueRequest(BaseModel):
    date: str  # YYYY-MM-DD
    product_category: str
    region: str
    agent_id: str
    marketing_spend: float
    lead_count: float

@app.post("/predict")
def predict_revenue(req: RevenueRequest):
    try:
        input_date = pd.to_datetime(req.date)
    except:
        return {"error": "Invalid date format. Use YYYY-MM-DD."}

    week_start = input_date.to_period('W').start_time

    # History for lag feature generation
    hist = weekly_df[(weekly_df['agent_id'] == req.agent_id)].sort_values('week')
    recent = hist[hist['week'] < week_start].tail(3)

    if len(recent) < 3:
        return {"error": "Not enough historical data for this agent_id to compute lag features."}

    row = {
        'marketing_spend': req.marketing_spend,
        'lead_count': req.lead_count,
        'revenue_lag_1': recent.iloc[-1]['revenue'],
        'revenue_lag_2': recent.iloc[-2]['revenue'],
        'revenue_lag_3': recent.iloc[-3]['revenue'],
        'revenue_roll_mean_3': recent['revenue'].mean(),
        'marketing_lag_1': recent.iloc[-1]['marketing_spend'],
        'marketing_lag_2': recent.iloc[-2]['marketing_spend'],
        'marketing_lag_3': recent.iloc[-3]['marketing_spend'],
        'leads_lag_1': recent.iloc[-1]['lead_count'],
        'leads_lag_2': recent.iloc[-2]['lead_count'],
        'leads_lag_3': recent.iloc[-3]['lead_count'],
        'weekofyear': week_start.isocalendar()[1],
        'month': week_start.month,
        'product_category_freq': product_freq.get(req.product_category, 0),
        'region_freq': region_freq.get(req.region, 0),
        'agent_id_freq': agent_freq.get(req.agent_id, 0),
        'product_category_mean_rev': product_mean.get(req.product_category, 0),
        'region_mean_rev': region_mean.get(req.region, 0),
        'agent_id_mean_rev': agent_mean.get(req.agent_id, 0)
    }

    X = pd.DataFrame([row])[features]
    log_pred = model.predict(X)[0]
    forecast = float(np.expm1(log_pred))

    return {"forecasted_revenue": round(forecast, 2)}
