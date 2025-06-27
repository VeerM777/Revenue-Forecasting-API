# Revenue-Forecasting-API
This project is a machine learning-powered forecasting pipeline designed to predict weekly revenue for various agents, product categories, and regions based on historical data.

It includes:

🧮 Model Training (with lag/rolling features and stacking)

🌐 A deployed FastAPI /predict endpoint

✅ Support for real-time revenue forecasting

Example input format: 
{
  "date": "2025-07-01",
  "product_category": "Cosmetic",
  "region": "North",
  "agent_id": "A4",
  "marketing_spend": 21000,
  "lead_count": 170
}
