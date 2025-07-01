📊 Revenue Forecasting API (Weekly Model)
This API predicts weekly revenue using aggregated inputs and estimates daily revenue for any given date.

🔧 Model Info
✅ Trained on weekly data (Mon–Sun)

✅ Inputs like marketing_spend and lead_count must be weekly totals

✅ Accurate for both weekly and daily estimates


📥 Input Format

{
  "date": "2025-03-29",  // Any date from the target week
  "product_category": "Cosmetic",
  "region": "North",
  "agent_id": "A4",
  "marketing_spend": 21000,  // Total for the week
  "lead_count": 170          // Total for the week
}


{
  "input_date": "2025-03-29",
  "predicted_revenue_for_that_day": 16874.88,
  "predicted_weekly_revenue": 118124.17
}
