# Data for ML-Investment-Readiness

This folder contains the datasets used in the **Machine Learning–Driven Startup Investment Readiness** project.

## Files
- **synthetic_startups.csv** — Synthetic benchmark dataset with 15 engineered features + label for readiness.
- **sdg_labels.json** — Mapping of UN Sustainable Development Goals (SDGs) to be used in the `Sustainability_Alignment` feature.

## Schema
| Column                     | Type   | Description |
|----------------------------|--------|-------------|
| Total_Funding_USD_M        | float  | Total funding in millions USD |
| Burn_Rate_USD_M            | float  | Monthly burn rate in millions USD |
| Runway_Months              | float  | Operational runway in months |
| Revenue_USD_M              | float  | Annual revenue in millions USD |
| Revenue_Growth_Rate        | float  | Year-on-year revenue growth rate |
| Net_Profit_Margin          | float  | Net profit margin |
| CAC                        | float  | Customer acquisition cost |
| LTV                        | float  | Customer lifetime value |
| LTV_CAC_Ratio              | float  | Ratio of LTV to CAC |
| ARR_USD_M                  | float  | Annual recurring revenue in millions USD |
| Churn_Rate                 | float  | Monthly churn rate |
| Novelty_Score              | float  | SBERT-based novelty score |
| Sustainability_Alignment   | int    | 1 if aligned with SDGs, else 0 |
| Sentiment_Score            | float  | FinBERT sentiment score |
| Social_Buzz_Score          | float  | Normalized measure of online buzz |
| label                      | int    | 1 = investment ready, 0 = not ready |

## Notes
- This synthetic dataset is generated for benchmarking (~80% accuracy, F1 ~0.77) and **does not** represent real startups.
- Replace with real data in the same schema for production use.
