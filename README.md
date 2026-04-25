# Customer Churn Prediction + Business Strategy

End-to-end ML pipeline that predicts customer churn, explains why with SHAP, recommends retention strategies, and quantifies revenue impact.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample data
python scripts/generate_sample_data.py

# 3. Run the full pipeline (train + score + explain + export)
python main.py --input data/customers.csv

# 4. Launch the Streamlit app
streamlit run app.py
```

## Project Structure

```
├── config/                  # YAML configuration files
│   ├── schema.yaml          # Column definitions and encoding strategy
│   ├── thresholds.yaml      # Risk segment thresholds and retention params
│   └── strategies.yaml      # (segment, driver) → strategy mappings
├── src/                     # Core modules
│   ├── ingestion.py         # CSV loading and validation
│   ├── preprocessing.py     # Feature engineering pipeline
│   ├── training.py          # Model training and evaluation
│   ├── scoring.py           # Churn probability scoring
│   ├── explainability.py    # SHAP explanations
│   ├── strategy.py          # Retention strategy recommendations
│   ├── impact.py            # Revenue impact analysis
│   ├── persistence.py       # Artifact save/load
│   └── dashboard_export.py  # Dashboard-ready data exports
├── tests/                   # Unit and property-based tests
├── data/                    # Input CSV files
├── artifacts/               # Versioned model outputs
├── scripts/                 # Utility scripts
│   └── generate_sample_data.py
├── main.py                  # CLI pipeline entry point
├── app.py                   # Streamlit web application
└── requirements.txt
```

## Required CSV Format

Your input CSV must contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | string/int | Unique customer identifier |
| `churn` | 0 or 1 | Target label (1 = churned) |
| `tenure_months` | numeric | Months as a customer |
| `monthly_charges` | numeric | Monthly billing amount |
| `total_charges` | numeric | Lifetime charges |
| `num_support_tickets` | numeric | Support ticket count |
| `annual_revenue` | numeric | Annual revenue from customer |
| `age` | numeric | Customer age |
| `gender` | categorical | Male/Female |
| `location` | categorical | City name |
| `contract_type` | categorical | Month-to-month, One year, Two year |
| `payment_method` | categorical | Credit card, Bank transfer, etc. |

## CLI Usage

```bash
# Full training pipeline
python main.py --input data/customers.csv

# Custom config and output directories
python main.py --input data/customers.csv --config-dir config --output-dir artifacts

# Inference-only mode (load existing model, skip training)
python main.py --input data/new_customers.csv --model-version artifacts/20260423_221425
```

## Streamlit App

The app supports two modes:
- **Upload CSV** — upload your own customer data file
- **Demo Mode** — uses the preloaded `data/customers.csv` sample dataset

## Pipeline Output

Each run creates a timestamped directory in `artifacts/` containing:
- `model.joblib` — trained model
- `pipeline.joblib` — preprocessing pipeline
- `metadata.json` — model type, metrics, feature list
- `feature_config.json` — column configuration
- `dashboards/` — exports for Power BI / Tableau:
  - `customer_summary.csv` — per-customer scores and strategies
  - `feature_importance.csv` — global SHAP feature rankings
  - `impact_summary.json` — revenue impact numbers
  - `shap_summary.png` — beeswarm plot

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, XGBoost (ML)
- SHAP (explainability)
- Streamlit (web app)
- Matplotlib (plots)
- Hypothesis, pytest (testing)
