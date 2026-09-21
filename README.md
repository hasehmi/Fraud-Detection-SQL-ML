# Fraud Detection — SQL Feature Engineering + Explainable ML

Flags fraudulent transactions using behavioral features engineered directly in **PostgreSQL** (via SQL window functions), then modeled with XGBoost and explained with SHAP.

## Why this project

This project exists specifically to demonstrate real SQL work, not just Python/ML — most entry-level ML portfolios skip straight from a CSV to a model. Here, the feature engineering that actually drives the model's performance is written as SQL against a proper relational schema:

- **A normalized 3-table schema** (`customers`, `merchants`, `transactions`) with foreign keys — not one flat pre-joined CSV.
- **Window-function features** — rolling transaction counts (`COUNT(*) OVER ... RANGE BETWEEN INTERVAL`), a customer's own historical average computed without leaking future data (`ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`), and time-since-last-transaction (`LAG`).
- **Verified, not assumed, in SQL first** — before any model is trained, a SQL query confirms the engineered features actually separate fraud from legitimate transactions.
- **Confirmed by SHAP** — the top model features are the SQL-engineered ones, not the raw transaction amount, which is the evidence that the feature engineering effort was the right call.
- **Tested and deployable, not just a notebook** — the feature logic has 33 unit tests (`pytest`), and the trained model is served through a containerized REST API (FastAPI + Docker), so it can be queried live instead of only living inside a notebook.

## Project structure

```
├── sql/
│   ├── 01_schema.sql              # Table definitions (customers, merchants, transactions)
│   └── 02_feature_engineering.sql # Window-function feature engineering (the core SQL work)
├── notebooks/
│   └── fraud_detection.ipynb      # Full pipeline: SQL → Python → XGBoost → SHAP
├── src/
│   ├── features.py                # Testable Python versions of the SQL feature logic
│   └── api.py                     # FastAPI scoring service (loads the trained model)
├── tests/
│   └── test_features.py           # 33 unit tests covering feature logic and edge cases
├── models/                         # Trained model artifacts, versioned so the API runs standalone
│   ├── fraud_model.pkl
│   ├── scaler.pkl
│   ├── model_columns.pkl
│   └── best_threshold.pkl
├── data/
│   ├── customers.csv
│   ├── merchants.csv
│   └── transactions.csv
├── generate_data.py                # Regenerates the synthetic dataset (optional — data/ already has it)
├── Dockerfile                      # Containerizes the scoring API
├── requirements.txt                # Full dependencies (notebook + API + tests)
└── requirements-api.txt            # Minimal dependencies for the Docker container
```

## Dataset

A synthetic-but-behaviorally-realistic dataset (800 customers, 150 merchants, ~20,500 transactions over 90 days, ~1.5% fraud rate — a realistic imbalance, not inflated for the demo). Fraud is **not** separable by any single column; it's embedded as combinations of signals (transaction bursts, amounts far above a customer's own norm, odd-hour activity, transactions away from home city) — the same kind of patterns SQL window functions are built to compute.

## Setup

**1. PostgreSQL must be running**, with a database created:
```bash
createdb fraud_detection
```

**2. Load the schema and data:**
```bash
psql -d fraud_detection -f sql/01_schema.sql

psql -d fraud_detection -c "\COPY customers FROM 'data/customers.csv' WITH CSV HEADER"
psql -d fraud_detection -c "\COPY merchants FROM 'data/merchants.csv' WITH CSV HEADER"
psql -d fraud_detection -c "\COPY transactions FROM 'data/transactions.csv' WITH CSV HEADER"

psql -d fraud_detection -f sql/02_feature_engineering.sql
```

**3. Update the database connection** in the notebook if your PostgreSQL username/password differ from the default (`postgres`/`postgres`):
```python
engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5432/fraud_detection")
```

**4. Install Python dependencies and run:**
```bash
pip install -r requirements.txt
jupyter notebook notebooks/fraud_detection.ipynb
```

## Approach

1. **Schema design** — normalized relational tables mirroring how transaction data is actually stored in production.
2. **SQL feature engineering** — a materialized view computing rolling transaction velocity, a customer's own historical spending average (leak-free), amount-vs-own-average ratio, time since last transaction, and behavioral flags (odd hour, away from home city, unusual merchant category).
3. **SQL-side validation** — confirming the engineered features separate fraud from legitimate transactions *before* any Python modeling.
4. **Modeling** — Logistic Regression baseline vs. XGBoost, trained on SMOTE-balanced data (test set kept at natural imbalance).
5. **Threshold tuning** — optimizing the decision cutoff for the fraud class rather than using the default 0.5.
6. **Explainability** — SHAP confirms the SQL-engineered features are what actually drive the model.

## Results

| Model | ROC-AUC | Fraud F1 |
|---|---|---|
| Logistic Regression (baseline) | 0.924 | 0.72 |
| XGBoost | 0.980 | 0.82 |
| XGBoost, tuned threshold | — | 0.86 (96% precision, 79% recall) |

**SQL-side validation of the engineered features**, before any model was trained:

| | avg txns/1h | avg amount ratio | % away from home | % odd hour |
|---|---|---|---|---|
| Legitimate | 1.02 | 1.00 | 0.0% | 2.2% |
| **Fraud** | **2.74** | **3.47** | **14.7%** | **40.5%** |

## Testing

The feature-engineering logic in `src/features.py` mirrors what's computed in SQL — each function has a matching set of unit tests, including edge cases (a customer's first-ever transaction, division-by-zero guards, out-of-range hours, case-insensitive city matching).

```bash
pip install -r requirements.txt
pytest tests/ -v
```

33 tests, covering schema validation, each engineered feature, and the decision-threshold logic.

## Scoring API + Docker

The trained model is also served as a small REST API (`src/api.py`, built with FastAPI), so it can be queried live rather than only used inside a notebook.

**Run locally:**
```bash
uvicorn src.api:app --reload --port 8000
```

**Or run it in Docker** (no local Python setup needed at all):
```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

**Example request:**
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "amount": 2500.0,
  "timestamp": "2026-03-15T03:15:00",
  "merchant_category": "electronics",
  "city": "Karachi",
  "customer_home_city": "Lahore",
  "customer_avg_spend": 50.0,
  "customer_favorite_category": "grocery",
  "customer_risk_profile": "high",
  "customer_rolling_avg_amount": 48.0,
  "txns_last_24h": 5,
  "txns_last_1h": 4
}'
```
```json
{"fraud_probability": 0.9754, "decision": "Fraud", "threshold_used": 0.9489}
```

## What I'd improve with more time

- Add a graph-based feature (shared devices/IPs between customers) — a common real-world fraud signal not modeled here.
- Test on a real anonymized fraud dataset (e.g. IEEE-CIS) to see whether these engineered feature *types* transfer.
- Move the SQL feature view from a materialized view (manually refreshed) to an incremental/streaming computation, since a production fraud system needs to score transactions within milliseconds, not after a batch refresh.

## Author

Ans Tanveer Hashmi — BS Data Science, MNS University of Agriculture, Multan.
[LinkedIn] · [GitHub]
