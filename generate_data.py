"""
Generate a synthetic but behaviorally realistic transactions dataset for fraud detection.

Design goals:
- Fraud is NOT trivially separable by one column (unlike the earlier logo-detection project) —
  it emerges from PATTERNS across time (velocity, odd hours, amount spikes relative to a
  customer's own history), which is exactly what SQL window functions are good at extracting.
- Small enough to run comfortably on an 8GB RAM laptop with no GPU.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N_CUSTOMERS = 800
N_MERCHANTS = 150
N_DAYS = 90
START = datetime(2026, 1, 1)

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "restaurant", "fuel", "clothing",
    "pharmacy", "online_retail", "travel", "entertainment", "utilities"
]

# ---------------------------------------------------------
# Customers: each has a "home" spending profile (avg amount, favorite category, home city)
# ---------------------------------------------------------
CITIES = ["Lahore", "Karachi", "Islamabad", "Faisalabad", "Multan", "Rawalpindi"]

customers = []
for cid in range(1, N_CUSTOMERS + 1):
    customers.append({
        "customer_id": cid,
        "home_city": random.choice(CITIES),
        "avg_spend": np.random.gamma(shape=3, scale=15),  # most customers spend modestly
        "favorite_category": random.choice(MERCHANT_CATEGORIES),
        "risk_profile": np.random.choice(["low", "low", "low", "medium", "high"]),  # most are low risk
    })
customers_df = pd.DataFrame(customers)

merchants = []
for mid in range(1, N_MERCHANTS + 1):
    merchants.append({
        "merchant_id": mid,
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "city": random.choice(CITIES),
    })
merchants_df = pd.DataFrame(merchants)

# ---------------------------------------------------------
# Transactions: each customer makes a random number of transactions over N_DAYS,
# mostly near their own average spend, in their own city, during normal hours.
# ---------------------------------------------------------
transactions = []
txn_id = 1

for _, cust in customers_df.iterrows():
    n_txns = np.random.poisson(lam=25)  # ~25 transactions per customer over 90 days
    for _ in range(n_txns):
        day_offset = random.randint(0, N_DAYS - 1)
        hour = int(np.clip(np.random.normal(loc=14, scale=4), 0, 23))  # normal daytime activity
        minute = random.randint(0, 59)
        ts = START + timedelta(days=day_offset, hours=hour, minutes=minute)

        merchant = merchants_df.sample(1).iloc[0]
        amount = max(1.0, np.random.normal(loc=cust["avg_spend"], scale=cust["avg_spend"] * 0.3))

        transactions.append({
            "txn_id": txn_id,
            "customer_id": cust["customer_id"],
            "merchant_id": merchant["merchant_id"],
            "amount": round(amount, 2),
            "timestamp": ts,
            "city": cust["home_city"],  # transaction happens in customer's home city normally
            "is_fraud": 0,
        })
        txn_id += 1

txns_df = pd.DataFrame(transactions)

# ---------------------------------------------------------
# Inject fraud patterns — these are the kinds of signals a real fraud team looks for.
# Each pattern is deliberately a COMBINATION of factors, not one column, so a model
# has to learn real structure rather than memorize a single feature.
# ---------------------------------------------------------
fraud_rows = []

# Pattern 1: Card testing — rapid burst of small transactions in a short time window
for _ in range(35):
    cust = customers_df.sample(1).iloc[0]
    day_offset = random.randint(0, N_DAYS - 1)
    base_ts = START + timedelta(days=day_offset, hours=random.randint(0, 23))
    burst_size = random.randint(4, 8)
    for i in range(burst_size):
        merchant = merchants_df.sample(1).iloc[0]
        ts = base_ts + timedelta(minutes=random.randint(1, 3) * i)
        fraud_rows.append({
            "txn_id": txn_id,
            "customer_id": cust["customer_id"],
            "merchant_id": merchant["merchant_id"],
            "amount": round(np.random.uniform(1, 15), 2),  # small "testing" amounts
            "timestamp": ts,
            "city": cust["home_city"],
            "is_fraud": 1,
        })
        txn_id += 1

# Pattern 2: Sudden high-value transaction far above the customer's own norm, at odd hours
for _ in range(60):
    cust = customers_df.sample(1).iloc[0]
    day_offset = random.randint(0, N_DAYS - 1)
    odd_hour = random.choice([1, 2, 3, 4, 23])
    ts = START + timedelta(days=day_offset, hours=odd_hour, minutes=random.randint(0, 59))
    merchant = merchants_df.sample(1).iloc[0]
    amount = cust["avg_spend"] * np.random.uniform(8, 20)  # far above normal
    fraud_rows.append({
        "txn_id": txn_id,
        "customer_id": cust["customer_id"],
        "merchant_id": merchant["merchant_id"],
        "amount": round(amount, 2),
        "timestamp": ts,
        "city": cust["home_city"],
        "is_fraud": 1,
    })
    txn_id += 1

# Pattern 3: Transaction in a city far from the customer's home city (impossible-travel-ish),
# combined with an above-average amount
for _ in range(45):
    cust = customers_df.sample(1).iloc[0]
    other_cities = [c for c in CITIES if c != cust["home_city"]]
    foreign_city = random.choice(other_cities)
    day_offset = random.randint(0, N_DAYS - 1)
    ts = START + timedelta(days=day_offset, hours=random.randint(0, 23), minutes=random.randint(0, 59))
    merchant = merchants_df.sample(1).iloc[0]
    amount = cust["avg_spend"] * np.random.uniform(2, 6)
    fraud_rows.append({
        "txn_id": txn_id,
        "customer_id": cust["customer_id"],
        "merchant_id": merchant["merchant_id"],
        "amount": round(amount, 2),
        "timestamp": ts,
        "city": foreign_city,
        "is_fraud": 1,
    })
    txn_id += 1

fraud_df = pd.DataFrame(fraud_rows)
full_df = pd.concat([txns_df, fraud_df], ignore_index=True)
full_df = full_df.sort_values("timestamp").reset_index(drop=True)

print(f"Total transactions: {len(full_df)}")
print(f"Fraud transactions: {full_df['is_fraud'].sum()} ({full_df['is_fraud'].mean()*100:.2f}%)")

customers_df.to_csv("data/customers.csv", index=False)
merchants_df.to_csv("data/merchants.csv", index=False)
full_df.to_csv("data/transactions.csv", index=False)
print("Saved: data/customers.csv, data/merchants.csv, data/transactions.csv")
