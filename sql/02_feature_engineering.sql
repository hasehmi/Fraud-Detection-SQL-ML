-- Feature Engineering for Fraud Detection
-- Everything here is computed IN SQL, using window functions — this is the kind of
-- feature engineering a real fraud team runs directly against a transactions database,
-- rather than pulling raw rows into pandas and doing it there.

DROP MATERIALIZED VIEW IF EXISTS transaction_features;

CREATE MATERIALIZED VIEW transaction_features AS
WITH base AS (
    SELECT
        t.txn_id,
        t.customer_id,
        t.merchant_id,
        t.amount,
        t.timestamp,
        t.city,
        t.is_fraud,
        c.home_city,
        c.avg_spend        AS customer_avg_spend,
        c.favorite_category,
        c.risk_profile,
        m.merchant_category,
        EXTRACT(HOUR FROM t.timestamp)::INT AS txn_hour,
        EXTRACT(DOW FROM t.timestamp)::INT  AS txn_day_of_week
    FROM transactions t
    JOIN customers c ON t.customer_id = c.customer_id
    JOIN merchants m ON t.merchant_id = m.merchant_id
),
windowed AS (
    SELECT
        *,

        -- Velocity: how many transactions has this customer made in the last 24 hours,
        -- INCLUDING the current one? A burst of activity is one of the strongest fraud signals.
        COUNT(*) OVER (
            PARTITION BY customer_id
            ORDER BY timestamp
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
        ) AS txns_last_24h,

        -- Same idea over a tighter 1-hour window, to catch rapid "card testing" bursts
        COUNT(*) OVER (
            PARTITION BY customer_id
            ORDER BY timestamp
            RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
        ) AS txns_last_1h,

        -- Rolling average of this customer's OWN past spending, computed only from
        -- transactions strictly before this one — never leaks the current or future
        -- transactions into the feature, which matters for a model to be trustworthy.
        AVG(amount) OVER (
            PARTITION BY customer_id
            ORDER BY timestamp
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS customer_rolling_avg_amount,

        -- Minutes since this customer's previous transaction (NULL for their first ever)
        EXTRACT(EPOCH FROM (
            timestamp - LAG(timestamp) OVER (PARTITION BY customer_id ORDER BY timestamp)
        )) / 60.0 AS minutes_since_last_txn

    FROM base
)
SELECT
    txn_id,
    customer_id,
    merchant_id,
    amount,
    timestamp,
    txn_hour,
    txn_day_of_week,
    city,
    home_city,
    (city != home_city)::INT AS is_away_from_home_city,
    merchant_category,
    favorite_category,
    (merchant_category != favorite_category)::INT AS is_unusual_category,
    risk_profile,
    txns_last_24h,
    txns_last_1h,
    COALESCE(customer_rolling_avg_amount, customer_avg_spend) AS customer_rolling_avg_amount,

    -- The single most important engineered feature: how far above this customer's OWN
    -- typical spend is this transaction? A flat "amount" column can't capture this —
    -- 5,000 is normal for one customer and wildly abnormal for another.
    ROUND(
        (amount / NULLIF(COALESCE(customer_rolling_avg_amount, customer_avg_spend), 0))::NUMERIC,
        2
    ) AS amount_vs_own_avg_ratio,

    COALESCE(minutes_since_last_txn, 999999) AS minutes_since_last_txn,
    (txn_hour BETWEEN 1 AND 5)::INT AS is_odd_hour,
    is_fraud

FROM windowed
ORDER BY timestamp;

CREATE INDEX idx_features_customer ON transaction_features (customer_id);

-- Quick sanity check: fraud rate should look meaningfully different across these
-- engineered features, which is what justifies including them in the model.
SELECT
    is_fraud,
    ROUND(AVG(txns_last_1h)::NUMERIC, 2)              AS avg_txns_last_1h,
    ROUND(AVG(amount_vs_own_avg_ratio)::NUMERIC, 2)    AS avg_amount_ratio,
    ROUND(AVG(is_away_from_home_city)::NUMERIC, 3)     AS pct_away_from_home,
    ROUND(AVG(is_odd_hour)::NUMERIC, 3)                AS pct_odd_hour
FROM transaction_features
GROUP BY is_fraud;
