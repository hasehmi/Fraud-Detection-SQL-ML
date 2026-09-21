"""
Feature engineering utilities for the fraud detection pipeline.

These mirror the logic computed in SQL (sql/02_feature_engineering.sql) but as
pure, testable Python functions — used both by unit tests and by the live
scoring script (score.py), so the same rules that were validated in SQL are
guaranteed to be applied consistently at prediction time.
"""
from datetime import datetime
from typing import Optional

REQUIRED_TRANSACTION_FIELDS = {
    "amount", "timestamp", "merchant_category", "city",
}
REQUIRED_CUSTOMER_FIELDS = {
    "home_city", "avg_spend", "favorite_category", "risk_profile",
}


class SchemaValidationError(ValueError):
    """Raised when an incoming transaction or customer record is missing required fields."""
    pass


def validate_transaction(transaction: dict) -> None:
    """Raise SchemaValidationError if a transaction dict is missing required fields."""
    missing = REQUIRED_TRANSACTION_FIELDS - transaction.keys()
    if missing:
        raise SchemaValidationError(f"Transaction is missing required fields: {sorted(missing)}")
    if transaction["amount"] is None or transaction["amount"] < 0:
        raise SchemaValidationError(f"Transaction amount must be non-negative, got {transaction['amount']!r}")


def validate_customer(customer: dict) -> None:
    """Raise SchemaValidationError if a customer dict is missing required fields."""
    missing = REQUIRED_CUSTOMER_FIELDS - customer.keys()
    if missing:
        raise SchemaValidationError(f"Customer is missing required fields: {sorted(missing)}")
    if customer["avg_spend"] is not None and customer["avg_spend"] < 0:
        raise SchemaValidationError(f"Customer avg_spend must be non-negative, got {customer['avg_spend']!r}")


def compute_amount_vs_own_avg_ratio(amount: float, customer_rolling_avg: float) -> float:
    """
    How many times larger is this transaction than the customer's own historical average?
    Mirrors the SQL: amount / NULLIF(customer_rolling_avg_amount, 0)

    A customer with no prior history (rolling avg of 0 or None) can't have a meaningful
    ratio yet, so we return 1.0 (neutral) rather than dividing by zero or crashing —
    the same NULLIF-guard behavior as the SQL query.
    """
    if customer_rolling_avg is None or customer_rolling_avg == 0:
        return 1.0
    return round(amount / customer_rolling_avg, 2)


def is_odd_hour(hour: int) -> bool:
    """Mirrors the SQL: txn_hour BETWEEN 1 AND 5"""
    if not (0 <= hour <= 23):
        raise ValueError(f"hour must be between 0 and 23, got {hour}")
    return 1 <= hour <= 5


def is_away_from_home_city(transaction_city: str, customer_home_city: str) -> bool:
    """Mirrors the SQL: city != home_city"""
    return transaction_city.strip().lower() != customer_home_city.strip().lower()


def is_unusual_category(merchant_category: str, favorite_category: str) -> bool:
    """Mirrors the SQL: merchant_category != favorite_category"""
    return merchant_category.strip().lower() != favorite_category.strip().lower()


def minutes_since(current_ts: datetime, previous_ts: Optional[datetime]) -> float:
    """
    Mirrors the SQL LAG()-based feature. Returns a large sentinel value (999999,
    matching the SQL's COALESCE fallback) when there's no previous transaction —
    e.g. a customer's very first transaction — rather than returning None, which
    would break a numeric feature vector.
    """
    if previous_ts is None:
        return 999999.0
    if current_ts < previous_ts:
        raise ValueError("current_ts cannot be earlier than previous_ts")
    return (current_ts - previous_ts).total_seconds() / 60.0


def apply_decision_threshold(fraud_probability: float, threshold: float) -> str:
    """Converts a model probability into a Legit/Fraud label at a given threshold."""
    if not (0.0 <= fraud_probability <= 1.0):
        raise ValueError(f"fraud_probability must be in [0, 1], got {fraud_probability}")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    return "Fraud" if fraud_probability >= threshold else "Legit"
