"""
Unit tests for src/features.py

Run with: pytest tests/ -v
"""
import sys
import os
from datetime import datetime
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import (
    validate_transaction,
    validate_customer,
    compute_amount_vs_own_avg_ratio,
    is_odd_hour,
    is_away_from_home_city,
    is_unusual_category,
    minutes_since,
    apply_decision_threshold,
    SchemaValidationError,
)


# ---------------------------------------------------------
# Schema validation
# ---------------------------------------------------------
class TestValidateTransaction:
    def test_valid_transaction_passes(self):
        txn = {"amount": 100.0, "timestamp": datetime(2026, 1, 1), "merchant_category": "grocery", "city": "Lahore"}
        validate_transaction(txn)  # should not raise

    def test_missing_field_raises(self):
        txn = {"amount": 100.0, "timestamp": datetime(2026, 1, 1), "city": "Lahore"}
        with pytest.raises(SchemaValidationError, match="merchant_category"):
            validate_transaction(txn)

    def test_negative_amount_raises(self):
        txn = {"amount": -5.0, "timestamp": datetime(2026, 1, 1), "merchant_category": "grocery", "city": "Lahore"}
        with pytest.raises(SchemaValidationError, match="non-negative"):
            validate_transaction(txn)


class TestValidateCustomer:
    def test_valid_customer_passes(self):
        cust = {"home_city": "Lahore", "avg_spend": 50.0, "favorite_category": "grocery", "risk_profile": "low"}
        validate_customer(cust)

    def test_missing_field_raises(self):
        cust = {"home_city": "Lahore", "avg_spend": 50.0, "risk_profile": "low"}
        with pytest.raises(SchemaValidationError, match="favorite_category"):
            validate_customer(cust)


# ---------------------------------------------------------
# Feature computations — each mirrors a specific SQL expression,
# so these tests are what guarantee Python and SQL stay in agreement.
# ---------------------------------------------------------
class TestAmountRatio:
    def test_normal_ratio(self):
        assert compute_amount_vs_own_avg_ratio(amount=200.0, customer_rolling_avg=100.0) == 2.0

    def test_below_average_ratio(self):
        assert compute_amount_vs_own_avg_ratio(amount=50.0, customer_rolling_avg=100.0) == 0.5

    def test_zero_avg_returns_neutral_ratio(self):
        # Guards against division by zero, matching the SQL's NULLIF behavior
        assert compute_amount_vs_own_avg_ratio(amount=100.0, customer_rolling_avg=0.0) == 1.0

    def test_none_avg_returns_neutral_ratio(self):
        # A brand-new customer with no transaction history yet
        assert compute_amount_vs_own_avg_ratio(amount=100.0, customer_rolling_avg=None) == 1.0


class TestOddHour:
    @pytest.mark.parametrize("hour", [1, 2, 3, 4, 5])
    def test_odd_hours_flagged_true(self, hour):
        assert is_odd_hour(hour) is True

    @pytest.mark.parametrize("hour", [0, 6, 12, 18, 23])
    def test_normal_hours_flagged_false(self, hour):
        assert is_odd_hour(hour) is False

    def test_invalid_hour_raises(self):
        with pytest.raises(ValueError):
            is_odd_hour(25)


class TestLocationAndCategory:
    def test_same_city_is_not_away(self):
        assert is_away_from_home_city("Lahore", "Lahore") is False

    def test_different_city_is_away(self):
        assert is_away_from_home_city("Karachi", "Lahore") is True

    def test_city_comparison_is_case_insensitive(self):
        # Real-world data is inconsistent about capitalization — the feature
        # shouldn't falsely flag "lahore" vs "Lahore" as a location anomaly.
        assert is_away_from_home_city("lahore", "Lahore") is False

    def test_unusual_category_detected(self):
        assert is_unusual_category("electronics", "grocery") is True

    def test_favorite_category_not_flagged(self):
        assert is_unusual_category("grocery", "grocery") is False


class TestMinutesSince:
    def test_computes_correct_minutes(self):
        prev = datetime(2026, 1, 1, 10, 0)
        curr = datetime(2026, 1, 1, 10, 30)
        assert minutes_since(curr, prev) == 30.0

    def test_no_previous_transaction_returns_sentinel(self):
        # First-ever transaction for a customer — matches the SQL COALESCE(..., 999999)
        assert minutes_since(datetime(2026, 1, 1), None) == 999999.0

    def test_out_of_order_timestamps_raise(self):
        prev = datetime(2026, 1, 1, 10, 0)
        curr = datetime(2026, 1, 1, 9, 0)  # earlier than "previous" — invalid
        with pytest.raises(ValueError):
            minutes_since(curr, prev)


# ---------------------------------------------------------
# Decision threshold
# ---------------------------------------------------------
class TestDecisionThreshold:
    def test_above_threshold_is_fraud(self):
        assert apply_decision_threshold(0.95, threshold=0.5) == "Fraud"

    def test_below_threshold_is_legit(self):
        assert apply_decision_threshold(0.1, threshold=0.5) == "Legit"

    def test_exactly_at_threshold_is_fraud(self):
        # Boundary condition — matches the model's >= threshold logic in score.py
        assert apply_decision_threshold(0.5, threshold=0.5) == "Fraud"

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError):
            apply_decision_threshold(1.5, threshold=0.5)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            apply_decision_threshold(0.5, threshold=-0.1)
