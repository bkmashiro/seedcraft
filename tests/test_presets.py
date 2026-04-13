"""Tests for preset schemas — verify they generate valid, correlated data."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine import SeedEngine
from src.generator import DataGenerator
from src.presets import PRESETS, ecommerce_customers, sensor_readings, student_records, financial_transactions
from src.analyzer import pearson_correlation, describe_dataset

SEEDS = [3813, 7889, 6140, 439, 5990, 2191, 5982, 6462,
         7318, 2050, 8243, 2900, 1314, 9843, 8348, 766]


def gen_preset(name, n=500):
    schema = PRESETS[name](n)
    engine = SeedEngine(SEEDS)
    return DataGenerator(engine).generate(schema)


class TestPresetRegistry:
    def test_all_presets_registered(self):
        assert "ecommerce_customers" in PRESETS
        assert "sensor_readings" in PRESETS
        assert "student_records" in PRESETS
        assert "financial_transactions" in PRESETS

    def test_all_presets_generate(self):
        for name in PRESETS:
            data = gen_preset(name, n=50)
            assert len(data) > 0
            for col_values in data.values():
                assert len(col_values) == 50


class TestEcommercePreset:
    def test_field_types(self):
        data = gen_preset("ecommerce_customers", 200)
        assert all(isinstance(v, str) and v.startswith("CUST-") for v in data["customer_id"])
        assert all(isinstance(v, (int, type(None))) for v in data["age"])
        assert all(v in ("bronze", "silver", "gold", "platinum") for v in data["membership_tier"])

    def test_age_income_correlation(self):
        data = gen_preset("ecommerce_customers", 1000)
        corr = pearson_correlation(data["age"], data["income"])
        assert corr is not None
        assert corr > 0.2  # Should be positively correlated

    def test_tier_discount_conditional(self):
        data = gen_preset("ecommerce_customers", 2000)
        tier_discounts = {}
        for tier, discount in zip(data["membership_tier"], data["discount_rate"]):
            if discount is not None:
                tier_discounts.setdefault(tier, []).append(discount)

        # Platinum should have higher discounts than bronze
        bronze_mean = sum(tier_discounts["bronze"]) / len(tier_discounts["bronze"])
        platinum_mean = sum(tier_discounts.get("platinum", [0.3])) / max(len(tier_discounts.get("platinum", [1])), 1)
        assert platinum_mean > bronze_mean

    def test_mutual_exclusive_shipping(self):
        data = gen_preset("ecommerce_customers", 500)
        for premium, shipping in zip(data["is_premium"], data["free_shipping"]):
            if premium:
                assert shipping is False


class TestSensorPreset:
    def test_temperature_humidity_inverse(self):
        data = gen_preset("sensor_readings", 1000)
        corr = pearson_correlation(data["temperature"], data["humidity"])
        assert corr is not None
        assert corr < 0  # Should be negatively correlated

    def test_realistic_ranges(self):
        data = gen_preset("sensor_readings", 500)
        for t in data["temperature"]:
            if t is not None:
                assert -10 <= t <= 45
        for p in data["pressure"]:
            if p is not None:
                assert 950 <= p <= 1060


class TestStudentPreset:
    def test_study_gpa_correlation(self):
        data = gen_preset("student_records", 1000)
        corr = pearson_correlation(data["study_hours_per_week"], data["gpa"])
        assert corr is not None
        assert corr > 0.2  # More study -> better GPA

    def test_gpa_range(self):
        data = gen_preset("student_records", 500)
        for gpa in data["gpa"]:
            if gpa is not None:
                assert 0 <= gpa <= 4.0


class TestFinancialPreset:
    def test_account_age_limit_correlation(self):
        data = gen_preset("financial_transactions", 1000)
        corr = pearson_correlation(data["account_age_days"], data["transaction_limit"])
        assert corr is not None
        assert corr > 0.1  # Older accounts -> higher limits

    def test_category_amount_conditional(self):
        data = gen_preset("financial_transactions", 3000)
        cat_amounts = {}
        for cat, amt in zip(data["merchant_category"], data["amount"]):
            if amt is not None:
                cat_amounts.setdefault(cat, []).append(amt)

        # Travel should have higher average than grocery
        if "travel" in cat_amounts and "grocery" in cat_amounts:
            travel_mean = sum(cat_amounts["travel"]) / len(cat_amounts["travel"])
            grocery_mean = sum(cat_amounts["grocery"]) / len(cat_amounts["grocery"])
            assert travel_mean > grocery_mean


class TestDeterminismAcrossPresets:
    def test_same_seeds_same_data(self):
        data1 = gen_preset("ecommerce_customers", 100)
        data2 = gen_preset("ecommerce_customers", 100)
        assert data1["customer_id"] == data2["customer_id"]
        assert data1["age"] == data2["age"]
