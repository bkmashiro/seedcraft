"""Pre-built schemas for common data generation scenarios.

Each preset demonstrates correlation-aware generation for a specific domain.
Users can use these directly or as templates for custom schemas.
"""

from __future__ import annotations

from .schema import (
    Correlation,
    CorrelationType,
    Distribution,
    DistributionKind,
    Field,
    FieldType,
    Schema,
)


def ecommerce_customers(n: int = 200) -> Schema:
    """E-commerce customer dataset with realistic correlations.

    Correlations:
    - age -> income (older customers tend to earn more)
    - income -> avg_order_value (higher income -> higher spending)
    - membership_tier -> discount_rate (tier determines discount)
    - is_premium -> free_shipping (premium members get free shipping)
    """
    schema = Schema("ecommerce_customers", row_count=n)

    schema.add_field(Field.string_id("customer_id", prefix="CUST"))
    schema.add_field(Field.integer("age", low=18, high=80,
                                   distribution=Distribution.normal(38, 14)))
    schema.add_field(Field.real("income", distribution=Distribution.lognormal(10.5, 0.6),
                                min_val=15000, max_val=500000))
    schema.add_field(Field.choice("membership_tier",
                                  [("bronze", 50), ("silver", 30), ("gold", 15), ("platinum", 5)]))
    schema.add_field(Field.real("avg_order_value",
                                distribution=Distribution.lognormal(3.5, 0.8),
                                min_val=5, max_val=5000))
    schema.add_field(Field.real("discount_rate", min_val=0, max_val=0.5))
    schema.add_field(Field.boolean("is_premium", true_probability=0.2))
    schema.add_field(Field.boolean("free_shipping"))
    schema.add_field(Field.date("signup_date", start="2019-01-01", end="2026-04-01"))
    schema.add_field(Field.integer("total_orders", low=0, high=500,
                                   distribution=Distribution.exponential(15)))

    # Correlations
    schema.add_correlation(Correlation.linear("age", "income",
                                              strength=0.6, slope=800, intercept=20000))
    schema.add_correlation(Correlation.linear("income", "avg_order_value",
                                              strength=0.5, slope=0.001, intercept=30))
    schema.add_correlation(Correlation.conditional("membership_tier", "discount_rate", {
        "bronze": Distribution.uniform(0, 0.05),
        "silver": Distribution.uniform(0.05, 0.15),
        "gold": Distribution.uniform(0.15, 0.25),
        "platinum": Distribution.uniform(0.25, 0.45),
    }))
    schema.add_correlation(Correlation.mutual_exclusive("is_premium", "free_shipping"))

    return schema


def sensor_readings(n: int = 500) -> Schema:
    """IoT sensor dataset with physically realistic correlations.

    Correlations:
    - temperature -> humidity (inverse relationship)
    - temperature -> pressure (slight positive correlation)
    - light_level -> temperature (sunlight warms sensors)
    """
    schema = Schema("sensor_readings", row_count=n)

    schema.add_field(Field.string_id("reading_id", prefix="SNS"))
    schema.add_field(Field.date("timestamp", start="2026-01-01", end="2026-04-13",
                                fmt="%Y-%m-%d %H:%M:%S"))
    schema.add_field(Field.real("temperature",
                                distribution=Distribution.normal(22, 8),
                                min_val=-10, max_val=45))
    schema.add_field(Field.real("humidity",
                                distribution=Distribution.beta(2, 5),
                                min_val=10, max_val=100))
    schema.add_field(Field.real("pressure",
                                distribution=Distribution.normal(1013, 15),
                                min_val=950, max_val=1060))
    schema.add_field(Field.real("light_level",
                                distribution=Distribution.uniform(0, 1000)))
    schema.add_field(Field.choice("sensor_zone", ["A", "B", "C", "D"]))
    schema.add_field(Field.boolean("alert_triggered", true_probability=0.05))

    # Temperature inversely affects humidity
    schema.add_correlation(Correlation.linear("temperature", "humidity",
                                              strength=0.7, slope=-1.5, intercept=90))
    schema.add_correlation(Correlation.linear("temperature", "pressure",
                                              strength=0.3, slope=0.5, intercept=1005))
    schema.add_correlation(Correlation.linear("light_level", "temperature",
                                              strength=0.4, slope=0.02, intercept=15))

    return schema


def student_records(n: int = 300) -> Schema:
    """Academic records with study-performance correlations.

    Correlations:
    - study_hours -> gpa (more study = better grades, with noise)
    - gpa -> scholarship_amount (conditional on GPA ranges)
    - extracurriculars -> leadership_score (monotonic)
    """
    schema = Schema("student_records", row_count=n)

    schema.add_field(Field.string_id("student_id", prefix="STU"))
    schema.add_field(Field.integer("age", low=17, high=25,
                                   distribution=Distribution.normal(20, 2)))
    schema.add_field(Field.choice("major", [
        ("Computer Science", 25), ("Biology", 15), ("Engineering", 20),
        ("Business", 18), ("Psychology", 12), ("Arts", 10),
    ]))
    schema.add_field(Field.real("study_hours_per_week",
                                distribution=Distribution.lognormal(2.5, 0.5),
                                min_val=0, max_val=60))
    schema.add_field(Field.real("gpa", distribution=Distribution.beta(5, 2),
                                min_val=0, max_val=4.0,
                                transform=lambda x: round(x, 2)))
    schema.add_field(Field.integer("extracurriculars", low=0, high=8,
                                   distribution=Distribution.poisson(2)))
    schema.add_field(Field.real("leadership_score",
                                distribution=Distribution.uniform(0, 10),
                                min_val=0, max_val=10))
    schema.add_field(Field.real("scholarship_amount",
                                min_val=0, max_val=50000))

    schema.add_correlation(Correlation.linear("study_hours_per_week", "gpa",
                                              strength=0.7, slope=0.05, intercept=1.5))
    schema.add_correlation(Correlation.monotonic("extracurriculars", "leadership_score",
                                                 strength=0.75, increasing=True))
    schema.add_correlation(Correlation.linear("gpa", "scholarship_amount",
                                              strength=0.65, slope=10000, intercept=-5000))

    return schema


def financial_transactions(n: int = 1000) -> Schema:
    """Financial transaction dataset with fraud-relevant patterns.

    Correlations:
    - transaction_amount -> is_flagged (large transactions more likely flagged)
    - account_age_days -> transaction_limit (older accounts get higher limits)
    - merchant_category -> avg_transaction (category determines spending range)
    """
    schema = Schema("financial_transactions", row_count=n)

    schema.add_field(Field.string_id("txn_id", prefix="TXN"))
    schema.add_field(Field.date("txn_date", start="2026-01-01", end="2026-04-13"))
    schema.add_field(Field.real("amount",
                                distribution=Distribution.lognormal(3.5, 1.2),
                                min_val=0.01, max_val=100000))
    schema.add_field(Field.choice("merchant_category", [
        ("grocery", 30), ("electronics", 15), ("restaurant", 25),
        ("travel", 10), ("entertainment", 12), ("utilities", 8),
    ]))
    schema.add_field(Field.choice("payment_method", [
        ("credit_card", 40), ("debit_card", 30), ("digital_wallet", 20), ("wire", 10),
    ]))
    schema.add_field(Field.integer("account_age_days", low=1, high=3650,
                                   distribution=Distribution.exponential(365)))
    schema.add_field(Field.real("transaction_limit",
                                distribution=Distribution.lognormal(7, 0.8),
                                min_val=100, max_val=500000))
    schema.add_field(Field.boolean("is_flagged", true_probability=0.03))
    schema.add_field(Field.boolean("is_international", true_probability=0.15))

    schema.add_correlation(Correlation.linear("account_age_days", "transaction_limit",
                                              strength=0.6, slope=50, intercept=500))
    schema.add_correlation(Correlation.conditional("merchant_category", "amount", {
        "grocery": Distribution.lognormal(3.0, 0.5),
        "electronics": Distribution.lognormal(5.0, 1.0),
        "restaurant": Distribution.lognormal(3.2, 0.6),
        "travel": Distribution.lognormal(6.0, 0.8),
        "entertainment": Distribution.lognormal(3.5, 0.7),
        "utilities": Distribution.lognormal(4.5, 0.3),
    }))

    return schema


PRESETS = {
    "ecommerce_customers": ecommerce_customers,
    "sensor_readings": sensor_readings,
    "student_records": student_records,
    "financial_transactions": financial_transactions,
}
