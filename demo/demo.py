#!/usr/bin/env python3
"""Seedcraft demo — showcasing correlation-aware synthetic data generation.

This demo uses the 16 seed numbers from the project specification to generate
four different datasets, then analyzes them to verify that the correlations hold.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import SeedEngine, DataGenerator, Schema, Field, Correlation, Distribution
from src.presets import PRESETS
from src.analyzer import (
    describe_dataset, correlation_matrix, pearson_correlation,
    format_stats, format_correlation_matrix, verify_correlations,
)
from src.exporters import to_csv, to_json, to_dict_list


# The 16 seed numbers
SEEDS = [3813, 7889, 6140, 439, 5990, 2191, 5982, 6462,
         7318, 2050, 8243, 2900, 1314, 9843, 8348, 766]


def separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demo_1_basic_usage():
    """Demonstrate basic schema definition and generation."""
    separator("DEMO 1: Basic Schema Definition")

    engine = SeedEngine(SEEDS)
    print(f"Engine: {engine}")

    schema = (
        Schema("demo_users", row_count=10)
        .add_field(Field.string_id("user_id", prefix="USR"))
        .add_field(Field.integer("age", low=18, high=65))
        .add_field(Field.real("balance", distribution=Distribution.lognormal(6, 1),
                              min_val=0, max_val=100000))
        .add_field(Field.choice("plan", ["free", "basic", "premium"]))
        .add_field(Field.boolean("active", true_probability=0.8))
        .add_field(Field.date("joined", start="2022-01-01", end="2026-04-13"))
    )

    gen = DataGenerator(engine)
    data = gen.generate(schema)
    rows = to_dict_list(data)

    print("Generated 10 users:")
    for row in rows:
        print(f"  {row['user_id']}: age={row['age']}, balance=${row['balance']:.2f}, "
              f"plan={row['plan']}, active={row['active']}, joined={row['joined']}")


def demo_2_correlations():
    """Demonstrate correlated field generation."""
    separator("DEMO 2: Correlated Fields")

    engine = SeedEngine(SEEDS)
    schema = Schema("demo_correlated", row_count=500)

    # Experience drives salary
    schema.add_field(Field.integer("years_experience", low=0, high=40,
                                   distribution=Distribution.exponential(8)))
    schema.add_field(Field.real("salary",
                                distribution=Distribution.lognormal(10, 0.5),
                                min_val=25000, max_val=500000))
    schema.add_correlation(Correlation.linear(
        "years_experience", "salary",
        strength=0.8, slope=5000, intercept=35000
    ))

    # Department determines bonus structure
    schema.add_field(Field.choice("department", [
        ("engineering", 35), ("sales", 25), ("marketing", 20), ("hr", 20)
    ]))
    schema.add_field(Field.real("bonus_pct", min_val=0, max_val=50))
    schema.add_correlation(Correlation.conditional("department", "bonus_pct", {
        "engineering": Distribution.normal(12, 3),
        "sales": Distribution.normal(25, 8),
        "marketing": Distribution.normal(10, 4),
        "hr": Distribution.normal(8, 2),
    }))

    gen = DataGenerator(engine)
    data = gen.generate(schema)

    # Show correlation analysis
    corr = pearson_correlation(data["years_experience"], data["salary"])
    print(f"Experience-Salary Pearson correlation: {corr}")

    # Show department bonus averages
    dept_bonuses = {}
    for dept, bonus in zip(data["department"], data["bonus_pct"]):
        dept_bonuses.setdefault(dept, []).append(bonus)

    print("\nAverage bonus % by department:")
    for dept in sorted(dept_bonuses.keys()):
        vals = dept_bonuses[dept]
        avg = sum(vals) / len(vals)
        print(f"  {dept:>15}: {avg:.1f}% (n={len(vals)})")

    # Show a few rows
    print("\nSample rows:")
    rows = to_dict_list(data)
    for row in rows[:8]:
        print(f"  exp={row['years_experience']:2d}yr  salary=${row['salary']:>10.2f}  "
              f"dept={row['department']:>12}  bonus={row['bonus_pct']:.1f}%")


def demo_3_presets():
    """Demonstrate preset schemas."""
    separator("DEMO 3: Preset Schemas")

    engine = SeedEngine(SEEDS)

    for name, factory in PRESETS.items():
        schema = factory(200)
        gen = DataGenerator(engine)
        data = gen.generate(schema)

        stats = describe_dataset(data)
        numeric_cols = [k for k, v in stats.items() if v.get("type") == "numeric"]

        print(f"\n--- {schema.name} ({schema.row_count} rows, {len(schema.fields)} fields, "
              f"{len(schema.correlations)} correlations) ---")
        print(f"  Fields: {', '.join(schema.field_names())}")
        print(f"  Numeric columns: {', '.join(numeric_cols)}")

        # Reset engine for next preset
        engine.reset()


def demo_4_analysis():
    """Demonstrate full statistical analysis."""
    separator("DEMO 4: Statistical Analysis")

    engine = SeedEngine(SEEDS)
    schema = PRESETS["sensor_readings"](1000)
    gen = DataGenerator(engine)
    data = gen.generate(schema)

    print("Dataset: sensor_readings (1000 rows)")
    print("\nColumn statistics:")
    stats = describe_dataset(data)
    print(format_stats(stats))

    print("\nCorrelation matrix (numeric columns):")
    corr = correlation_matrix(data)
    print(format_correlation_matrix(corr))

    # Verify expected correlations
    print("\nCorrelation verification:")
    results = verify_correlations(data, [
        ("temperature", "humidity", -0.5),
        ("temperature", "pressure", 0.2),
        ("light_level", "temperature", 0.3),
    ], tolerance=0.35)

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['source']} -> {r['target']}: "
              f"expected={r['expected']}, actual={r.get('actual', 'N/A')}")


def demo_5_determinism():
    """Demonstrate deterministic reproducibility."""
    separator("DEMO 5: Deterministic Reproducibility")

    schema = Schema("demo_determinism", row_count=5)
    schema.add_field(Field.integer("value", low=0, high=10000))
    schema.add_field(Field.choice("color", ["red", "green", "blue"]))

    print("Same seeds produce identical data every time:\n")
    for run in range(3):
        engine = SeedEngine(SEEDS)
        data = DataGenerator(engine).generate(schema)
        print(f"  Run {run+1}: values={data['value']}, colors={data['color']}")

    print("\nDifferent seeds produce different data:\n")
    for seed_offset in [0, 1000, 9999]:
        modified_seeds = [s + seed_offset for s in SEEDS]
        engine = SeedEngine(modified_seeds)
        data = DataGenerator(engine).generate(schema)
        print(f"  Offset {seed_offset:>4}: values={data['value']}")


def demo_6_custom_derived():
    """Demonstrate derived (computed) fields."""
    separator("DEMO 6: Derived Fields")

    engine = SeedEngine(SEEDS)
    schema = Schema("order_with_tax", row_count=8)

    schema.add_field(Field.string_id("order_id", prefix="ORD"))
    schema.add_field(Field.real("subtotal", distribution=Distribution.lognormal(3.5, 0.8),
                                min_val=5, max_val=500))
    schema.add_field(Field.real("tax"))
    schema.add_field(Field.real("total"))

    # Tax is 8.5% of subtotal
    schema.add_correlation(Correlation.derived("subtotal", "tax",
                                               func=lambda x: round(x * 0.085, 2)))
    # Total = subtotal + tax
    # For chained derivation, we need subtotal -> total directly
    schema.add_correlation(Correlation.derived("subtotal", "total",
                                               func=lambda x: round(x * 1.085, 2)))

    data = DataGenerator(engine).generate(schema)
    rows = to_dict_list(data)

    print("Orders with derived tax and total:")
    for row in rows:
        print(f"  {row['order_id']}: subtotal=${row['subtotal']:>7.2f}  "
              f"tax=${row['tax']:>6.2f}  total=${row['total']:>7.2f}")


def demo_7_export():
    """Demonstrate export formats."""
    separator("DEMO 7: Export Formats")

    engine = SeedEngine(SEEDS)
    schema = PRESETS["ecommerce_customers"](5)
    data = DataGenerator(engine).generate(schema)

    print("CSV format (first 3 lines):")
    csv_str = to_csv(data)
    for line in csv_str.strip().split("\n")[:3]:
        print(f"  {line}")

    print("\nJSON format (first record):")
    import json
    json_str = to_json(data, orient="records")
    records = json.loads(json_str)
    print(f"  {json.dumps(records[0], indent=4)}")


def main():
    print("=" * 70)
    print("  SEEDCRAFT — Deterministic, Correlation-Aware Synthetic Data")
    print(f"  Seeds: {SEEDS}")
    print("=" * 70)

    demo_1_basic_usage()
    demo_2_correlations()
    demo_3_presets()
    demo_4_analysis()
    demo_5_determinism()
    demo_6_custom_derived()
    demo_7_export()

    separator("DEMO COMPLETE")
    print("All demonstrations completed successfully.")
    print(f"Seedcraft generated data using {len(SEEDS)} seed numbers across")
    print(f"{len(PRESETS)} preset schemas with correlation-aware generation.\n")


if __name__ == "__main__":
    main()
