"""Tests for the DataGenerator."""

import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine import SeedEngine
from src.generator import DataGenerator
from src.schema import (
    Schema, Field, FieldType, Correlation, CorrelationType,
    Distribution, DistributionKind,
)
from src.analyzer import pearson_correlation


SEEDS = [3813, 7889, 6140, 439]


def make_gen():
    return DataGenerator(SeedEngine(SEEDS))


class TestBasicGeneration:
    def test_integer_field(self):
        schema = Schema("test", row_count=100)
        schema.add_field(Field.integer("x", low=0, high=100))
        data = make_gen().generate(schema)
        assert len(data["x"]) == 100
        assert all(isinstance(v, int) for v in data["x"])
        assert all(0 <= v <= 100 for v in data["x"])

    def test_float_field(self):
        schema = Schema("test", row_count=50)
        schema.add_field(Field.real("x", distribution=Distribution.normal(10, 2)))
        data = make_gen().generate(schema)
        assert len(data["x"]) == 50
        assert all(isinstance(v, float) for v in data["x"])

    def test_choice_field(self):
        schema = Schema("test", row_count=200)
        schema.add_field(Field.choice("color", ["red", "green", "blue"]))
        data = make_gen().generate(schema)
        assert all(v in ["red", "green", "blue"] for v in data["color"])

    def test_weighted_choice_field(self):
        schema = Schema("test", row_count=1000)
        schema.add_field(Field.choice("tier", [("common", 90), ("rare", 10)]))
        data = make_gen().generate(schema)
        common_count = sum(1 for v in data["tier"] if v == "common")
        # Should be roughly 90%
        assert common_count > 800

    def test_boolean_field(self):
        schema = Schema("test", row_count=1000)
        schema.add_field(Field.boolean("active", true_probability=0.3))
        data = make_gen().generate(schema)
        true_count = sum(1 for v in data["active"] if v is True)
        # Should be roughly 30%
        assert 200 < true_count < 400

    def test_date_field(self):
        schema = Schema("test", row_count=50)
        schema.add_field(Field.date("d", start="2024-01-01", end="2024-12-31"))
        data = make_gen().generate(schema)
        assert all(v.startswith("2024-") for v in data["d"])

    def test_string_id_field(self):
        schema = Schema("test", row_count=10)
        schema.add_field(Field.string_id("id", prefix="TST"))
        data = make_gen().generate(schema)
        assert data["id"][0] == "TST-000001"
        assert data["id"][9] == "TST-000010"
        assert len(set(data["id"])) == 10  # All unique


class TestDistributions:
    def test_normal_distribution_stats(self):
        schema = Schema("test", row_count=5000)
        schema.add_field(Field.real("x", distribution=Distribution.normal(100, 10)))
        data = make_gen().generate(schema)
        mean = sum(data["x"]) / len(data["x"])
        assert 95 < mean < 105

    def test_uniform_distribution_bounds(self):
        schema = Schema("test", row_count=1000)
        schema.add_field(Field.real("x", distribution=Distribution.uniform(5, 15),
                                    min_val=5, max_val=15))
        data = make_gen().generate(schema)
        assert all(5 <= v <= 15 for v in data["x"])

    def test_exponential_distribution(self):
        schema = Schema("test", row_count=2000)
        schema.add_field(Field.real("x", distribution=Distribution.exponential(10),
                                    min_val=0))
        data = make_gen().generate(schema)
        mean = sum(data["x"]) / len(data["x"])
        # Exponential mean should be close to scale
        assert 7 < mean < 14

    def test_beta_distribution_bounded(self):
        schema = Schema("test", row_count=1000)
        schema.add_field(Field.real("x", distribution=Distribution.beta(2, 5)))
        data = make_gen().generate(schema)
        assert all(0 <= v <= 1 for v in data["x"])

    def test_poisson_distribution(self):
        schema = Schema("test", row_count=2000)
        schema.add_field(Field.real("x", distribution=Distribution.poisson(5)))
        data = make_gen().generate(schema)
        mean = sum(data["x"]) / len(data["x"])
        assert 4 < mean < 6.5

    def test_lognormal_positive(self):
        schema = Schema("test", row_count=500)
        schema.add_field(Field.real("x", distribution=Distribution.lognormal(2, 0.5)))
        data = make_gen().generate(schema)
        assert all(v > 0 for v in data["x"])


class TestNullability:
    def test_nullable_field(self):
        schema = Schema("test", row_count=1000)
        schema.add_field(Field.integer("x", low=0, high=100, nullable=0.2))
        data = make_gen().generate(schema)
        null_count = sum(1 for v in data["x"] if v is None)
        assert 150 < null_count < 250  # ~20% nulls

    def test_zero_nullable(self):
        schema = Schema("test", row_count=100)
        schema.add_field(Field.integer("x", low=0, high=10, nullable=0))
        data = make_gen().generate(schema)
        assert all(v is not None for v in data["x"])


class TestTransforms:
    def test_transform_applied(self):
        schema = Schema("test", row_count=50)
        schema.add_field(Field.real("x", distribution=Distribution.uniform(0, 100),
                                    transform=lambda v: round(v, 1)))
        data = make_gen().generate(schema)
        # All values should have at most 1 decimal place
        for v in data["x"]:
            assert v == round(v, 1)


class TestDeterminism:
    def test_same_seeds_same_output(self):
        schema = Schema("test", row_count=100)
        schema.add_field(Field.integer("x", low=0, high=1000))
        schema.add_field(Field.choice("cat", ["a", "b", "c"]))

        data1 = DataGenerator(SeedEngine(SEEDS)).generate(schema)
        data2 = DataGenerator(SeedEngine(SEEDS)).generate(schema)
        assert data1 == data2

    def test_different_seeds_different_output(self):
        schema = Schema("test", row_count=100)
        schema.add_field(Field.integer("x", low=0, high=1000))

        data1 = DataGenerator(SeedEngine([1])).generate(schema)
        data2 = DataGenerator(SeedEngine([2])).generate(schema)
        assert data1["x"] != data2["x"]

    def test_field_independence(self):
        """Adding a field shouldn't change other fields' values."""
        schema1 = Schema("test", row_count=100)
        schema1.add_field(Field.integer("x", low=0, high=1000))

        schema2 = Schema("test", row_count=100)
        schema2.add_field(Field.integer("x", low=0, high=1000))
        schema2.add_field(Field.integer("y", low=0, high=1000))

        data1 = DataGenerator(SeedEngine(SEEDS)).generate(schema1)
        data2 = DataGenerator(SeedEngine(SEEDS)).generate(schema2)

        # x values should be identical regardless of y's presence
        assert data1["x"] == data2["x"]


class TestCorrelations:
    def test_linear_correlation_positive(self):
        schema = Schema("test", row_count=2000)
        schema.add_field(Field.real("x", distribution=Distribution.uniform(0, 100)))
        schema.add_field(Field.real("y"))
        schema.add_correlation(Correlation.linear("x", "y", strength=0.9,
                                                  slope=2, intercept=10))
        data = make_gen().generate(schema)
        corr = pearson_correlation(data["x"], data["y"])
        assert corr is not None
        assert corr > 0.5  # Should be positively correlated

    def test_linear_correlation_negative(self):
        schema = Schema("test", row_count=2000)
        schema.add_field(Field.real("x", distribution=Distribution.uniform(0, 100)))
        schema.add_field(Field.real("y"))
        schema.add_correlation(Correlation.linear("x", "y", strength=0.9,
                                                  slope=-3, intercept=500))
        data = make_gen().generate(schema)
        corr = pearson_correlation(data["x"], data["y"])
        assert corr is not None
        assert corr < -0.3  # Should be negatively correlated

    def test_derived_correlation(self):
        schema = Schema("test", row_count=100)
        schema.add_field(Field.real("x", distribution=Distribution.uniform(1, 50)))
        schema.add_field(Field.real("y"))
        schema.add_correlation(Correlation.derived("x", "y", func=lambda v: v * 2 + 1))
        data = make_gen().generate(schema)
        for x, y in zip(data["x"], data["y"]):
            assert abs(y - (x * 2 + 1)) < 0.001

    def test_conditional_correlation(self):
        schema = Schema("test", row_count=2000)
        schema.add_field(Field.choice("category", ["low", "high"]))
        schema.add_field(Field.real("value", min_val=0, max_val=1000))
        schema.add_correlation(Correlation.conditional("category", "value", {
            "low": Distribution.normal(10, 2),
            "high": Distribution.normal(100, 10),
        }))
        data = make_gen().generate(schema)

        low_vals = [v for c, v in zip(data["category"], data["value"]) if c == "low"]
        high_vals = [v for c, v in zip(data["category"], data["value"]) if c == "high"]

        low_mean = sum(low_vals) / len(low_vals)
        high_mean = sum(high_vals) / len(high_vals)

        assert low_mean < 30  # Should be around 10
        assert high_mean > 60  # Should be around 100

    def test_monotonic_correlation(self):
        schema = Schema("test", row_count=500)
        schema.add_field(Field.real("x", distribution=Distribution.uniform(0, 100)))
        schema.add_field(Field.real("y", distribution=Distribution.uniform(0, 100)))
        schema.add_correlation(Correlation.monotonic("x", "y", strength=0.9, increasing=True))
        data = make_gen().generate(schema)
        corr = pearson_correlation(data["x"], data["y"])
        assert corr is not None
        assert corr > 0.5

    def test_mutual_exclusive(self):
        schema = Schema("test", row_count=500)
        schema.add_field(Field.boolean("a", true_probability=0.5))
        schema.add_field(Field.boolean("b"))
        schema.add_correlation(Correlation.mutual_exclusive("a", "b"))
        data = make_gen().generate(schema)
        # When a is True, b must be False
        for a, b in zip(data["a"], data["b"]):
            if a:
                assert b is False


class TestDependencyGraph:
    def test_chained_dependencies(self):
        """a -> b -> c should work."""
        schema = Schema("test", row_count=100)
        schema.add_field(Field.real("a", distribution=Distribution.uniform(0, 100)))
        schema.add_field(Field.real("b"))
        schema.add_field(Field.real("c"))
        schema.add_correlation(Correlation.derived("a", "b", func=lambda x: x * 2))
        schema.add_correlation(Correlation.derived("b", "c", func=lambda x: x + 10))
        data = make_gen().generate(schema)
        for a, b, c in zip(data["a"], data["b"], data["c"]):
            assert abs(b - a * 2) < 0.001
            assert abs(c - (a * 2 + 10)) < 0.001

    def test_circular_dependency_detected(self):
        schema = Schema("test", row_count=10)
        schema.add_field(Field.real("a"))
        schema.add_field(Field.real("b"))
        schema.add_correlation(Correlation.linear("a", "b"))
        schema.add_correlation(Correlation.linear("b", "a"))
        with pytest.raises(ValueError, match="Circular"):
            make_gen().generate(schema)


class TestUniqueness:
    def test_unique_strings(self):
        schema = Schema("test", row_count=100)
        schema.add_field(Field.string_id("id", prefix="X"))
        data = make_gen().generate(schema)
        assert len(set(data["id"])) == 100

    def test_unique_integers(self):
        schema = Schema("test", row_count=50)
        schema.add_field(Field.integer("x", low=0, high=10000, unique=True))
        data = make_gen().generate(schema)
        non_null = [v for v in data["x"] if v is not None]
        assert len(set(non_null)) == len(non_null)


class TestEdgeCases:
    def test_single_row(self):
        schema = Schema("test", row_count=1)
        schema.add_field(Field.integer("x", low=5, high=5))
        data = make_gen().generate(schema)
        assert len(data["x"]) == 1
        assert data["x"][0] == 5

    def test_validation_error(self):
        schema = Schema("test", row_count=10)
        schema.add_field(Field.integer("x"))
        schema.add_field(Field.integer("x"))  # duplicate
        with pytest.raises(ValueError, match="validation"):
            make_gen().generate(schema)

    def test_multiple_fields(self):
        schema = Schema("test", row_count=50)
        schema.add_field(Field.integer("a", low=0, high=10))
        schema.add_field(Field.real("b", distribution=Distribution.normal(0, 1)))
        schema.add_field(Field.choice("c", ["x", "y"]))
        schema.add_field(Field.boolean("d"))
        schema.add_field(Field.string_id("e"))
        data = make_gen().generate(schema)
        assert set(data.keys()) == {"a", "b", "c", "d", "e"}
        assert all(len(v) == 50 for v in data.values())
