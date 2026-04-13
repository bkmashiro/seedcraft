"""Tests for Schema definitions."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.schema import (
    Schema, Field, FieldType, Correlation, CorrelationType,
    Distribution, DistributionKind,
)


class TestDistribution:
    def test_uniform(self):
        d = Distribution.uniform(5, 10)
        assert d.kind == DistributionKind.UNIFORM
        assert d.params["low"] == 5
        assert d.params["high"] == 10

    def test_normal(self):
        d = Distribution.normal(100, 15)
        assert d.kind == DistributionKind.NORMAL
        assert d.params["mean"] == 100
        assert d.params["std"] == 15

    def test_lognormal(self):
        d = Distribution.lognormal(2, 0.5)
        assert d.kind == DistributionKind.LOGNORMAL

    def test_exponential(self):
        d = Distribution.exponential(10)
        assert d.params["scale"] == 10

    def test_beta(self):
        d = Distribution.beta(2, 5)
        assert d.params["alpha"] == 2
        assert d.params["beta"] == 5

    def test_poisson(self):
        d = Distribution.poisson(3.5)
        assert d.params["lam"] == 3.5

    def test_zipf(self):
        d = Distribution.zipf(1.5)
        assert d.params["a"] == 1.5


class TestField:
    def test_integer_convenience(self):
        f = Field.integer("age", low=0, high=120)
        assert f.field_type == FieldType.INTEGER
        assert f.min_val == 0
        assert f.max_val == 120

    def test_real_convenience(self):
        f = Field.real("price", distribution=Distribution.lognormal(3, 1))
        assert f.field_type == FieldType.FLOAT

    def test_choice_convenience(self):
        f = Field.choice("color", ["red", "green", "blue"])
        assert f.field_type == FieldType.CHOICE
        assert len(f.choices) == 3

    def test_boolean_convenience(self):
        f = Field.boolean("active", true_probability=0.7)
        assert f.field_type == FieldType.BOOLEAN
        assert f.min_val == 0.7  # threshold stored in min_val

    def test_date_convenience(self):
        f = Field.date("created", start="2020-01-01", end="2025-12-31")
        assert f.field_type == FieldType.DATE
        assert "2020-01-01" in f.prefix

    def test_string_id_convenience(self):
        f = Field.string_id("user_id", prefix="USR")
        assert f.field_type == FieldType.STRING
        assert f.prefix == "USR"
        assert f.unique is True

    def test_nullable(self):
        f = Field.integer("x", nullable=0.3)
        assert f.nullable == 0.3


class TestCorrelation:
    def test_linear(self):
        c = Correlation.linear("age", "income", strength=0.8, slope=500, intercept=10000)
        assert c.correlation_type == CorrelationType.LINEAR
        assert c.strength == 0.8
        assert c.params["slope"] == 500

    def test_monotonic(self):
        c = Correlation.monotonic("x", "y", increasing=False)
        assert c.correlation_type == CorrelationType.MONOTONIC
        assert c.params["increasing"] is False

    def test_conditional(self):
        mapping = {"A": Distribution.normal(10, 2), "B": Distribution.normal(20, 3)}
        c = Correlation.conditional("category", "value", mapping)
        assert c.correlation_type == CorrelationType.CONDITIONAL

    def test_derived(self):
        c = Correlation.derived("x", "y", func=lambda x: x * 2)
        assert c.correlation_type == CorrelationType.DERIVED

    def test_mutual_exclusive(self):
        c = Correlation.mutual_exclusive("a", "b")
        assert c.correlation_type == CorrelationType.MUTUAL_EXCLUSIVE


class TestSchema:
    def test_create_schema(self):
        s = Schema("test")
        assert s.name == "test"
        assert s.fields == []
        assert s.correlations == []

    def test_add_field_chaining(self):
        s = Schema("test")
        result = s.add_field(Field.integer("x")).add_field(Field.integer("y"))
        assert result is s
        assert len(s.fields) == 2

    def test_field_names(self):
        s = Schema("test")
        s.add_field(Field.integer("a"))
        s.add_field(Field.real("b"))
        assert s.field_names() == ["a", "b"]

    def test_get_field(self):
        s = Schema("test")
        s.add_field(Field.integer("x", low=0, high=10))
        f = s.get_field("x")
        assert f.name == "x"

    def test_get_field_missing(self):
        s = Schema("test")
        with pytest.raises(KeyError):
            s.get_field("nonexistent")

    def test_validate_ok(self):
        s = Schema("test")
        s.add_field(Field.integer("x"))
        s.add_field(Field.integer("y"))
        s.add_correlation(Correlation.linear("x", "y"))
        assert s.validate() == []

    def test_validate_duplicate_fields(self):
        s = Schema("test")
        s.add_field(Field.integer("x"))
        s.add_field(Field.integer("x"))
        errors = s.validate()
        assert any("Duplicate" in e for e in errors)

    def test_validate_missing_correlation_field(self):
        s = Schema("test")
        s.add_field(Field.integer("x"))
        s.add_correlation(Correlation.linear("x", "y"))
        errors = s.validate()
        assert any("y" in e for e in errors)

    def test_validate_choice_no_options(self):
        s = Schema("test")
        s.add_field(Field("cat", field_type=FieldType.CHOICE))
        errors = s.validate()
        assert any("no choices" in e for e in errors)

    def test_to_dict(self):
        s = Schema("test", row_count=50)
        s.add_field(Field.integer("x"))
        d = s.to_dict()
        assert d["name"] == "test"
        assert d["row_count"] == 50
        assert len(d["fields"]) == 1
