"""Tests for the statistical analyzer."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analyzer import (
    describe_column, describe_dataset, pearson_correlation,
    correlation_matrix, verify_correlations, format_stats,
    format_correlation_matrix,
)


class TestDescribeColumn:
    def test_numeric_column(self):
        vals = list(range(100))
        stats = describe_column(vals)
        assert stats["type"] == "numeric"
        assert stats["count"] == 100
        assert stats["non_null"] == 100
        assert stats["min"] == 0
        assert stats["max"] == 99
        assert 45 < stats["mean"] < 55

    def test_categorical_column(self):
        vals = ["a", "b", "a", "c", "b", "a"]
        stats = describe_column(vals)
        assert stats["type"] == "categorical"
        assert stats["unique"] == 3
        assert stats["top_values"][0] == ("a", 3)

    def test_with_nulls(self):
        vals = [1, 2, None, 4, None]
        stats = describe_column(vals)
        assert stats["null_rate"] == 0.4
        assert stats["non_null"] == 3

    def test_all_nulls(self):
        vals = [None, None, None]
        stats = describe_column(vals)
        assert stats["null_rate"] == 1.0


class TestPearsonCorrelation:
    def test_perfect_positive(self):
        xs = list(range(100))
        ys = [x * 2 + 1 for x in xs]
        corr = pearson_correlation(xs, ys)
        assert corr is not None
        assert abs(corr - 1.0) < 0.001

    def test_perfect_negative(self):
        xs = list(range(100))
        ys = [-x for x in xs]
        corr = pearson_correlation(xs, ys)
        assert corr is not None
        assert abs(corr - (-1.0)) < 0.001

    def test_no_correlation(self):
        # Alternating pattern has near-zero correlation with linear
        import random
        rng = random.Random(42)
        xs = list(range(1000))
        ys = [rng.gauss(0, 1) for _ in range(1000)]
        corr = pearson_correlation(xs, ys)
        assert corr is not None
        assert abs(corr) < 0.15

    def test_with_nulls(self):
        xs = [1, 2, None, 4, 5]
        ys = [2, 4, 6, 8, 10]
        corr = pearson_correlation(xs, ys)
        assert corr is not None
        assert corr > 0.9

    def test_too_few_values(self):
        assert pearson_correlation([1, 2], [3, 4]) is None

    def test_zero_variance(self):
        assert pearson_correlation([5, 5, 5, 5], [1, 2, 3, 4]) is None


class TestCorrelationMatrix:
    def test_basic_matrix(self):
        data = {
            "x": list(range(100)),
            "y": [v * 2 for v in range(100)],
            "z": list(range(100, 0, -1)),
            "cat": ["a"] * 100,  # non-numeric, should be excluded
        }
        matrix = correlation_matrix(data)
        assert "x" in matrix
        assert "y" in matrix
        assert "z" in matrix
        assert "cat" not in matrix
        assert matrix["x"]["x"] == 1.0
        assert abs(matrix["x"]["y"] - 1.0) < 0.001
        assert abs(matrix["x"]["z"] - (-1.0)) < 0.001


class TestVerifyCorrelations:
    def test_passing(self):
        data = {
            "x": list(range(100)),
            "y": [v * 3 for v in range(100)],
        }
        results = verify_correlations(data, [("x", "y", 1.0)])
        assert len(results) == 1
        assert results[0]["passed"] is True

    def test_failing(self):
        data = {
            "x": list(range(100)),
            "y": [v * 3 for v in range(100)],
        }
        results = verify_correlations(data, [("x", "y", -0.5)], tolerance=0.1)
        assert results[0]["passed"] is False

    def test_missing_field(self):
        data = {"x": [1, 2, 3]}
        results = verify_correlations(data, [("x", "missing", 0.5)])
        assert results[0]["passed"] is False
        assert "not found" in results[0]["reason"]


class TestFormatting:
    def test_format_stats(self):
        stats = describe_dataset({
            "x": list(range(10)),
            "cat": ["a", "b"] * 5,
        })
        output = format_stats(stats)
        assert "x:" in output
        assert "cat:" in output

    def test_format_correlation_matrix(self):
        data = {"x": list(range(50)), "y": list(range(50))}
        matrix = correlation_matrix(data)
        output = format_correlation_matrix(matrix)
        assert "x" in output
        assert "y" in output

    def test_format_empty_matrix(self):
        assert "no numeric" in format_correlation_matrix({})
