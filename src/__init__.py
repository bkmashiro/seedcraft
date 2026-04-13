"""Seedcraft — Deterministic, correlation-aware synthetic data generation."""

__version__ = "0.1.0"

from .schema import Schema, Field, Correlation, Distribution
from .engine import SeedEngine
from .generator import DataGenerator
from .exporters import to_csv, to_json, to_dict_list

__all__ = [
    "Schema",
    "Field",
    "Correlation",
    "Distribution",
    "SeedEngine",
    "DataGenerator",
    "to_csv",
    "to_json",
    "to_dict_list",
]
