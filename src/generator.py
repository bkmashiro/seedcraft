"""Core data generation engine with correlation support.

The generator processes a Schema in dependency order:
1. Independent fields (no incoming correlations) are generated first.
2. Dependent fields are generated using their source field's values.
3. Correlations modulate the target field's values to achieve the
   desired relationship strength.
"""

from __future__ import annotations

import math
import random as _random_mod
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional, Sequence

from .schema import (
    Correlation,
    CorrelationType,
    Distribution,
    DistributionKind,
    Field,
    FieldType,
    Schema,
)
from .engine import SeedEngine


class DataGenerator:
    """Generate synthetic datasets from schemas with correlated fields.

    Parameters
    ----------
    engine : SeedEngine
        The seed engine for deterministic generation.
    """

    def __init__(self, engine: SeedEngine):
        self.engine = engine

    def generate(self, schema: Schema) -> dict[str, list[Any]]:
        """Generate a complete dataset from the given schema.

        Returns a dict mapping field names to lists of values.
        """
        errors = schema.validate()
        if errors:
            raise ValueError(f"Schema validation failed: {'; '.join(errors)}")

        n = schema.row_count
        data: dict[str, list[Any]] = {}

        # Build dependency graph
        deps = self._build_dependency_graph(schema)
        gen_order = self._topological_sort(schema.fields, deps)

        # Correlation lookup: target_field -> list of Correlations
        corr_by_target: dict[str, list[Correlation]] = defaultdict(list)
        for c in schema.correlations:
            corr_by_target[c.target].append(c)

        # Generate in order
        for field_obj in gen_order:
            correlations = corr_by_target.get(field_obj.name, [])
            if correlations:
                values = self._generate_correlated(field_obj, correlations, data, n)
            else:
                values = self._generate_independent(field_obj, n)

            # Apply nullability
            if field_obj.nullable > 0:
                rng = self.engine.stream(f"{field_obj.name}__null")
                values = [
                    None if rng.random() < field_obj.nullable else v
                    for v in values
                ]

            # Apply transform
            if field_obj.transform is not None:
                values = [
                    field_obj.transform(v) if v is not None else None
                    for v in values
                ]

            # Enforce uniqueness
            if field_obj.unique:
                values = self._enforce_unique(field_obj, values, n)

            data[field_obj.name] = values

        return data

    # ---- Independent field generation ----------------------------------------

    def _generate_independent(self, f: Field, n: int) -> list[Any]:
        """Generate values for a field with no dependencies."""
        rng = self.engine.stream(f.name)

        if f.field_type == FieldType.STRING:
            return self._gen_string(f, rng, n)
        elif f.field_type == FieldType.CHOICE:
            return self._gen_choice(f, rng, n)
        elif f.field_type == FieldType.BOOLEAN:
            return self._gen_boolean(f, rng, n)
        elif f.field_type == FieldType.DATE:
            return self._gen_date(f, rng, n)
        elif f.field_type in (FieldType.INTEGER, FieldType.FLOAT):
            return self._gen_numeric(f, rng, n)
        else:
            raise ValueError(f"Unsupported field type: {f.field_type}")

    def _gen_numeric(self, f: Field, rng: _random_mod.Random, n: int) -> list[Any]:
        dist = f.distribution or Distribution.uniform(0, 100)
        raw = self._sample_distribution(dist, rng, n)

        # Clamp
        if f.min_val is not None:
            raw = [max(f.min_val, v) for v in raw]
        if f.max_val is not None:
            raw = [min(f.max_val, v) for v in raw]

        if f.field_type == FieldType.INTEGER:
            return [int(round(v)) for v in raw]
        return [round(v, 6) for v in raw]

    def _gen_string(self, f: Field, rng: _random_mod.Random, n: int) -> list[Any]:
        prefix = f.prefix or "VAL"
        return [f"{prefix}-{i:06d}" for i in range(1, n + 1)]

    def _gen_choice(self, f: Field, rng: _random_mod.Random, n: int) -> list[Any]:
        if not f.choices:
            raise ValueError(f"Choice field '{f.name}' has no choices")

        # Support weighted choices: [(value, weight), ...]
        if isinstance(f.choices[0], (list, tuple)) and len(f.choices[0]) == 2:
            values = [c[0] for c in f.choices]
            weights = [c[1] for c in f.choices]
        else:
            values = list(f.choices)
            weights = None

        return rng.choices(values, weights=weights, k=n)

    def _gen_boolean(self, f: Field, rng: _random_mod.Random, n: int) -> list[Any]:
        # min_val is overloaded as true_probability threshold
        threshold = f.min_val if f.min_val is not None else 0.5
        return [rng.random() < threshold for _ in range(n)]

    def _gen_date(self, f: Field, rng: _random_mod.Random, n: int) -> list[Any]:
        fmt = f.format_str or "%Y-%m-%d"
        if f.prefix and "|" in f.prefix:
            start_str, end_str = f.prefix.split("|", 1)
            start = datetime.strptime(start_str, "%Y-%m-%d")
            end = datetime.strptime(end_str, "%Y-%m-%d")
        else:
            start = datetime(2020, 1, 1)
            end = datetime(2025, 12, 31)

        span = (end - start).total_seconds()
        dates = []
        for _ in range(n):
            offset = rng.random() * span
            dt = start + timedelta(seconds=offset)
            dates.append(dt.strftime(fmt))
        return dates

    # ---- Correlated field generation -----------------------------------------

    def _generate_correlated(
        self,
        target: Field,
        correlations: list[Correlation],
        data: dict[str, list[Any]],
        n: int,
    ) -> list[Any]:
        """Generate a field whose values depend on other fields."""
        # Use the first correlation as primary (can extend later)
        corr = correlations[0]
        source_values = data[corr.source]

        if corr.correlation_type == CorrelationType.DERIVED:
            func = corr.params.get("func")
            if func is None:
                raise ValueError("DERIVED correlation requires 'func' parameter")
            return [func(sv) if sv is not None else None for sv in source_values]

        elif corr.correlation_type == CorrelationType.CONDITIONAL:
            return self._gen_conditional(target, corr, source_values, n)

        elif corr.correlation_type == CorrelationType.LINEAR:
            return self._gen_linear(target, corr, source_values, n)

        elif corr.correlation_type == CorrelationType.MONOTONIC:
            return self._gen_monotonic(target, corr, source_values, n)

        elif corr.correlation_type == CorrelationType.MUTUAL_EXCLUSIVE:
            return self._gen_mutual_exclusive(target, corr, source_values, n)

        else:
            # Fallback to independent
            return self._generate_independent(target, n)

    def _gen_linear(
        self, target: Field, corr: Correlation,
        source_values: list[Any], n: int,
    ) -> list[Any]:
        """Generate values with a linear correlation to source."""
        rng = self.engine.stream(target.name)
        slope = corr.params.get("slope", 1.0)
        intercept = corr.params.get("intercept", 0.0)
        strength = corr.strength

        # Noise distribution: use target's distribution or default normal
        noise_dist = target.distribution or Distribution.normal(0, 1)

        noise = self._sample_distribution(noise_dist, rng, n)

        # Normalize noise to zero mean
        noise_mean = sum(noise) / len(noise)
        noise = [v - noise_mean for v in noise]

        # Normalize source to get variance
        numeric_src = [float(v) if v is not None else 0.0 for v in source_values]
        src_mean = sum(numeric_src) / len(numeric_src) if numeric_src else 0
        src_std = math.sqrt(sum((v - src_mean) ** 2 for v in numeric_src) / max(len(numeric_src), 1))

        if src_std == 0:
            src_std = 1.0

        noise_std = math.sqrt(sum(v ** 2 for v in noise) / max(len(noise), 1))
        if noise_std == 0:
            noise_std = 1.0

        # Scale noise to achieve desired correlation
        # result = strength * (slope * x + intercept) + (1 - strength) * noise
        results = []
        for i, sv in enumerate(numeric_src):
            signal = slope * sv + intercept
            scaled_noise = noise[i] * (src_std * abs(slope) / noise_std)
            val = strength * signal + (1.0 - strength) * scaled_noise
            results.append(val)

        # Clamp
        if target.min_val is not None:
            results = [max(target.min_val, v) for v in results]
        if target.max_val is not None:
            results = [min(target.max_val, v) for v in results]

        if target.field_type == FieldType.INTEGER:
            return [int(round(v)) for v in results]
        return [round(v, 6) for v in results]

    def _gen_monotonic(
        self, target: Field, corr: Correlation,
        source_values: list[Any], n: int,
    ) -> list[Any]:
        """Generate values that monotonically track source."""
        rng = self.engine.stream(target.name)
        increasing = corr.params.get("increasing", True)
        strength = corr.strength

        # Sort indices by source value
        numeric_src = [float(v) if v is not None else 0.0 for v in source_values]
        sorted_indices = sorted(range(n), key=lambda i: numeric_src[i])
        if not increasing:
            sorted_indices = list(reversed(sorted_indices))

        # Generate target values independently
        independent = self._generate_independent(target, n)

        # Sort independent values
        sorted_independent = sorted(independent, key=lambda x: float(x) if x is not None else 0)
        if not increasing:
            sorted_independent = list(reversed(sorted_independent))

        # Assign sorted values to sorted positions
        monotonic = [None] * n
        for rank, idx in enumerate(sorted_indices):
            monotonic[idx] = sorted_independent[rank]

        # Blend with independent noise based on strength
        rng2 = self.engine.stream(f"{target.name}__monotonic_noise")
        noise = self._generate_independent(target, n)

        results = []
        for i in range(n):
            mv = float(monotonic[i]) if monotonic[i] is not None else 0
            nv = float(noise[i]) if noise[i] is not None else 0
            val = strength * mv + (1 - strength) * nv
            results.append(val)

        if target.field_type == FieldType.INTEGER:
            return [int(round(v)) for v in results]
        return [round(v, 6) for v in results]

    def _gen_conditional(
        self, target: Field, corr: Correlation,
        source_values: list[Any], n: int,
    ) -> list[Any]:
        """Generate values conditioned on source categories."""
        mapping = corr.params.get("mapping", {})
        rng = self.engine.stream(target.name)
        results = []

        for sv in source_values:
            dist = mapping.get(sv)
            if dist is None:
                # Fallback: use target's default distribution
                dist = target.distribution or Distribution.uniform(0, 100)

            val = self._sample_distribution(dist, rng, 1)[0]

            if target.min_val is not None:
                val = max(target.min_val, val)
            if target.max_val is not None:
                val = min(target.max_val, val)

            if target.field_type == FieldType.INTEGER:
                val = int(round(val))
            elif target.field_type == FieldType.FLOAT:
                val = round(val, 6)

            results.append(val)

        return results

    def _gen_mutual_exclusive(
        self, target: Field, corr: Correlation,
        source_values: list[Any], n: int,
    ) -> list[Any]:
        """Generate boolean values that are mutually exclusive with source."""
        results = []
        rng = self.engine.stream(target.name)
        for sv in source_values:
            if sv:
                results.append(False)
            else:
                # When source is False, target can be True or False
                results.append(rng.random() < 0.5)
        return results

    # ---- Distribution sampling -----------------------------------------------

    def _sample_distribution(
        self, dist: Distribution, rng: _random_mod.Random, n: int,
    ) -> list[float]:
        """Draw n samples from the given distribution using rng."""
        kind = dist.kind
        p = dist.params

        if kind == DistributionKind.UNIFORM:
            low, high = p.get("low", 0), p.get("high", 1)
            return [rng.uniform(low, high) for _ in range(n)]

        elif kind == DistributionKind.NORMAL:
            mean, std = p.get("mean", 0), p.get("std", 1)
            return [rng.gauss(mean, std) for _ in range(n)]

        elif kind == DistributionKind.LOGNORMAL:
            mean, sigma = p.get("mean", 0), p.get("sigma", 1)
            return [rng.lognormvariate(mean, sigma) for _ in range(n)]

        elif kind == DistributionKind.EXPONENTIAL:
            scale = p.get("scale", 1)
            return [rng.expovariate(1.0 / scale) for _ in range(n)]

        elif kind == DistributionKind.BETA:
            alpha, beta = p.get("alpha", 2), p.get("beta", 5)
            return [rng.betavariate(alpha, beta) for _ in range(n)]

        elif kind == DistributionKind.POISSON:
            lam = p.get("lam", 5)
            # Python's random doesn't have Poisson, use inverse CDF
            return [self._poisson_sample(rng, lam) for _ in range(n)]

        elif kind == DistributionKind.ZIPF:
            a = p.get("a", 2)
            return [self._zipf_sample(rng, a) for _ in range(n)]

        else:
            raise ValueError(f"Unsupported distribution: {kind}")

    @staticmethod
    def _poisson_sample(rng: _random_mod.Random, lam: float) -> float:
        """Sample from Poisson distribution using Knuth's algorithm."""
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= rng.random()
            if p <= L:
                return float(k - 1)

    @staticmethod
    def _zipf_sample(rng: _random_mod.Random, a: float, n_max: int = 1000) -> float:
        """Sample from Zipf distribution using rejection method."""
        # Simple approximation
        u = rng.random()
        return float(int((u ** (-1.0 / (a - 1.0))) if a > 1 else 1))

    # ---- Uniqueness enforcement ----------------------------------------------

    def _enforce_unique(
        self, f: Field, values: list[Any], n: int,
    ) -> list[Any]:
        """Ensure all non-None values are unique."""
        seen: set = set()
        result = []
        counter = n + 1

        for v in values:
            if v is None:
                result.append(None)
                continue
            if v in seen:
                # Generate a replacement
                if f.field_type == FieldType.STRING:
                    prefix = f.prefix or "VAL"
                    while True:
                        replacement = f"{prefix}-{counter:06d}"
                        counter += 1
                        if replacement not in seen:
                            seen.add(replacement)
                            result.append(replacement)
                            break
                else:
                    # For numeric types, add small perturbation
                    rng = self.engine.stream(f"{f.name}__unique")
                    while True:
                        perturbed = v + rng.gauss(0, max(abs(v) * 0.01, 1))
                        if f.field_type == FieldType.INTEGER:
                            perturbed = int(round(perturbed))
                        key = perturbed
                        if key not in seen:
                            seen.add(key)
                            result.append(perturbed)
                            break
            else:
                seen.add(v)
                result.append(v)

        return result

    # ---- Dependency graph ----------------------------------------------------

    def _build_dependency_graph(
        self, schema: Schema,
    ) -> dict[str, set[str]]:
        """Build a map of field -> set of fields it depends on."""
        deps: dict[str, set[str]] = {f.name: set() for f in schema.fields}
        for c in schema.correlations:
            deps[c.target].add(c.source)
        return deps

    def _topological_sort(
        self, fields: list[Field], deps: dict[str, set[str]],
    ) -> list[Field]:
        """Sort fields so dependencies are generated first."""
        field_map = {f.name: f for f in fields}
        visited: set[str] = set()
        order: list[str] = []
        temp_mark: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in temp_mark:
                raise ValueError(f"Circular dependency detected involving '{name}'")
            temp_mark.add(name)
            for dep in deps.get(name, set()):
                visit(dep)
            temp_mark.discard(name)
            visited.add(name)
            order.append(name)

        for f in fields:
            visit(f.name)

        return [field_map[n] for n in order]
