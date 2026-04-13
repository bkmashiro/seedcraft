"""Statistical analysis of generated data — verify correlations hold."""

from __future__ import annotations

import math
from typing import Any


def describe_column(values: list[Any]) -> dict[str, Any]:
    """Compute descriptive statistics for a single column."""
    non_null = [v for v in values if v is not None]
    n = len(non_null)

    if n == 0:
        return {"count": len(values), "non_null": 0, "null_rate": 1.0}

    null_rate = 1.0 - n / len(values) if values else 0.0

    # Check if numeric
    try:
        nums = [float(v) for v in non_null]
        is_numeric = True
    except (ValueError, TypeError):
        is_numeric = False

    if is_numeric:
        mean = sum(nums) / n
        variance = sum((x - mean) ** 2 for x in nums) / n if n > 1 else 0
        std = math.sqrt(variance)
        sorted_nums = sorted(nums)
        median = sorted_nums[n // 2] if n % 2 else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2

        return {
            "count": len(values),
            "non_null": n,
            "null_rate": round(null_rate, 4),
            "type": "numeric",
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(min(nums), 4),
            "max": round(max(nums), 4),
            "median": round(median, 4),
            "p25": round(sorted_nums[n // 4], 4),
            "p75": round(sorted_nums[3 * n // 4], 4),
        }
    else:
        # Categorical
        freq: dict[Any, int] = {}
        for v in non_null:
            freq[v] = freq.get(v, 0) + 1

        sorted_freq = sorted(freq.items(), key=lambda x: -x[1])
        unique = len(freq)

        return {
            "count": len(values),
            "non_null": n,
            "null_rate": round(null_rate, 4),
            "type": "categorical",
            "unique": unique,
            "top_values": sorted_freq[:5],
        }


def describe_dataset(data: dict[str, list[Any]]) -> dict[str, dict]:
    """Compute descriptive statistics for all columns."""
    return {name: describe_column(values) for name, values in data.items()}


def pearson_correlation(
    xs: list[Any], ys: list[Any],
) -> float | None:
    """Compute Pearson correlation between two numeric columns.

    Returns None if either column is non-numeric or has zero variance.
    """
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x is not None and y is not None
    ]

    if len(pairs) < 3:
        return None

    try:
        x_vals = [p[0] for p in pairs]
        y_vals = [p[1] for p in pairs]
    except (ValueError, TypeError):
        return None

    n = len(pairs)
    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n

    cov = sum((x - x_mean) * (y - y_mean) for x, y in pairs) / n
    x_std = math.sqrt(sum((x - x_mean) ** 2 for x in x_vals) / n)
    y_std = math.sqrt(sum((y - y_mean) ** 2 for y in y_vals) / n)

    if x_std == 0 or y_std == 0:
        return None

    return round(cov / (x_std * y_std), 4)


def correlation_matrix(data: dict[str, list[Any]]) -> dict[str, dict[str, float | None]]:
    """Compute pairwise Pearson correlations for all numeric columns."""
    # Filter to numeric columns
    numeric_cols = {}
    for name, values in data.items():
        non_null = [v for v in values if v is not None]
        if non_null:
            try:
                [float(v) for v in non_null[:5]]  # Quick type check
                numeric_cols[name] = values
            except (ValueError, TypeError):
                pass

    names = list(numeric_cols.keys())
    matrix: dict[str, dict[str, float | None]] = {}

    for i, a in enumerate(names):
        matrix[a] = {}
        for j, b in enumerate(names):
            if i == j:
                matrix[a][b] = 1.0
            elif j < i:
                matrix[a][b] = matrix[b][a]
            else:
                matrix[a][b] = pearson_correlation(numeric_cols[a], numeric_cols[b])

    return matrix


def verify_correlations(
    data: dict[str, list[Any]],
    expected: list[tuple[str, str, float]],
    tolerance: float = 0.25,
) -> list[dict[str, Any]]:
    """Verify that expected correlations hold in the generated data.

    Parameters
    ----------
    expected : list of (source, target, expected_correlation)
    tolerance : float
        Acceptable deviation from expected correlation.

    Returns
    -------
    list of verification results with pass/fail status.
    """
    results = []
    for source, target, expected_corr in expected:
        if source not in data or target not in data:
            results.append({
                "source": source,
                "target": target,
                "expected": expected_corr,
                "actual": None,
                "passed": False,
                "reason": "field not found",
            })
            continue

        actual = pearson_correlation(data[source], data[target])
        if actual is None:
            results.append({
                "source": source,
                "target": target,
                "expected": expected_corr,
                "actual": None,
                "passed": False,
                "reason": "could not compute correlation",
            })
            continue

        passed = abs(actual - expected_corr) <= tolerance
        results.append({
            "source": source,
            "target": target,
            "expected": expected_corr,
            "actual": actual,
            "deviation": round(abs(actual - expected_corr), 4),
            "passed": passed,
        })

    return results


def format_stats(stats: dict[str, dict]) -> str:
    """Pretty-print dataset statistics."""
    lines = []
    for name, s in stats.items():
        lines.append(f"\n  {name}:")
        lines.append(f"    count={s['count']}, non_null={s['non_null']}, null_rate={s['null_rate']}")
        if s.get("type") == "numeric":
            lines.append(f"    mean={s['mean']}, std={s['std']}, min={s['min']}, max={s['max']}")
            lines.append(f"    median={s['median']}, p25={s['p25']}, p75={s['p75']}")
        elif s.get("type") == "categorical":
            lines.append(f"    unique={s['unique']}, top={s['top_values']}")
    return "\n".join(lines)


def format_correlation_matrix(matrix: dict[str, dict[str, float | None]]) -> str:
    """Pretty-print a correlation matrix."""
    if not matrix:
        return "(no numeric columns)"

    names = list(matrix.keys())
    # Header
    col_width = max(len(n) for n in names) + 2
    header = " " * col_width + "".join(n[:8].rjust(9) for n in names)
    lines = [header]

    for name in names:
        row = name.ljust(col_width)
        for other in names:
            val = matrix[name].get(other)
            if val is None:
                row += "     N/A "
            else:
                row += f"  {val:+.4f} "
        lines.append(row)

    return "\n".join(lines)
