"""Export generated datasets to various formats."""

from __future__ import annotations

import csv
import io
import json
from typing import Any


def to_dict_list(data: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Convert columnar data to a list of row dictionaries."""
    if not data:
        return []
    keys = list(data.keys())
    n = len(data[keys[0]])
    return [{k: data[k][i] for k in keys} for i in range(n)]


def to_csv(data: dict[str, list[Any]], path: str | None = None) -> str:
    """Export data as CSV.  If path is given, write to file and return path.
    Otherwise return the CSV as a string."""
    rows = to_dict_list(data)
    if not rows:
        return ""

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    csv_str = output.getvalue()

    if path:
        with open(path, "w", newline="") as f:
            f.write(csv_str)
        return path

    return csv_str


def to_json(
    data: dict[str, list[Any]],
    path: str | None = None,
    orient: str = "records",
    indent: int = 2,
) -> str:
    """Export data as JSON.

    Parameters
    ----------
    orient : str
        "records" -> list of dicts, "columns" -> dict of lists.
    """
    if orient == "records":
        payload = to_dict_list(data)
    elif orient == "columns":
        payload = data
    else:
        raise ValueError(f"Unknown orient: {orient}")

    json_str = json.dumps(payload, indent=indent, default=str)

    if path:
        with open(path, "w") as f:
            f.write(json_str)
        return path

    return json_str
