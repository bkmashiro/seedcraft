"""Tests for data exporters."""

import json
import os
import tempfile
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.exporters import to_csv, to_json, to_dict_list


SAMPLE_DATA = {
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "score": [95.5, 87.3, 91.0],
}


class TestDictList:
    def test_basic(self):
        rows = to_dict_list(SAMPLE_DATA)
        assert len(rows) == 3
        assert rows[0] == {"id": 1, "name": "Alice", "score": 95.5}
        assert rows[2]["name"] == "Charlie"

    def test_empty(self):
        assert to_dict_list({}) == []

    def test_with_nulls(self):
        data = {"x": [1, None, 3]}
        rows = to_dict_list(data)
        assert rows[1]["x"] is None


class TestCSV:
    def test_csv_string(self):
        csv_str = to_csv(SAMPLE_DATA)
        lines = csv_str.strip().split("\n")
        assert "id" in lines[0] and "name" in lines[0] and "score" in lines[0]
        assert "Alice" in lines[1]

    def test_csv_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            result = to_csv(SAMPLE_DATA, path=path)
            assert result == path
            with open(path) as f:
                content = f.read()
            assert "Bob" in content
        finally:
            os.unlink(path)

    def test_empty(self):
        assert to_csv({}) == ""


class TestJSON:
    def test_json_records(self):
        result = to_json(SAMPLE_DATA, orient="records")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 3
        assert parsed[0]["name"] == "Alice"

    def test_json_columns(self):
        result = to_json(SAMPLE_DATA, orient="columns")
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert parsed["name"] == ["Alice", "Bob", "Charlie"]

    def test_json_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            result = to_json(SAMPLE_DATA, path=path)
            assert result == path
            with open(path) as f:
                parsed = json.load(f)
            assert len(parsed) == 3
        finally:
            os.unlink(path)

    def test_invalid_orient(self):
        with pytest.raises(ValueError, match="Unknown orient"):
            to_json(SAMPLE_DATA, orient="invalid")
