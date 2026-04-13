"""Tests for the CLI interface."""

import json
import os
import tempfile
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cli import main


class TestCLIList:
    def test_list(self, capsys):
        ret = main(["list"])
        assert ret == 0
        output = capsys.readouterr().out
        assert "ecommerce_customers" in output
        assert "sensor_readings" in output

class TestCLIInfo:
    def test_info(self, capsys):
        ret = main(["info", "ecommerce_customers"])
        assert ret == 0
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert parsed["name"] == "ecommerce_customers"

class TestCLIGenerate:
    def test_generate_csv(self, capsys):
        ret = main(["generate", "sensor_readings", "-n", "10", "-f", "csv"])
        assert ret == 0
        output = capsys.readouterr().out
        lines = output.strip().split("\n")
        assert len(lines) >= 11  # header + 10 rows
        assert "reading_id" in lines[0]

    def test_generate_json(self, capsys):
        ret = main(["generate", "ecommerce_customers", "-n", "5", "-f", "json"])
        assert ret == 0
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert len(parsed) == 5

    def test_generate_to_file(self, capsys):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            ret = main(["generate", "student_records", "-n", "20", "-o", path])
            assert ret == 0
            with open(path) as f:
                content = f.read()
            assert "student_id" in content
        finally:
            os.unlink(path)

    def test_generate_custom_seeds(self, capsys):
        ret = main(["generate", "sensor_readings", "-n", "5", "-s", "42", "99"])
        assert ret == 0

    def test_generate_deterministic(self, capsys):
        main(["generate", "ecommerce_customers", "-n", "10", "-f", "json",
              "-s", "3813", "7889"])
        out1 = capsys.readouterr().out

        main(["generate", "ecommerce_customers", "-n", "10", "-f", "json",
              "-s", "3813", "7889"])
        out2 = capsys.readouterr().out

        assert out1 == out2


class TestCLIAnalyze:
    def test_analyze(self, capsys):
        ret = main(["analyze", "ecommerce_customers", "-n", "100"])
        assert ret == 0
        output = capsys.readouterr().out
        assert "Column Statistics" in output
        assert "Correlation Matrix" in output


class TestCLIHelp:
    def test_no_args(self, capsys):
        ret = main([])
        assert ret == 0
