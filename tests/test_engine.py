"""Tests for the SeedEngine."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine import SeedEngine


class TestSeedEngineBasics:
    def test_requires_at_least_one_seed(self):
        with pytest.raises(ValueError, match="At least one seed"):
            SeedEngine([])

    def test_single_seed(self):
        engine = SeedEngine([42])
        assert engine.master_seed != 0

    def test_multiple_seeds(self):
        engine = SeedEngine([3813, 7889, 6140, 439])
        assert engine.master_seed != 0

    def test_seed_order_matters(self):
        e1 = SeedEngine([1, 2, 3])
        e2 = SeedEngine([3, 2, 1])
        assert e1.master_seed != e2.master_seed

    def test_different_seeds_different_masters(self):
        e1 = SeedEngine([100])
        e2 = SeedEngine([200])
        assert e1.master_seed != e2.master_seed


class TestSeedEngineStreams:
    def test_stream_returns_random_instance(self):
        engine = SeedEngine([42])
        rng = engine.stream("test")
        # Should be able to generate random numbers
        val = rng.random()
        assert 0 <= val <= 1

    def test_same_name_returns_same_stream(self):
        engine = SeedEngine([42])
        s1 = engine.stream("field_a")
        s2 = engine.stream("field_a")
        assert s1 is s2

    def test_different_names_different_streams(self):
        engine = SeedEngine([42])
        s1 = engine.stream("field_a")
        s2 = engine.stream("field_b")
        assert s1 is not s2
        # They should produce different sequences
        v1 = s1.random()
        v2 = s2.random()
        assert v1 != v2

    def test_deterministic_across_instances(self):
        """Two engines with the same seeds produce the same values."""
        e1 = SeedEngine([3813, 7889])
        e2 = SeedEngine([3813, 7889])

        vals1 = [e1.stream("x").random() for _ in range(10)]
        vals2 = [e2.stream("x").random() for _ in range(10)]
        assert vals1 == vals2

    def test_stream_independence(self):
        """Drawing from one stream doesn't affect another."""
        e1 = SeedEngine([42])
        e2 = SeedEngine([42])

        # In e1, draw from "a" then "b"
        _ = e1.stream("a").random()
        val_b1 = e1.stream("b").random()

        # In e2, just draw from "b"
        val_b2 = e2.stream("b").random()

        assert val_b1 == val_b2


class TestSeedEngineFork:
    def test_fork_creates_different_engine(self):
        parent = SeedEngine([42])
        child = parent.fork("child_namespace")
        assert child.master_seed != parent.master_seed

    def test_fork_is_deterministic(self):
        e1 = SeedEngine([42])
        e2 = SeedEngine([42])
        c1 = e1.fork("ns")
        c2 = e2.fork("ns")
        assert c1.master_seed == c2.master_seed

    def test_different_namespaces_different_children(self):
        parent = SeedEngine([42])
        c1 = parent.fork("ns_a")
        c2 = parent.fork("ns_b")
        assert c1.master_seed != c2.master_seed


class TestSeedEngineReset:
    def test_reset_clears_streams(self):
        engine = SeedEngine([42])
        engine.stream("a")
        engine.stream("b")
        assert len(engine._streams) == 2
        engine.reset()
        assert len(engine._streams) == 0

    def test_reset_reproduces_values(self):
        engine = SeedEngine([42])
        v1 = engine.stream("x").random()
        engine.reset()
        v2 = engine.stream("x").random()
        assert v1 == v2


class TestSeedEngineRepr:
    def test_repr_contains_info(self):
        engine = SeedEngine([42, 99])
        engine.stream("test")
        r = repr(engine)
        assert "42" in r
        assert "99" in r
        assert "test" in r


class TestSeedEngineWithProjectSeeds:
    """Tests using the actual seed numbers from the project."""

    SEEDS = [3813, 7889, 6140, 439, 5990, 2191, 5982, 6462,
             7318, 2050, 8243, 2900, 1314, 9843, 8348, 766]

    def test_all_sixteen_seeds(self):
        engine = SeedEngine(self.SEEDS)
        assert engine.master_seed != 0
        # Generate some values
        vals = [engine.stream("test").random() for _ in range(100)]
        assert len(set(vals)) == 100  # All unique

    def test_subset_seeds_differ(self):
        e_full = SeedEngine(self.SEEDS)
        e_half = SeedEngine(self.SEEDS[:8])
        assert e_full.master_seed != e_half.master_seed
