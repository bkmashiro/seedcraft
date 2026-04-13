"""Seed engine — deterministic random number generation with multi-stream support.

The engine takes a list of seed numbers and derives independent random streams
for each field using a hash-based stream splitting approach.  This ensures:
1. Determinism: same seeds always produce the same data.
2. Independence: changing one field's schema doesn't affect other fields.
3. Reproducibility: adding a field doesn't change existing fields' values.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Sequence

import random as _random_mod


class SeedEngine:
    """Manages deterministic random streams derived from seed numbers.

    The engine combines the provided seed numbers into a master seed using
    a hash function, then derives per-field streams by hashing the master
    seed with the field name.  Each stream is a standard Python Random
    instance, giving access to all distribution methods.

    Parameters
    ----------
    seeds : sequence of int
        One or more seed numbers.  The order matters.
    """

    def __init__(self, seeds: Sequence[int]):
        if not seeds:
            raise ValueError("At least one seed number is required")
        self._original_seeds = list(seeds)
        self._master_seed = self._combine_seeds(seeds)
        self._streams: dict[str, _random_mod.Random] = {}

    @staticmethod
    def _combine_seeds(seeds: Sequence[int]) -> int:
        """Combine multiple seeds into a single master seed via SHA-256."""
        h = hashlib.sha256()
        for s in seeds:
            h.update(struct.pack(">q", s))
        # Take first 8 bytes as a 64-bit seed
        return struct.unpack(">Q", h.digest()[:8])[0]

    @property
    def master_seed(self) -> int:
        return self._master_seed

    def stream(self, name: str) -> _random_mod.Random:
        """Get or create a deterministic random stream for the given name.

        Streams are cached, so calling stream("age") twice returns the
        same Random instance (in the same state if no values drawn).
        """
        if name not in self._streams:
            self._streams[name] = self._make_stream(name)
        return self._streams[name]

    def _make_stream(self, name: str) -> _random_mod.Random:
        """Create a new Random instance seeded by master_seed + name."""
        h = hashlib.sha256()
        h.update(struct.pack(">Q", self._master_seed))
        h.update(name.encode("utf-8"))
        stream_seed = struct.unpack(">Q", h.digest()[:8])[0]
        return _random_mod.Random(stream_seed)

    def fork(self, namespace: str) -> SeedEngine:
        """Create a child engine with a derived master seed.

        Useful for generating nested/related datasets deterministically.
        """
        h = hashlib.sha256()
        h.update(struct.pack(">Q", self._master_seed))
        h.update(namespace.encode("utf-8"))
        child_seed = struct.unpack(">Q", h.digest()[:8])[0]
        engine = SeedEngine.__new__(SeedEngine)
        engine._original_seeds = self._original_seeds
        engine._master_seed = child_seed
        engine._streams = {}
        return engine

    def reset(self) -> None:
        """Reset all streams (re-derive from master seed)."""
        self._streams.clear()

    def __repr__(self) -> str:
        return (
            f"SeedEngine(seeds={self._original_seeds}, "
            f"master=0x{self._master_seed:016x}, "
            f"streams={list(self._streams.keys())})"
        )
