"""SeedEngine: reproducible audio identity + per-generation identity.

AudioSeed      = SHA256(content_hash | analysis_version | feature_digest)
GenerationSeed = SHA256(audio_seed | generation_id | time_component | engine_version)

The PRNG is a SHA-256 counter-mode generator implemented here, so a given seed
produces identical choices on any machine, OS, or numpy version (numpy does not
guarantee Generator streams across releases). Named sub-streams (`fork`) keep
consumers independent: adding a new random draw to the camera never shifts the
palette or the event schedule.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from .audio_analyzer import AudioAnalysis


def _sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def feature_digest(feature_digest_input: dict) -> str:
    return _sha(json.dumps(feature_digest_input, sort_keys=True, separators=(",", ":")))


@dataclass(frozen=True)
class Seeds:
    audio_seed: str
    generation_seed: str
    generation_id: str
    time_component: str
    engine_version: str
    deterministic: bool

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class DeterministicRNG:
    """SHA-256 counter-mode PRNG with named, independent sub-streams."""

    def __init__(self, seed_hex: str, label: str = ""):
        self._key = _sha(seed_hex, label)
        self._label = label
        self._counter = 0
        self._buf = b""

    def _bytes(self, n: int) -> bytes:
        while len(self._buf) < n:
            self._buf += hashlib.sha256(
                self._key.encode() + self._counter.to_bytes(8, "big")).digest()
            self._counter += 1
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def fork(self, name: str) -> "DeterministicRNG":
        return DeterministicRNG(self._key, f"{self._label}/{name}")

    def random(self) -> float:
        return (int.from_bytes(self._bytes(8), "big") >> 11) / float(1 << 53)

    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.random()

    def integer(self, a: int, b: int) -> int:
        """Inclusive range."""
        return a + int(self.random() * (b - a + 1))

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        u1, u2 = max(self.random(), 1e-12), self.random()
        return mu + sigma * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    def choice(self, seq):
        return seq[self.integer(0, len(seq) - 1)]

    def weighted_choice(self, items, weights):
        total = float(sum(weights))
        r = self.random() * total
        acc = 0.0
        for it, w in zip(items, weights):
            acc += w
            if r <= acc:
                return it
        return items[-1]


class SeedEngine:
    def __init__(self, engine_version: str):
        self.engine_version = engine_version

    @staticmethod
    def audio_seed(analysis: AudioAnalysis) -> str:
        return _sha(analysis.content_hash, analysis.analysis_version,
                    feature_digest(analysis.feature_digest_input))

    def make_seeds(self, analysis: AudioAnalysis, generation_id: str = "g1",
                   deterministic: bool = True, time_component: str | None = None) -> Seeds:
        a = self.audio_seed(analysis)
        if deterministic:
            tc = ""
        else:
            if time_component is None:
                import datetime as _dt
                time_component = _dt.datetime.now(_dt.timezone.utc).isoformat()
            tc = time_component
        g = _sha(a, generation_id, tc, self.engine_version)
        return Seeds(a, g, generation_id, tc, self.engine_version, deterministic)

    @staticmethod
    def rng(seeds: Seeds, stream: str) -> DeterministicRNG:
        return DeterministicRNG(seeds.generation_seed, stream)
