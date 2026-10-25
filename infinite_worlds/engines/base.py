"""Engine interface and registry.

An engine is a pure function of (scene, time, controls, size): the same inputs always
give the same frame. That property is what makes chunked, crash-safe resume possible
and lets frames be rendered in any order or in parallel.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VisualEngine(ABC):
    name: str = ""
    version: str = "0"

    @abstractmethod
    def sample_scene(self, rng, ctx: dict) -> dict:
        """Create JSON-serialisable scene parameters.

        ctx keys: duration, intensity (0..1 mean), quiet (bool), climax (bool),
        direction (str), avoid_mode (str|None), progress_knots (list[float]).
        """

    @abstractmethod
    def render(self, scene: dict, t: float, controls: dict, width: int, height: int,
               supersample: int = 1) -> np.ndarray:
        """Return float32 RGB image (height, width, 3) in 0..1. t = scene-local seconds."""

    @abstractmethod
    def fingerprint(self, scene: dict) -> list[float]:
        """Coarse vector describing the scene, for the novelty memory."""

    def describe(self, scene: dict) -> str:
        return self.name


class VisualEngineRegistry:
    def __init__(self):
        self._engines: dict[str, VisualEngine] = {}

    def register(self, engine: VisualEngine) -> None:
        self._engines[engine.name] = engine

    def get(self, name: str) -> VisualEngine:
        return self._engines[name]

    def names(self) -> list[str]:
        return sorted(self._engines)

    def versions(self) -> dict[str, str]:
        return {n: e.version for n, e in sorted(self._engines.items())}


def default_registry() -> VisualEngineRegistry:
    from .fractal import FractalEngine
    reg = VisualEngineRegistry()
    reg.register(FractalEngine())
    return reg
