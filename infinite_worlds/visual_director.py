"""VisualDirector: turn audio analysis + seeds into a coherent visual timeline.

Output is a VisualPlan: scenes (engine + parameters + time span), transitions,
control curves (audio-derived, sampled at the video frame rate), and a list of
events. The plan is a pure function of (analysis, seeds, settings) so it can be
recomputed identically after a crash.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .audio_analyzer import AudioAnalysis, FRAME_RATE
from .engines.base import VisualEngineRegistry
from .seed_engine import SeedEngine, Seeds

MIN_SCENE_S = 10.0          # default; "classic" style raises it via settings
PROGRESS_HZ = 10.0


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


@dataclass
class VisualPlan:
    duration: float
    fps: int
    scenes: list[dict]            # each: engine, start, end, scene params
    transitions: list[dict]       # between scenes[i] and scenes[i+1]
    events: list[dict]
    controls: dict                # name -> np.ndarray at fps (not serialised in JSON)
    fade_in: float = 1.5
    fade_out: float = 4.0
    settings: dict = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        return {
            "duration": self.duration, "fps": self.fps, "scenes": self.scenes,
            "transitions": self.transitions, "events": self.events,
            "fade_in": self.fade_in, "fade_out": self.fade_out, "settings": self.settings,
        }

    def plan_hash(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(self.to_json_dict(), sort_keys=True).encode())
        for k in sorted(self.controls):
            h.update(np.round(self.controls[k], 4).tobytes())
        return h.hexdigest()

    # -- evaluated by the renderer ------------------------------------------
    def controls_at(self, t: float) -> dict:
        i = int(min(max(round(t * self.fps), 0), len(next(iter(self.controls.values()))) - 1))
        return {k: float(v[i]) for k, v in self.controls.items()}

    def active_scenes(self, t: float) -> list[tuple[int, float]]:
        """[(scene_index, mix_weight)] for time t (weights sum to 1)."""
        out = []
        for i, sc in enumerate(self.scenes):
            if sc["start"] - 1e-9 <= t <= sc["end"] + 1e-9:
                out.append(i)
        if not out:
            return [(len(self.scenes) - 1, 1.0)] if t >= self.duration else [(0, 1.0)]
        if len(out) == 1:
            return [(out[0], 1.0)]
        a, b = out[0], out[1]
        tr = self.transitions[a]
        u = _smoothstep((t - tr["start"]) / max(tr["end"] - tr["start"], 1e-6))
        return [(a, 1.0 - u), (b, u)]

    def global_fade(self, t: float) -> float:
        f = 1.0
        if self.fade_in > 0:
            f *= _smoothstep(t / self.fade_in)
        if self.fade_out > 0:
            f *= _smoothstep((self.duration - t) / self.fade_out)
        return f


class VisualDirector:
    def __init__(self, registry: VisualEngineRegistry, settings: dict | None = None):
        self.registry = registry
        settings = dict(settings or {})
        style = settings.get("style", "classic")
        # Pacing follows the style: classic = few long scenes with slow fades (zoom in / zoom out);
        # moody = shorter scenes and quicker fades.
        pacing = ({"min_scene_s": 30.0, "scene_s": (38.0, 60.0), "fade_s": (6.0, 10.0)} if style == "classic"
                  else {"min_scene_s": MIN_SCENE_S, "scene_s": (20.0, 34.0), "fade_s": (4.0, 7.0)})
        self.settings = {"direction": "auto", "engines": None, "style": style, **pacing}
        self.settings.update(settings)

    # ------------------------------------------------------------------
    def build_plan(self, analysis: AudioAnalysis, seeds: Seeds, fps: int) -> VisualPlan:
        duration = float(analysis.metadata["analysed_duration"])   # decoded length, not the header claim
        rng = SeedEngine.rng(seeds, "director")
        controls = self._controls(analysis, fps, duration)
        intensity_s = analysis.series["intensity"]

        bounds = self._scene_bounds(analysis, rng.fork("bounds"), duration)
        events: list[dict] = []

        # climax: the loudest sustained point, only for tracks long enough to build to one
        cl_time = None
        sm = ndimage.gaussian_filter1d(intensity_s.astype(float), FRAME_RATE * 3)
        if duration > 40 and sm.max() > 0.7:
            cl_time = float(np.argmax(sm) / FRAME_RATE)
            events.append({"type": "climax", "time": round(cl_time, 2),
                           "intensity": round(float(sm.max()), 3)})

        # transitions centred on the boundaries
        transitions = []
        for k, (bt, kind) in enumerate(bounds[1:-1]):
            f0, f1 = self.settings["fade_s"]
            d = rng.uniform(f0, f1) if kind == "section" else rng.uniform(0.8 * f0, 0.9 * f1)
            if cl_time is not None and abs(bt - cl_time) < 6:
                d = max(d, 0.9 * f1)
            d = min(d, 0.8 * min(bounds[k + 1][0] - bounds[k][0], bounds[k + 2][0] - bounds[k + 1][0]))
            transitions.append({"center": round(bt, 3), "start": round(bt - d / 2, 3),
                                "end": round(bt + d / 2, 3),
                                "type": rng.weighted_choice(["crossfade", "dissolve"], [0.65, 0.35]),
                                "boundary_kind": kind})
            events.append({"type": "scene_change" if kind == "sub" else "section_change",
                           "time": round(bt, 2)})

        scenes: list[dict] = []
        memory: list[list[float]] = []      # what has already been shown in this video
        avoid_targets: list[tuple[float, float]] = []
        prev_mode = None
        engines = self.settings.get("engines") or self.registry.names()
        for i in range(len(bounds) - 1):
            span_start = bounds[i][0] - (transitions[i - 1]["end"] - transitions[i - 1]["start"]) / 2 if i > 0 else 0.0
            span_end = bounds[i + 1][0] + (transitions[i]["end"] - transitions[i]["start"]) / 2 if i < len(transitions) else duration
            span_start, span_end = max(0.0, span_start), min(duration, span_end)
            dur = span_end - span_start
            a, b = int(span_start * FRAME_RATE), max(int(span_end * FRAME_RATE), int(span_start * FRAME_RATE) + 2)
            seg_int = intensity_s[a:b]
            mean_int = float(seg_int.mean())
            is_climax = cl_time is not None and span_start <= cl_time <= span_end
            ctx = {
                "duration": dur, "intensity": mean_int, "quiet": mean_int < 0.28,
                "climax": is_climax, "direction": self.settings["direction"],
                "avoid_mode": prev_mode, "avoid_targets": avoid_targets,
                "style": self.settings["style"],
                # in -> fade -> out -> fade -> in ...: the camera alternates between diving deep and surfacing
                "zoom_dir": "in" if i % 2 == 0 else "out",
            }
            ctx["progress_knots"] = self._progress_knots(seg_int, dur, is_climax)
            ctx["progress_rate_hz"] = PROGRESS_HZ

            erng = rng.fork(f"scene{i}")
            engine_name = erng.choice(engines)
            engine = self.registry.get(engine_name)
            best, best_d = None, -1.0
            for attempt in range(8):                        # novelty: avoid recent visual states
                scene = engine.sample_scene(erng.fork(f"try{attempt}"), ctx)
                fp = np.array(engine.fingerprint(scene))
                d = min((float(np.linalg.norm(fp - np.array(m))) for m in memory), default=9.9)
                if (scene.get("discovery_score") or 1.0) < 0.08:
                    d *= 0.1                          # bare regions are a last resort
                if hasattr(engine, "validate_scene") and not engine.validate_scene(scene)["ok"]:
                    d *= 0.02                         # path passes through black/flat views: reject
                if d > best_d:
                    best, best_d = (scene, fp), d
                if d > 0.9:
                    break
            scene, fp = best
            memory.append(list(fp))
            if "target" in scene:
                avoid_targets.append(tuple(scene["target"]))
            prev_mode = scene.get("mode")
            scenes.append({"index": i, "engine": engine_name, "start": round(span_start, 3),
                           "end": round(span_end, 3), "mean_intensity": round(mean_int, 3),
                           "climax": bool(is_climax), "novelty_distance": round(best_d, 3),
                           "scene": scene})
            events.append({"type": "discovery", "time": round(span_start, 2),
                           "engine": engine_name, "what": engine.describe(scene),
                           "score": scene.get("discovery_score")})
        if cl_time is not None:
            events.append({"type": "resolution", "time": round(max(0.0, duration - 4.0), 2)})
        events.sort(key=lambda e: e["time"])
        return VisualPlan(duration=duration, fps=fps, scenes=scenes, transitions=transitions,
                          events=events, controls=controls, settings=dict(self.settings))

    # ------------------------------------------------------------------
    def _scene_bounds(self, analysis: AudioAnalysis, rng, duration: float):
        """[(time, kind)] starting at 0 and ending at duration."""
        inten = analysis.series["intensity"]
        pts: list[tuple[float, str]] = [(0.0, "start")]
        for sec in analysis.sections:
            if sec["start"] > 0.5:
                pts.append((float(sec["start"]), "section"))
        pts.append((duration, "end"))
        out: list[tuple[float, str]] = [pts[0]]
        for (a, _), (b, kb) in zip(pts[:-1], pts[1:]):
            seg = b - a
            mean_i = float(inten[int(a * FRAME_RATE):max(int(b * FRAME_RATE), int(a * FRAME_RATE) + 1)].mean())
            s0, s1 = self.settings["scene_s"]
            target = rng.uniform(s0, s1) * (1.0 - 0.2 * mean_i)
            n = max(1, int(round(seg / target)))
            for j in range(1, n):
                t = a + seg * (j + rng.uniform(-0.15, 0.15)) / n
                out.append((t, "sub"))
            out.append((b, kb))
        # snap to confident beats, then merge scenes shorter than MIN_SCENE_S
        beats = np.array(analysis.beats) if (analysis.tempo["confidence"] > 0.3 and analysis.beats) else None
        snapped = [out[0]]
        for t, k in out[1:-1]:
            if beats is not None:
                j = int(np.argmin(np.abs(beats - t)))
                if abs(beats[j] - t) < 0.6:
                    t = float(beats[j])
            snapped.append((t, k))
        snapped.append(out[-1])
        merged = [snapped[0]]
        for t, k in snapped[1:-1]:
            min_s = float(self.settings["min_scene_s"])
            if t - merged[-1][0] < min_s or snapped[-1][0] - t < min_s:
                continue
            merged.append((t, k))
        merged.append(snapped[-1])
        return merged

    @staticmethod
    def _progress_knots(seg_intensity: np.ndarray, dur: float, climax: bool) -> list[float]:
        """Monotone 0..1 camera progress at PROGRESS_HZ: faster in loud music, slower in quiet."""
        n = max(2, int(math.ceil(dur * PROGRESS_HZ)) + 1)
        x = np.linspace(0, len(seg_intensity) - 1, n)
        inten = np.interp(x, np.arange(len(seg_intensity)), seg_intensity.astype(float))
        speed = 0.45 + 1.1 * inten ** 1.2
        if climax:
            speed *= 1.25
        speed = ndimage.gaussian_filter1d(speed, 6.0, mode="nearest")
        cum = np.concatenate([[0.0], np.cumsum(speed[:-1])])
        return [round(float(v), 6) for v in cum / cum[-1]]

    @staticmethod
    def _controls(analysis: AudioAnalysis, fps: int, duration: float) -> dict:
        """Audio-derived controls at video fps. Deliberately smoothed: music steers the
        picture, it must not strobe it."""
        n = int(math.ceil(duration * fps)) + 1
        t = np.arange(n) / fps
        src_t = np.arange(len(analysis.series["intensity"])) / FRAME_RATE

        def grab(name, sigma_s):
            v = np.interp(t, src_t, analysis.series[name].astype(float))
            return ndimage.gaussian_filter1d(v, sigma_s * fps, mode="nearest")

        pulse = grab("bass_energy", 0.12)
        pulse = np.clip((pulse - np.percentile(pulse, 20)) / (np.ptp(pulse) + 1e-9), 0, 1)
        # chroma-driven slow colour movement
        chroma = analysis.series["chroma"].astype(float)
        ang = np.arctan2((chroma * np.sin(np.arange(12) * math.pi / 6)).sum(1),
                         (chroma * np.cos(np.arange(12) * math.pi / 6)).sum(1))
        hue = np.interp(t, src_t, np.unwrap(ang) / (2 * math.pi))
        hue = ndimage.gaussian_filter1d(hue, 3.0 * fps, mode="nearest")
        return {
            "intensity": grab("intensity", 0.5).astype(np.float32),
            "pulse": pulse.astype(np.float32),
            "detail": grab("high", 0.4).astype(np.float32),
            "hue": (hue - hue.mean()).astype(np.float32),
        }
