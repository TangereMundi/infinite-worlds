"""QualityController: automated checks on a finished video.

Implemented checks (all measured, none assumed):
  duration vs audio, resolution, frame count, black frames, brightness extremes,
  flicker (second difference of mean luma), abrupt frame-to-frame jumps, frozen
  segments, near-duplicate scenes inside the video, bit-exact audio integrity.

Perceptual fingerprints are 64-bit difference hashes (dHash) of 1-fps frames.
They catch near-identical imagery; they are NOT learned embeddings and cannot judge
"same style". Embedding-based comparison is future work and is not claimed here.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio_analyzer import find_tool


def _probe(path: Path) -> dict:
    out = subprocess.run([find_tool("ffprobe"), "-v", "error", "-print_format", "json",
                          "-show_format", "-show_streams", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def audio_hash(path: Path, copy: bool = False) -> str:
    """SHA-256 of the first audio stream: decoded samples, or (copy=True) the compressed packets."""
    cmd = [find_tool("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0"]
    if copy:
        cmd += ["-c", "copy"]
    out = subprocess.run(cmd + ["-f", "hash", "-hash", "sha256", "-"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return out.split("=", 1)[-1]


def audio_integrity(source: Path, output: Path) -> dict:
    """The soundtrack is intact if the compressed packets are identical (stream copy of a lossy
    source) or the decoded samples are identical (lossless source). Both are reported: for
    MP3 in MP4 the packets match exactly while decoded output can differ at the very start,
    because the container signals encoder delay differently."""
    packets = audio_hash(source, True) == audio_hash(output, True)
    decoded = audio_hash(source) == audio_hash(output)
    return {"packets_match": packets, "decoded_match": decoded, "ok": packets or decoded}


def _gray_frames(path: Path, vf: str, w: int, h: int) -> np.ndarray:
    raw = subprocess.run([find_tool("ffmpeg"), "-v", "error", "-i", str(path), "-an", "-vf",
                          f"{vf},scale={w}:{h}:flags=area,format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, h, w)


def dhash(img: np.ndarray) -> int:
    """64-bit dHash from a (h>=8, w>=9) grayscale image."""
    h, w = img.shape
    ys = np.linspace(0, h - 1, 8).astype(int)
    xs = np.linspace(0, w - 1, 9).astype(int)
    small = img[np.ix_(ys, xs)].astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class QCResult:
    passed: bool
    issues: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    fingerprints: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "issues": self.issues, "warnings": self.warnings,
                "metrics": self.metrics, "fingerprints": [format(f, "016x") for f in self.fingerprints]}


def compare_fingerprints(a: list[int], b: list[int], max_dist: int = 6) -> float:
    """Share of frames in `a` that have a near-duplicate anywhere in `b`."""
    if not a or not b:
        return 0.0
    hits = sum(1 for x in a if any(hamming(x, y) <= max_dist for y in b))
    return hits / len(a)


class QualityController:
    def __init__(self, fade_in: float = 1.5, fade_out: float = 4.0):
        self.fade_in, self.fade_out = fade_in, fade_out

    def check(self, video: Path, source_audio: Path, expected_size: tuple[int, int],
              expected_fps: int, expected_duration: float) -> QCResult:
        res = QCResult(passed=True)
        issues, warns, m = res.issues, res.warnings, res.metrics

        info = _probe(video)
        vs = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
        as_ = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
        if vs is None or as_ is None:
            issues.append("missing video or audio stream")
            res.passed = False
            return res
        m["width"], m["height"] = int(vs["width"]), int(vs["height"])
        if (m["width"], m["height"]) != tuple(expected_size):
            issues.append(f"resolution {m['width']}x{m['height']} != expected {expected_size}")
        m["aspect"] = round(m["width"] / m["height"], 4)
        num, den = vs["avg_frame_rate"].split("/")
        m["fps"] = round(float(num) / float(den), 3)
        if abs(m["fps"] - expected_fps) > 0.05:
            issues.append(f"frame rate {m['fps']} != {expected_fps}")
        vdur = float(vs.get("duration") or info["format"]["duration"])
        adur = float(as_.get("duration") or info["format"]["duration"])
        m["video_duration"], m["audio_duration"] = round(vdur, 3), round(adur, 3)
        tol = 1.5 / expected_fps + 0.05
        if abs(vdur - expected_duration) > tol:
            issues.append(f"video duration {vdur:.3f}s differs from track {expected_duration:.3f}s")
        if abs(adur - expected_duration) > 0.1:
            issues.append(f"audio duration {adur:.3f}s differs from track {expected_duration:.3f}s")

        # audio integrity (decoded-sample hash vs the source file)
        integ = audio_integrity(source_audio, video)
        m["audio_hash_match"] = integ["ok"]
        m["audio_packets_match"], m["audio_decoded_match"] = integ["packets_match"], integ["decoded_match"]
        if not integ["ok"]:
            warns.append("audio differs from source (expected only if the lossy compat policy is on)")

        # frame statistics at full frame rate on tiny frames
        small = _gray_frames(video, "null", 32, 18).astype(np.float32) / 255.0
        luma = small.mean(axis=(1, 2))
        n = len(luma)
        m["frames_decoded"] = n
        t = np.arange(n) / expected_fps
        body = (t > self.fade_in + 0.3) & (t < expected_duration - self.fade_out - 0.3)
        m["luma_mean"] = round(float(luma[body].mean()), 3) if body.any() else None
        m["luma_p99"] = round(float(np.percentile(luma, 99)), 3)
        if body.any():
            black = float((luma[body] < 0.02).mean())
            m["black_frame_ratio"] = round(black, 4)
            if black > 0.01:
                issues.append(f"{black:.1%} black frames outside fades")
            if luma[body].mean() > 0.8:
                issues.append("excessive brightness")
            if luma[body].mean() < 0.04:
                issues.append("video is almost entirely dark")

        # flicker and abrupt jumps
        diff = np.abs(np.diff(small, axis=0)).mean(axis=(1, 2))
        m["frame_diff_mean"] = round(float(diff.mean()), 5)
        m["frame_diff_max"] = round(float(diff.max()), 5)
        if n > 4:
            dd = np.abs(np.diff(luma, n=2))
            b2 = body[2:n]
            if b2.any():
                m["flicker_p99"] = round(float(np.percentile(dd[b2], 99)), 5)
                m["flicker_max"] = round(float(dd[b2].max()), 5)
                if dd[b2].max() > 0.06:
                    issues.append(f"luminance flicker spike ({dd[b2].max():.3f})")
        if diff.max() > 0.22:
            warns.append(f"large frame-to-frame jump ({diff.max():.3f}) - possible hard cut")

        # perceptual fingerprints at 1 fps, repetition + freezes
        g = _gray_frames(video, "fps=1", 72, 40)
        fps_list = [dhash(f) for f in g]
        res.fingerprints = fps_list
        # featureless frames: almost no spatial detail (a flat colour field is not a visual experience)
        gf = g.astype(np.float32) / 255.0
        detail = np.hypot(np.diff(gf, axis=2)[:, :-1, :], np.diff(gf, axis=1)[:, :, :-1]).mean(axis=(1, 2))
        sec = np.arange(len(fps_list))
        bodysec = (sec > self.fade_in + 1) & (sec < expected_duration - self.fade_out - 1)
        dup = 0
        pairs = 0
        frozen_run = max_run = 0
        for i in range(len(fps_list)):
            if not bodysec[i]:
                continue
            if i > 0 and bodysec[i - 1] and hamming(fps_list[i], fps_list[i - 1]) == 0:
                frozen_run += 1
                max_run = max(max_run, frozen_run)
            else:
                frozen_run = 0
            for j in range(0, i - 8):
                if not bodysec[j]:
                    continue
                pairs += 1
                if hamming(fps_list[i], fps_list[j]) <= 4:
                    dup += 1
                    break
        body_count = int(bodysec.sum())
        m["detail_p05"] = round(float(np.percentile(detail[bodysec], 5)), 5) if body_count else None
        flat = float((detail[bodysec] < 0.004).mean()) if body_count else 0.0
        m["flat_frame_ratio"] = round(flat, 3)
        if flat > 0.05:
            issues.append(f"{flat:.0%} of frames are featureless")
        m["near_duplicate_ratio"] = round(dup / max(body_count, 1), 3)
        m["longest_frozen_seconds"] = max_run + 1 if max_run else 0
        if m["longest_frozen_seconds"] >= 5:
            issues.append(f"image frozen for ~{m['longest_frozen_seconds']}s")
        if m["near_duplicate_ratio"] > 0.35:
            issues.append(f"{m['near_duplicate_ratio']:.0%} of the video repeats earlier imagery")

        res.passed = not issues
        return res
