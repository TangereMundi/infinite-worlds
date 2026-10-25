"""AudioAnalyzer: measure a track without ever modifying the original file.

Only numpy, scipy and ffmpeg/ffprobe are required. The original audio is decoded
to a temporary float stream purely for analysis; the master file is never
re-encoded by this module.

Every uncertain measurement (tempo, sections) carries a confidence value so that
downstream modules never depend on a single possibly-wrong estimate.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import ANALYSIS_VERSION

ANALYSIS_SR = 22050
N_FFT = 2048
HOP = 512
FRAME_RATE = ANALYSIS_SR / HOP  # ~43.07 analysis frames per second


def find_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(
            f"'{name}' was not found on PATH. Install FFmpeg and add it to PATH."
        )
    return path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def probe(path: Path) -> dict:
    """Source metadata (container/stream properties). Not measured audio quality."""
    try:
        out = subprocess.run(
            [find_tool("ffprobe"), "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        raise ValueError(f"'{path.name}' is not a readable audio file") from None
    info = json.loads(out)
    stream = next((s for s in info["streams"] if s.get("codec_type") == "audio"), None)
    if stream is None:
        raise ValueError(f"No audio stream in {path}")
    fmt = info.get("format", {})
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    bits = stream.get("bits_per_raw_sample") or stream.get("bits_per_sample")
    return {
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate", 0)),
        "channels": int(stream.get("channels", 0)),
        "bit_depth": int(bits) if bits and str(bits).isdigit() and int(bits) > 0 else None,
        "bitrate": int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
        "file_size": int(fmt["size"]) if fmt.get("size") else path.stat().st_size,
        "duration": duration,
        "title": tags.get("title"),
        "artist": tags.get("artist"),
        "album": tags.get("album"),
    }


def decode_for_analysis(path: Path) -> np.ndarray:
    """Decode to float32 stereo at ANALYSIS_SR, shape (n, 2). Analysis copy only."""
    proc = subprocess.run(
        [find_tool("ffmpeg"), "-v", "error", "-i", str(path), "-vn",
         "-ac", "2", "-ar", str(ANALYSIS_SR), "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    if data.size < 2 * N_FFT:
        raise ValueError("Audio too short to analyse")
    return data.reshape(-1, 2).copy()


def measure_loudness(path: Path) -> dict:
    """EBU R128 integrated loudness / range via ffmpeg's ebur128 filter."""
    res = subprocess.run(
        [find_tool("ffmpeg"), "-hide_banner", "-nostats", "-i", str(path), "-vn",
         "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    text = res.stderr
    idx = text.rfind("Summary:")
    tail = text[idx:] if idx >= 0 else ""

    def grab(label: str):
        m = re.search(rf"{label}:\s*(-?\d+(?:\.\d+)?)", tail)
        return float(m.group(1)) if m else None

    return {
        "integrated_lufs": grab("I"),
        "loudness_range_lu": grab("LRA"),
        "true_peak_dbfs": grab("Peak"),
    }


def _stft_mag(x: np.ndarray) -> np.ndarray:
    """Magnitude STFT, shape (frames, bins)."""
    pad = N_FFT // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - N_FFT) // HOP
    shape = (n_frames, N_FFT)
    strides = (xp.strides[0] * HOP, xp.strides[0])
    frames = np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)
    win = np.hanning(N_FFT).astype(np.float32)
    out = np.empty((n_frames, N_FFT // 2 + 1), dtype=np.float32)
    # chunked to bound memory on long tracks
    step = 2048
    for i in range(0, n_frames, step):
        out[i:i + step] = np.abs(np.fft.rfft(frames[i:i + step] * win, axis=1))
    return out


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter1d(x.astype(np.float64), sigma, mode="nearest")


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def estimate_tempo(onset: np.ndarray) -> dict:
    """Autocorrelation tempo with candidates + confidence (never a single trusted value)."""
    o = onset - onset.mean()
    if np.allclose(o, 0):
        return {"bpm": None, "confidence": 0.0, "candidates": []}
    n = len(o)
    ac = np.correlate(o, o, mode="full")[n - 1:]
    ac /= (ac[0] + 1e-12)
    lag_min = int(FRAME_RATE * 60 / 200)
    lag_max = min(int(FRAME_RATE * 60 / 60), n - 1)
    if lag_max <= lag_min + 2:
        return {"bpm": None, "confidence": 0.0, "candidates": []}
    seg = ac[lag_min:lag_max + 1].copy()
    # mild prior favouring ~120 BPM so octave errors are less likely
    lags = np.arange(lag_min, lag_max + 1)
    bpms = 60.0 * FRAME_RATE / lags
    prior = np.exp(-0.5 * (np.log2(bpms / 120.0) / 0.9) ** 2)
    scored = seg * prior
    peaks = [i for i in range(1, len(scored) - 1)
             if scored[i] > scored[i - 1] and scored[i] >= scored[i + 1]]
    peaks.sort(key=lambda i: scored[i], reverse=True)
    cands = []
    for i in peaks[:4]:
        cands.append({"bpm": round(float(bpms[i]), 2), "strength": round(float(seg[i]), 4)})
    if not cands:
        return {"bpm": None, "confidence": 0.0, "candidates": []}
    top = cands[0]["strength"]
    second = cands[1]["strength"] if len(cands) > 1 else 0.0
    conf = float(np.clip(top * 1.5, 0, 1) * (1.0 - 0.5 * (second / (top + 1e-9))))
    return {"bpm": cands[0]["bpm"], "confidence": round(conf, 3), "candidates": cands}


def track_beats(onset: np.ndarray, bpm: float) -> np.ndarray:
    """Simple dynamic-programming beat tracker; returns beat times in seconds."""
    period = FRAME_RATE * 60.0 / bpm
    n = len(onset)
    o = onset / (onset.std() + 1e-9)
    score = o.copy()
    back = np.full(n, -1, dtype=np.int64)
    lo, hi = int(round(period / 2)), int(round(period * 2))
    for t in range(n):
        a, b = t - hi, t - lo
        if b < 0:
            continue
        a = max(a, 0)
        prev = np.arange(a, b + 1)
        pen = -100.0 * (np.log((t - prev) / period) ** 2)
        cand = score[prev] + pen
        j = int(np.argmax(cand))
        if cand[j] > 0:
            score[t] = o[t] + cand[j]
            back[t] = prev[j]
    tail_start = max(0, n - int(period))
    t = tail_start + int(np.argmax(score[tail_start:]))
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    beats = np.array(beats[::-1], dtype=np.float64)
    return beats / FRAME_RATE


def detect_sections(feat: np.ndarray, duration: float, min_gap_s: float = 10.0) -> list[dict]:
    """Novelty-based section boundaries from a self-similarity matrix.

    feat: (frames, dims) feature matrix at analysis frame rate.
    """
    ds = max(1, int(FRAME_RATE / 2))  # ~2 Hz
    n = (len(feat) // ds) * ds
    if n < ds * 8:
        return [{"start": 0.0, "end": duration, "novelty": 0.0}]
    f = feat[:n].reshape(-1, ds, feat.shape[1]).mean(axis=1)
    f = (f - f.mean(0)) / (f.std(0) + 1e-9)
    f /= (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
    ssm = f @ f.T
    k = 8
    kern = np.zeros((2 * k, 2 * k))
    kern[:k, :k] = 1
    kern[k:, k:] = 1
    kern[:k, k:] = -1
    kern[k:, :k] = -1
    g = np.outer(np.hanning(2 * k), np.hanning(2 * k))
    kern *= g
    m = len(ssm)
    padded = np.pad(ssm, k, mode="edge")
    nov = np.array([np.sum(padded[i:i + 2 * k, i:i + 2 * k] * kern) for i in range(m)])
    nov = np.clip(nov, 0, None)
    nov = _smooth(nov, 1.0)
    thresh = nov.mean() + 0.6 * nov.std()
    min_gap = int(min_gap_s * 2)
    peaks = []
    for i in np.argsort(nov)[::-1]:
        if nov[i] < thresh:
            break
        if all(abs(i - p) >= min_gap for p in peaks):
            peaks.append(int(i))
    peaks.sort()
    bounds = [0.0] + [p / 2.0 for p in peaks if 4.0 < p / 2.0 < duration - 4.0] + [duration]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        mid = min(int((a + b) / 2 * 2), m - 1)
        out.append({"start": round(a, 3), "end": round(b, 3),
                    "novelty": round(float(nov[min(int(a * 2), m - 1)]), 4)})
    return out


@dataclass
class AudioAnalysis:
    analysis_version: str
    content_hash: str
    metadata: dict          # source metadata (container facts)
    measured: dict          # measured audio properties
    feature_digest_input: dict  # quantised scalars used for the seed digest
    sections: list
    tempo: dict
    beats: list
    series: dict = field(default_factory=dict, repr=False)  # name -> np.ndarray @ FRAME_RATE

    def to_json_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "series"}
        d["series_frame_rate"] = FRAME_RATE
        d["series_names"] = sorted(self.series)
        return d


class AudioAnalyzer:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_paths(self, content_hash: str):
        base = self.cache_dir / f"{content_hash}.v{ANALYSIS_VERSION}"
        return base.with_suffix(".json"), base.with_suffix(".npz")

    def analyse(self, path: Path) -> AudioAnalysis:
        path = Path(path)
        content_hash = sha256_file(path)
        # Cached records are reused so seeds stay stable even if libraries change later.
        if self.cache_dir:
            jp, zp = self._cache_paths(content_hash)
            if jp.exists() and zp.exists():
                return self._load(jp, zp)
        result = self._compute(path, content_hash)
        if self.cache_dir:
            self._save(result)
        return result

    # ------------------------------------------------------------------
    def _compute(self, path: Path, content_hash: str) -> AudioAnalysis:
        meta = probe(path)
        stereo = decode_for_analysis(path)
        mono = stereo.mean(axis=1)
        duration = len(stereo) / ANALYSIS_SR
        meta["analysed_duration"] = round(duration, 3)

        loud = measure_loudness(path)

        mag = _stft_mag(mono)
        freqs = np.fft.rfftfreq(N_FFT, 1.0 / ANALYSIS_SR)
        power = mag ** 2
        total = power.sum(axis=1) + 1e-12

        low = power[:, freqs < 200].sum(axis=1)
        mid = power[:, (freqs >= 200) & (freqs < 2000)].sum(axis=1)
        high = power[:, freqs >= 2000].sum(axis=1)

        centroid = (mag * freqs).sum(axis=1) / (mag.sum(axis=1) + 1e-12)
        bandwidth = np.sqrt((mag * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1)
                            / (mag.sum(axis=1) + 1e-12))
        cum = np.cumsum(power, axis=1)
        rolloff = freqs[np.argmax(cum >= 0.85 * cum[:, -1:], axis=1)]

        # spectral contrast (6 octave-ish bands, peak vs valley in dB)
        edges = [0, 200, 400, 800, 1600, 3200, 11025]
        contrast = []
        for a, b in zip(edges[:-1], edges[1:]):
            band = np.sort(mag[:, (freqs >= a) & (freqs < b)], axis=1)
            q = max(1, band.shape[1] // 5)
            contrast.append(20 * np.log10((band[:, -q:].mean(1) + 1e-9) / (band[:, :q].mean(1) + 1e-9)))
        contrast = np.stack(contrast, axis=1)

        # onset strength: positive log-spectral flux
        logm = np.log1p(mag * 10)
        flux = np.maximum(0, np.diff(logm, axis=0, prepend=logm[:1])).mean(axis=1)
        onset = _smooth(flux, 1.0)

        # harmonic / percussive split on a reduced spectrogram
        red_edges = np.unique(np.geomspace(1, mag.shape[1] - 1, 129).astype(int))
        red = np.stack([mag[:, a:b].mean(axis=1) for a, b in zip(red_edges[:-1], red_edges[1:])], axis=1)
        harm = ndimage.median_filter(red, size=(17, 1), mode="nearest")
        perc = ndimage.median_filter(red, size=(1, 9), mode="nearest")
        h_e, p_e = float((harm ** 2).sum()), float((perc ** 2).sum())
        harmonic_ratio = h_e / (h_e + p_e + 1e-12)
        perc_series = (perc ** 2).sum(axis=1)

        # chroma
        chroma = np.zeros((mag.shape[0], 12), dtype=np.float64)
        valid = (freqs > 55) & (freqs < 2000)
        pcs = (np.round(12 * np.log2(freqs[valid] / 440.0)) + 9).astype(int) % 12
        for pc in range(12):
            chroma[:, pc] = power[:, valid][:, pcs == pc].sum(axis=1)
        chroma /= (chroma.sum(axis=1, keepdims=True) + 1e-12)
        mean_chroma = chroma.mean(axis=0)

        # RMS / dynamics
        rms = np.sqrt(total / N_FFT)  # proportional; used for relative dynamics
        rms_db = 20 * np.log10(_smooth(rms, 4) + 1e-9)
        dyn_range = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10))
        peak = float(np.max(np.abs(stereo)))
        peak_db = 20 * np.log10(peak + 1e-12)

        # stereo width
        mid_s = stereo.mean(axis=1)
        side_s = (stereo[:, 0] - stereo[:, 1]) / 2
        width = float(np.sqrt((side_s ** 2).mean()) / (np.sqrt((mid_s ** 2).mean()) + 1e-12))

        # tempo + beats
        tempo = estimate_tempo(onset)
        beats = []
        if tempo["bpm"] and tempo["confidence"] > 0.15:
            beats = [round(float(b), 3) for b in track_beats(onset, tempo["bpm"])]

        # intensity curve (0..1): loudness + onset density + brightness
        intensity = _norm01(_smooth(
            0.5 * _norm01(rms_db) + 0.3 * _norm01(_smooth(onset, 6)) + 0.2 * _norm01(centroid), 8))

        # Absolute energy level of the whole track (0..1). The intensity curve above is relative
        # (it shows *where* the music builds); this scalar says *how energetic the track is overall*,
        # so a gentle ambient piece and a driving one do not end up with the same visual energy.
        onset_density = float((onset > onset.mean() + onset.std()).sum() / duration)
        lufs = loud["integrated_lufs"] if loud["integrated_lufs"] is not None else -20.0
        energy_level = float(np.clip(
            0.35 * np.clip(onset_density / 2.0, 0, 1) + 0.30 * (1.0 - harmonic_ratio)
            + 0.20 * np.clip(float(centroid.mean()) / 3500.0, 0, 1)
            + 0.15 * np.clip((lufs + 30.0) / 18.0, 0, 1), 0, 1))
        intensity = np.clip(intensity * (0.45 + 0.55 * energy_level), 0, 1)

        # sections
        lb = np.stack([_norm01(_smooth(np.log10(low + 1e-9), 3)),
                       _norm01(_smooth(np.log10(mid + 1e-9), 3)),
                       _norm01(_smooth(np.log10(high + 1e-9), 3)),
                       _norm01(_smooth(centroid, 3)), intensity], axis=1)
        sect_feat = np.concatenate([lb, _smooth_cols(chroma, 4) * 3], axis=1)
        sections = detect_sections(sect_feat, duration)

        # repetition: how self-similar are distant parts of the track
        repetition = _repetition_score(sect_feat)

        def n01(x):  # per-frame band energies as relative shares
            return x

        shares = np.stack([low, mid, high], axis=1)
        shares = shares / (shares.sum(axis=1, keepdims=True) + 1e-12)

        series = {
            "intensity": intensity.astype(np.float32),
            "low": _norm01(_smooth(shares[:, 0], 2)).astype(np.float32),
            "mid": _norm01(_smooth(shares[:, 1], 2)).astype(np.float32),
            "high": _norm01(_smooth(shares[:, 2], 2)).astype(np.float32),
            "bass_energy": _norm01(_smooth(np.log10(low + 1e-9), 2)).astype(np.float32),
            "onset": _norm01(onset).astype(np.float32),
            "percussive": _norm01(_smooth(np.log10(perc_series + 1e-9), 2)).astype(np.float32),
            "centroid": _norm01(_smooth(centroid, 2)).astype(np.float32),
            "chroma": chroma.astype(np.float32),
        }

        measured = {
            "integrated_lufs": loud["integrated_lufs"],
            "loudness_range_lu": loud["loudness_range_lu"],
            "true_peak_dbfs": loud["true_peak_dbfs"],
            "sample_peak_dbfs": round(peak_db, 2),
            "dynamic_range_db": round(dyn_range, 2),
            "tempo_bpm": tempo["bpm"],
            "tempo_confidence": tempo["confidence"],
            "onset_density_per_s": round(onset_density, 3),
            "energy_level": round(energy_level, 3),
            "centroid_hz_mean": round(float(centroid.mean()), 1),
            "bandwidth_hz_mean": round(float(bandwidth.mean()), 1),
            "rolloff_hz_mean": round(float(rolloff.mean()), 1),
            "contrast_db_mean": [round(float(v), 2) for v in contrast.mean(axis=0)],
            "low_share": round(float(shares[:, 0].mean()), 4),
            "mid_share": round(float(shares[:, 1].mean()), 4),
            "high_share": round(float(shares[:, 2].mean()), 4),
            "harmonic_ratio": round(harmonic_ratio, 4),
            "chroma_mean": [round(float(v), 4) for v in mean_chroma],
            "dominant_pitch_class": int(np.argmax(mean_chroma)),
            "stereo_width": round(width, 4),
            "repetition": round(repetition, 4),
            "section_count": len(sections),
        }

        # Coarsely quantised scalars for the seed digest: small float jitter between
        # library versions must not change the audio identity.
        digest_input = {
            "duration_s": round(duration, 1),
            "lufs": round(loud["integrated_lufs"] or 0.0),
            "dynamic_range_db": round(dyn_range),
            "low_share": round(float(shares[:, 0].mean()), 2),
            "mid_share": round(float(shares[:, 1].mean()), 2),
            "high_share": round(float(shares[:, 2].mean()), 2),
            "harmonic_ratio": round(harmonic_ratio, 1),
            "dominant_pitch_class": int(np.argmax(mean_chroma)),
            "section_count": len(sections),
            "stereo_width": round(width, 1),
        }

        return AudioAnalysis(
            analysis_version=ANALYSIS_VERSION, content_hash=content_hash,
            metadata=meta, measured=measured, feature_digest_input=digest_input,
            sections=sections, tempo=tempo, beats=beats, series=series,
        )

    # ------------------------------------------------------------------
    def _save(self, a: AudioAnalysis) -> None:
        jp, zp = self._cache_paths(a.content_hash)
        np.savez_compressed(zp, **a.series)
        tmp = jp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(a.to_json_dict(), indent=2))
        tmp.replace(jp)

    @staticmethod
    def _load(jp: Path, zp: Path) -> AudioAnalysis:
        d = json.loads(jp.read_text())
        with np.load(zp) as z:
            series = {k: z[k] for k in z.files}
        return AudioAnalysis(
            analysis_version=d["analysis_version"], content_hash=d["content_hash"],
            metadata=d["metadata"], measured=d["measured"],
            feature_digest_input=d["feature_digest_input"], sections=d["sections"],
            tempo=d["tempo"], beats=d["beats"], series=series,
        )


def _smooth_cols(x: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter1d(x, sigma, axis=0, mode="nearest")


def _repetition_score(feat: np.ndarray) -> float:
    ds = max(1, int(FRAME_RATE))
    n = (len(feat) // ds) * ds
    if n < ds * 20:
        return 0.0
    f = feat[:n].reshape(-1, ds, feat.shape[1]).mean(axis=1)
    f = (f - f.mean(0)) / (f.std(0) + 1e-9)
    f /= (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
    ssm = f @ f.T
    m = len(ssm)
    mask = np.abs(np.subtract.outer(np.arange(m), np.arange(m))) > 8
    best = np.where(mask, ssm, -1).max(axis=1)
    return float(np.clip(np.mean(best), 0, 1))


def sample_series(series: np.ndarray, t: float) -> float:
    """Linear interpolation of an analysis series at time t (seconds)."""
    x = t * FRAME_RATE
    i = int(np.clip(np.floor(x), 0, len(series) - 1))
    j = min(i + 1, len(series) - 1)
    w = x - i
    return float(series[i] * (1 - w) + series[j] * w)
