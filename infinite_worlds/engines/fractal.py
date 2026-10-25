"""FractalEngine (CPU / numpy reference backend).

Modes
  mandelbrot : zoom toward an autonomously *discovered* region of the boundary
  julia      : animated Julia set (c orbits a seeded point) with slow zoom

Deep-zoom policy
  This backend uses float64 and refuses to go below MIN_WIDTH (pixel step stays
  >> double-precision epsilon at 4K). It never "zooms forever": when a scene's
  depth budget is spent the VisualDirector hands over to a new discovered region.
  Perturbation-theory deep zoom is the planned GPU-backend technique (see docs).

Not implemented here: Mandelbulb / quaternion / distance-estimated 3D fractals.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.interpolate import PchipInterpolator

from .base import VisualEngine

MIN_WIDTH = 5e-10
BAILOUT2 = 256.0 * 256.0

CURATED_JULIA = [
    (-0.8, 0.156), (-0.4, 0.6), (0.285, 0.01), (-0.70176, -0.3842),
    (-0.835, -0.2321), (0.37, 0.1), (-0.74543, 0.11301), (-0.1, 0.651),
    (-0.162, 1.04), (0.355, 0.355), (-0.54, 0.54), (-0.7269, 0.1889),
]


# ----------------------------------------------------------------------------
# escape-time kernel
# ----------------------------------------------------------------------------
def escape_mu(z0: np.ndarray, c: np.ndarray, maxiter: int, mandelbrot: bool) -> np.ndarray:
    """Smooth iteration value per pixel (flattened). Interior pixels get -1."""
    n = z0.size
    mu = np.full(n, -1.0)
    idx = np.arange(n)
    zc = z0.copy()
    cc = c.copy()
    if mandelbrot:
        x, y = cc.real, cc.imag
        q = (x - 0.25) ** 2 + y ** 2
        inside = (q * (q + (x - 0.25)) < 0.25 * y ** 2) | ((x + 1) ** 2 + y ** 2 < 0.0625)
        keep = ~inside
        idx, zc, cc = idx[keep], zc[keep], cc[keep]
    with np.errstate(over="ignore", invalid="ignore"):
        for it in range(maxiter):
            if idx.size == 0:
                break
            zc = zc * zc + cc
            m2 = zc.real * zc.real + zc.imag * zc.imag
            esc = m2 > BAILOUT2
            if esc.any():
                e = np.nonzero(esc)[0]
                lz = np.log(m2[e]) * 0.5
                mu[idx[e]] = it + 1.0 - np.log2(lz)
                keep = ~esc
                idx, zc, cc = idx[keep], zc[keep], cc[keep]
    return mu


def _maxiter_for_width(w: float, quality: float = 1.0) -> int:
    octaves = max(0.0, math.log2(3.0 / max(w, MIN_WIDTH)))
    return int(min(3000, (160 + 55 * octaves) * quality))


def _structure_score(cx: float, cy: float, width: float, mandelbrot: bool = True,
                     c: complex = 0j) -> float:
    """How visually structured is a window? Used by the discovery search.

    Rewards windows that contain plenty of boundary detail (varied iteration counts)
    and a moderate interior share; penalises flat exterior or solid interior.
    """
    g = 28
    xs = (np.arange(g) / (g - 1) - 0.5) * width + cx
    ys = (np.arange(g) / (g - 1) - 0.5) * width + cy
    pts = (xs[None, :] + 1j * ys[:, None]).ravel()
    mi = _maxiter_for_width(width) // 2 + 100
    if mandelbrot:
        mu = escape_mu(np.zeros_like(pts), pts, mi, True)
    else:
        mu = escape_mu(pts, np.full_like(pts, c), mi, False)
    inside = mu < 0
    inter = float(inside.mean())
    if inter > 0.65 or (~inside).sum() < 30:
        return 0.0
    h = np.log1p(np.where(inside, 0.0, mu)).reshape(g, g)
    h = np.where(inside.reshape(g, g), h.max() + 1.0, h)
    gy, gx = np.gradient(h)
    mag = np.hypot(gx, gy)
    edge_density = float((mag > 0.12).mean())          # share of pixels on visible detail
    interior_term = 1.0 - abs(inter - 0.15) / 0.65
    return edge_density * max(0.0, interior_term)


def _descend(x: float, y: float, rng, w_final: float, mandelbrot: bool, c: complex = 0j,
             w_start: float = 0.4, factor: float = 6.0) -> tuple[float, float, float]:
    """Coarse-to-fine tracking: at every scale nudge the centre toward the most
    structured nearby window, so structure persists all the way down to w_final.
    Returns (x, y, worst_score_along_path)."""
    w = w_start
    worst = 1e9
    while True:
        best, bx, by = _structure_score(x, y, w, mandelbrot, c), x, y
        for _ in range(8):
            nx, ny = x + rng.normal(0, 0.28 * w), y + rng.normal(0, 0.28 * w)
            sc = _structure_score(nx, ny, w, mandelbrot, c)
            if sc > best:
                best, bx, by = sc, nx, ny
        x, y = bx, by
        worst = min(worst, best)
        if w <= w_final * 1.01:
            break
        w = max(w / factor, w_final)
    return x, y, worst


# Well-known boundary regions (standard published coordinates): extra candidates for the
# discovery search, so a poor random draw can never leave a scene without structure.
KNOWN_REGIONS = [
    (-0.743643887037151, 0.131825904205330),   # seahorse valley
    (-0.7453, 0.1127),
    (0.2549870375144766, -0.0005679790528465), # elephant valley
    (-0.10109636384562, 0.95628651080914),
    (-1.25066, 0.02012),
    (-0.77568377, 0.13646737),
    (-0.16, 1.0405),
    (-1.7685, -0.0009),
]
MIN_DISCOVERY_SCORE = 0.08


def discover_target(rng, depth_width: float, n_candidates: int = 14,
                    avoid: list[tuple[float, float]] | None = None) -> tuple[float, float, float]:
    """Autonomously find an interesting Mandelbrot boundary location.

    Samples near-boundary points, ranks them cheaply, then refines the best few with
    a coarse-to-fine descent down to depth_width. Returns (re, im, score).
    """
    avoid = avoid or []
    cands = []
    tries = 0
    while len(cands) < n_candidates and tries < n_candidates * 80:
        tries += 1
        x, y = rng.uniform(-2.0, 0.65), rng.uniform(0.0, 1.25)
        if rng.random() < 0.5:
            y = -y
        mu = escape_mu(np.array([0j]), np.array([complex(x, y)]), 400, True)[0]
        if mu >= 22:
            cands.append((_structure_score(x, y, 0.1) + _structure_score(x, y, 0.01), x, y))
    cands.sort(reverse=True)
    best = (-0.743643887037151, 0.131825904205330, 0.0)
    best_score = -1.0
    starts = [(x, y) for _, x, y in cands[:3]]
    starts += [KNOWN_REGIONS[rng.integer(0, len(KNOWN_REGIONS) - 1)] for _ in range(2)]
    for x, y in starts:
        fx, fy, sc = _descend(x, y, rng, depth_width, True)
        if any(math.hypot(fx - ax, fy - ay) < 0.02 for ax, ay in avoid):
            sc *= 0.3
        if sc > best_score:
            best_score, best = sc, (fx, fy, sc)
    return best


def discover_julia_target(rng, c: complex, depth_width: float) -> tuple[float, float, float]:
    """Find a structured region of the Julia set for parameter c."""
    cands = []
    for _ in range(60):
        x, y = rng.uniform(-1.4, 1.4), rng.uniform(-1.0, 1.0)
        cands.append((_structure_score(x, y, 0.5, False, c), x, y))
    cands.sort(reverse=True)
    best, best_score = (0.0, 0.0, 0.0), -1.0
    for _, x, y in cands[:3]:
        fx, fy, sc = _descend(x, y, rng, depth_width, False, c, w_start=0.5)
        if sc > best_score:
            best_score, best = sc, (fx, fy, sc)
    return best


# ----------------------------------------------------------------------------
class FractalEngine(VisualEngine):
    name = "fractal"
    version = "0.2.0-cpu"

    # ------------------------------------------------------------------ scene
    def sample_scene(self, rng, ctx: dict) -> dict:
        duration = float(ctx["duration"])
        intensity = float(ctx.get("intensity", 0.5))
        avoid_mode = ctx.get("avoid_mode")
        avoid_targets = ctx.get("avoid_targets", [])
        direction = ctx.get("direction", "auto")
        style = ctx.get("style", "classic")
        zoom_dir = ctx.get("zoom_dir", "in")

        mode_w = {"mandelbrot": 1.0, "julia": 0.8}
        if ctx.get("quiet"):
            mode_w["julia"] = 1.3
        if ctx.get("climax") and avoid_mode:
            mode_w[avoid_mode] = 0.0
        if direction in ("mandelbrot", "julia"):
            mode_w = {direction: 1.0, **{k: 0.0 for k in mode_w if k != direction}}
        if style == "classic" and direction not in ("mandelbrot", "julia"):
            mode_w = {"mandelbrot": 1.0, "julia": 0.0}
        mode = rng.weighted_choice(list(mode_w), list(mode_w.values()))

        # octaves per second (mean). The classic look zooms slowly; depth is capped by float64.
        rate = (rng.uniform(0.2, 0.34) if style == "classic" else rng.uniform(0.22, 0.5)) * (0.8 + 0.5 * intensity)
        octaves = rate * duration
        scene: dict = {"engine": self.name, "mode": mode, "duration": duration,
                       "style": style, "zoom_dir": zoom_dir,
                       "progress_knots": ctx.get("progress_knots"),
                       "progress_rate_hz": ctx.get("progress_rate_hz", 10.0)}

        if mode == "mandelbrot":
            w0 = rng.uniform(1.6, 3.2)
            w1 = max(MIN_WIDTH * 4, w0 / (2.0 ** octaves))
            x, y, score = discover_target(rng, w1, avoid=avoid_targets)
            for _ in range(3):
                if score >= MIN_DISCOVERY_SCORE:
                    break
                x2, y2, s2 = discover_target(rng, w1, avoid=avoid_targets)
                if s2 > score:
                    x, y, score = x2, y2, s2
            if score < MIN_DISCOVERY_SCORE:       # still bare: make the scene shallower instead
                w1 = max(w1, 0.02)
            scene.update({
                "target": [x, y], "w0": w0, "w1": w1, "discovery_score": round(score, 4),
                "off": [rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06)],
                "wob_r": [rng.uniform(0.03, 0.11), rng.uniform(0.03, 0.11)],
                "wob_w": [rng.uniform(0.04, 0.16), rng.uniform(0.04, 0.16)],
                "wob_p": [rng.uniform(0, 6.283), rng.uniform(0, 6.283)],
            })
        else:
            c0 = rng.choice(CURATED_JULIA)
            c0 = (c0[0] + rng.normal(0, 0.012), c0[1] + rng.normal(0, 0.012))
            octs = min(octaves * 0.35, 5.0)
            w0 = rng.uniform(3.0, 3.6)
            w1 = w0 / (2.0 ** octs)
            jx, jy, jscore = discover_julia_target(rng, complex(*c0), w1)
            for _ in range(3):
                if jscore >= MIN_DISCOVERY_SCORE:
                    break
                a_, b_, s_ = discover_julia_target(rng, complex(*c0), w1)
                if s_ > jscore:
                    jx, jy, jscore = a_, b_, s_
            scene.update({
                "c0": [c0[0], c0[1]], "orbit_r": rng.uniform(0.003, 0.02),
                "orbit_w": rng.uniform(0.05, 0.22) * (0.6 + 0.8 * intensity),
                "orbit_p": rng.uniform(0, 6.283),
                "w0": w0, "w1": w1, "target": [jx, jy], "discovery_score": round(jscore, 4),
                "off": [rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)],
                "wob_r": [rng.uniform(0.02, 0.08), rng.uniform(0.02, 0.08)],
                "wob_w": [rng.uniform(0.04, 0.12), rng.uniform(0.04, 0.12)],
                "wob_p": [rng.uniform(0, 6.283), rng.uniform(0, 6.283)],
            })

        scene["rot0"] = rng.uniform(0, 6.283)
        scene["rot_w"] = rng.uniform(-0.05, 0.05) * (0.5 + intensity)
        h0 = rng.random()
        scene["palette"] = {
            "a": [rng.uniform(0.42, 0.56)] * 3,
            "b": [rng.uniform(0.26, 0.42) for _ in range(3)],
            "c": [rng.uniform(0.8, 1.25) for _ in range(3)],
            "d": [h0, (h0 + 0.33 + rng.normal(0, 0.06)) % 1.0, (h0 + 0.67 + rng.normal(0, 0.06)) % 1.0],
            "freq": rng.uniform(0.18, 0.5),
            "hue": h0,
            "interior": [rng.uniform(0.0, 0.05), rng.uniform(0.0, 0.05), rng.uniform(0.02, 0.1)],
            "period": rng.uniform(22.0, 55.0),        # (unused by classic; kept for old records)
            "cycles": rng.uniform(0.7, 1.3),          # classic style: colour cycles per e-fold of iterations
            "sat": rng.uniform(0.85, 1.0),
            "fog": [rng.uniform(0.02, 0.1), rng.uniform(0.02, 0.1), rng.uniform(0.04, 0.16)],
        }
        scene["light_angle"] = rng.uniform(0, 6.283)
        scene["light_strength"] = rng.uniform(0.35, 0.7)
        return scene

    def fingerprint(self, scene: dict) -> list[float]:
        t = scene["target"] if scene["mode"] == "mandelbrot" else scene["c0"]
        return [1.0 if scene["mode"] == "julia" else 0.0, t[0], t[1],
                scene["palette"]["hue"], math.log2(scene["w0"] / scene["w1"]) / 20.0,
                0.5 if scene.get("zoom_dir") == "out" else 0.0]

    def describe(self, scene: dict) -> str:
        return f"fractal/{scene['mode']}/zoom-{scene.get('zoom_dir', 'in')}"

    # ----------------------------------------------------------------- camera
    @staticmethod
    def _progress(scene: dict, t: float) -> float:
        knots = scene.get("progress_knots")
        dur = scene["duration"]
        if not knots:
            return min(max(t / dur, 0.0), 1.0)
        xs = np.arange(len(knots)) / scene.get("progress_rate_hz", 10.0)
        f = PchipInterpolator(xs, np.asarray(knots, dtype=np.float64), extrapolate=True)
        return float(np.clip(f(np.clip(t, 0, xs[-1])), 0.0, 1.0))

    def camera(self, scene: dict, t: float, pulse: float = 0.0) -> dict:
        p = self._progress(scene, t)
        if scene.get("zoom_dir") == "out":       # same path, travelled backwards: deep -> wide
            p = 1.0 - p
        lw = math.log(scene["w0"]) * (1 - p) + math.log(scene["w1"]) * p
        w = math.exp(lw) * (1.0 - 0.018 * pulse)
        off = scene["off"]
        wr, ww, wp = scene["wob_r"], scene["wob_w"], scene["wob_p"]
        ox = off[0] + wr[0] * math.cos(wp[0] + ww[0] * t)
        oy = off[1] + wr[1] * math.sin(wp[1] + ww[1] * t)
        tx, ty = scene["target"]
        cx, cy = tx + w * ox, ty + w * oy
        rot = scene["rot0"] + scene["rot_w"] * t
        return {"cx": cx, "cy": cy, "w": max(w, MIN_WIDTH), "rot": rot, "p": p}

    def _mu_field(self, scene, t, W, H, pulse=0.0, detail=0.0):
        """Smooth-iteration field (H, W) for scene-local time t, plus the camera used."""
        cam = self.camera(scene, t, pulse)
        w = cam["w"]
        aspect = W / H
        xs = (np.arange(W) + 0.5) / W - 0.5
        ys = (np.arange(H) + 0.5) / H - 0.5
        X, Y = np.meshgrid(xs * w, ys * (w / aspect))
        cr, sr = math.cos(cam["rot"]), math.sin(cam["rot"])
        px = cam["cx"] + X * cr - Y * sr
        py = cam["cy"] + X * sr + Y * cr
        pts = (px + 1j * py).ravel()
        maxiter = _maxiter_for_width(w, 1.0 + 0.25 * detail)
        if scene["mode"] == "mandelbrot":
            mu = escape_mu(np.zeros_like(pts), pts, maxiter, True)
        else:
            c0 = complex(*scene["c0"])
            ang = scene["orbit_p"] + scene["orbit_w"] * t
            c = c0 + scene["orbit_r"] * complex(math.cos(ang), math.sin(ang))
            mu = escape_mu(pts, np.full_like(pts, c), maxiter, False)
        return mu.reshape(H, W), cam

    def validate_scene(self, scene: dict, samples: int = 9) -> dict:
        """Probe the whole camera path at low resolution. A scene is usable only if no sampled
        view is mostly solid interior (black) or featureless. Cheap: 9 views of 64x36 pixels."""
        W, H = 64, 36
        worst_interior, worst_flat = 0.0, 0.0
        for k in range(samples):
            t = scene["duration"] * k / (samples - 1)
            mu, _ = self._mu_field(scene, t, W, H)
            inside = mu < 0
            worst_interior = max(worst_interior, float(inside.mean()))
            if (~inside).sum() < 40:
                worst_flat = 1.0
                continue
            h = np.log1p(mu[~inside])
            gy, gx = np.gradient(np.log1p(np.where(inside, 0.0, mu)))
            detail = float((np.hypot(gx, gy) > 0.05).mean())
            worst_flat = max(worst_flat, 1.0 if detail < 0.03 else 0.0)
        return {"max_interior": round(worst_interior, 3), "has_flat_view": bool(worst_flat),
                "ok": worst_interior < 0.55 and not worst_flat}

    # ----------------------------------------------------------------- render
    def render(self, scene, t, controls, width, height, supersample=1):
        ss = max(1, int(supersample))
        W, H = width * ss, height * ss
        pulse = float(controls.get("pulse", 0.0))
        intensity = float(controls.get("intensity", 0.5))
        detail = float(controls.get("detail", 0.0))
        hue_shift = float(controls.get("hue", 0.0))

        mu, cam = self._mu_field(scene, t, W, H, pulse, detail)
        w = cam["w"]
        aspect = W / H

        img = self._colour(scene, mu, t, intensity, detail, hue_shift, pulse, w, aspect)
        if ss > 1:
            img = img.reshape(height, ss, width, ss, 3).mean(axis=(1, 3))
        return img.astype(np.float32)

    def _colour_classic(self, scene, mu, t, intensity, hue_shift, pulse):
        """Saturated colour bands cycling with iteration count, black interior, no haze.
        This is the look of the classic Mandelbrot zoom videos; music only nudges brightness,
        hue drift and a very gentle pulse."""
        pal = scene["palette"]
        inside = mu < 0
        # colour follows log(iterations): bands stay wide and smooth far from the set and tighten
        # gracefully near it (linear iterations alias into pixel noise at depth)
        phase = np.log1p(np.where(inside, 0.0, mu)) * pal["cycles"] + hue_shift * 0.5 + t * 0.004
        d = np.array(pal["d"])
        cf = np.array(pal["c"])                                  # per-channel frequency ratios
        col = 0.52 + 0.46 * np.cos(2 * np.pi * (phase[..., None] * cf + d))
        # depth shading inside each band keeps gradients readable
        band = 0.86 + 0.14 * np.cos(2 * np.pi * phase * 3.0)
        col = col * band[..., None]
        col = np.clip(col, 0, 1) ** 1.1
        gain = 0.8 + 0.2 * intensity + 0.07 * pulse
        col = np.where(inside[..., None], 0.0, col * gain)
        return np.clip(col, 0, 1)

    def _colour(self, scene, mu, t, intensity, detail, hue_shift, pulse, w, aspect):
        if scene.get("style") == "classic":
            return self._colour_classic(scene, mu, t, intensity, hue_shift, pulse)
        pal = scene["palette"]
        H, W = mu.shape
        inside = mu < 0
        h = np.log1p(np.where(inside, 0.0, mu))
        phase = h * pal["freq"] * (1.0 + 0.15 * detail) + hue_shift * 0.35 + t * 0.006
        a, b, c, d = (np.array(pal[k]) for k in ("a", "b", "c", "d"))
        col = a + b * np.cos(2 * np.pi * (c * phase[..., None] + d))

        # relief lighting from the gradient of the smooth iteration field
        gy, gx = np.gradient(ndimage.gaussian_filter(h, 0.8 * max(1, W / 480)))
        la = scene["light_angle"]
        g = gx * math.cos(la) + gy * math.sin(la)
        scale = max(np.percentile(np.abs(g[~inside]), 90), 0.03) if (~inside).any() else 1.0
        shade = np.clip(g / (scale + 1e-9), -2, 2)
        col = col * (1.0 + scene["light_strength"] * 0.5 * shade[..., None])

        # moody falloff: the far field stays deep and dim, the boundary glows
        lum = col.mean(axis=-1, keepdims=True)
        col = lum + (col - lum) * 0.8                      # gentle desaturation
        fog = np.array(pal["fog"])
        edge = ndimage.gaussian_filter(np.hypot(gx, gy), 14.0 * max(1, W / 480))
        edge = np.clip(edge / max(np.percentile(edge, 95), 0.05), 0, 1)
        near = (0.2 + 0.8 * edge ** 0.6)[..., None]         # detail glows, flat field recedes
        col = col * near + fog * (1 - near) * 0.9

        # interior: dark body with faint glow near the boundary
        glow = ndimage.gaussian_filter((~inside).astype(np.float32), 9.0 * max(1, W / 480))
        tint = np.array(pal["a"]) + np.array(pal["b"]) * np.cos(2 * np.pi * (np.array(pal["c"]) * (0.2 + hue_shift * 0.35) + np.array(pal["d"])))
        interior = np.array(pal["interior"]) + 0.45 * glow[..., None] * tint * 0.55
        col = np.where(inside[..., None], interior, col)

        # music-driven grading: loud = richer contrast, quiet = softer and darker
        gain = 0.72 + 0.4 * intensity
        col = (col - 0.5) * (0.9 + 0.35 * intensity) + 0.5
        col = col * gain

        # bloom (strength follows bass pulse and brightness)
        bright = np.clip(col - 0.65, 0, None)
        bloom = np.stack([ndimage.gaussian_filter(bright[..., k], 6 * max(1, W / 480)) for k in range(3)], -1)
        col = col + bloom * (0.5 + 0.8 * pulse)

        # vignette
        yy, xx = np.mgrid[0:H, 0:W]
        r2 = ((xx / W - 0.5) * 2) ** 2 * 0.8 + ((yy / H - 0.5) * 2) ** 2
        col = col * (1.0 - 0.28 * np.clip(r2, 0, 1.5))[..., None]

        # soft shoulder tone map
        col = np.clip(col, 0, None)
        col = col / (1.0 + 0.25 * col)
        col = np.clip(col * 1.15, 0, 1)
        return col
