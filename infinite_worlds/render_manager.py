"""RenderManager + AudioVideoMuxer.

Frames are a pure function of (plan, frame index), so rendering is split into
fixed-length chunks. Each chunk is encoded to its own file and only becomes
visible (atomic rename) when complete; after a crash or restart, finished chunks
are verified and skipped and only the missing ones are rendered.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from collections import deque
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage

from .audio_analyzer import find_tool
from .engines.base import default_registry
from .seed_engine import DeterministicRNG
from .visual_director import VisualPlan, _smoothstep

CHUNK_SECONDS = 10

# --------------------------------------------------------------------------
# worker-side frame composition (module level so it pickles on Windows 'spawn')
# --------------------------------------------------------------------------
_W: dict = {}


def _load_watermark(wm: dict, out_w: int, out_h: int):
    if not wm or not wm.get("enabled") or not wm.get("path"):
        return None
    from PIL import Image
    im = Image.open(wm["path"]).convert("RGBA")
    tw = max(8, int(out_w * float(wm.get("width_fraction", 0.12))))
    th = max(8, int(im.height * tw / im.width))
    im = im.resize((tw, th), Image.LANCZOS)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr[..., 3] *= float(wm.get("opacity", 0.6))
    m = int(out_w * float(wm.get("margin_fraction", 0.03)))
    pos = wm.get("position", "bottom-right")
    x = m if "left" in pos else out_w - tw - m
    y = m if "top" in pos else out_h - th - m
    return arr, x, y


def _init_worker(plan_json: dict, controls_blob: dict, size: tuple, ss: int, wm: dict, out_size: tuple):
    plan = VisualPlan(
        duration=plan_json["duration"], fps=plan_json["fps"], scenes=plan_json["scenes"],
        transitions=plan_json["transitions"], events=plan_json["events"],
        controls={k: np.asarray(v) for k, v in controls_blob.items()},
        fade_in=plan_json["fade_in"], fade_out=plan_json["fade_out"])
    _W["plan"] = plan
    _W["registry"] = default_registry()
    _W["size"] = size
    _W["ss"] = ss
    _W["wm"] = _load_watermark(wm, *size) if wm else None
    _W["masks"] = {}


def _dissolve_mask(idx: int, h: int, w: int) -> np.ndarray:
    key = (idx, h, w)
    if key not in _W["masks"]:
        rng = DeterministicRNG("dissolve" + "0" * 56, f"t{idx}")
        gh, gw = 9, 16
        grid = np.array([[rng.random() for _ in range(gw)] for _ in range(gh)])
        m = ndimage.zoom(grid, (h / gh, w / gw), order=3)[:h, :w]
        m = (m - m.min()) / (np.ptp(m) + 1e-9)
        _W["masks"][key] = m.astype(np.float32)
    return _W["masks"][key]


def render_frame_rgb(i: int) -> bytes:
    plan: VisualPlan = _W["plan"]
    reg = _W["registry"]
    W, H = _W["size"]
    t = i / plan.fps
    ctrl = plan.controls_at(t)
    acts = plan.active_scenes(t)
    frames = []
    for si, wgt in acts:
        sc = plan.scenes[si]
        eng = reg.get(sc["engine"])
        frames.append(eng.render(sc["scene"], t - sc["start"], ctrl, W, H, _W["ss"]))
    if len(frames) == 1:
        img = frames[0]
    else:
        u = acts[1][1]
        tr = plan.transitions[acts[0][0]]
        if tr["type"] == "dissolve":
            m = _dissolve_mask(acts[0][0], H, W)
            local = np.clip(u * 1.5 - m * 0.5, 0.0, 1.0)
            local = local * local * (3 - 2 * local)
            img = frames[0] * (1 - local[..., None]) + frames[1] * local[..., None]
        else:
            img = frames[0] * (1 - u) + frames[1] * u
    img = img * plan.global_fade(t)
    wm = _W["wm"]
    if wm is not None:
        arr, x, y = wm
        a = arr[..., 3:4]
        h, w = arr.shape[:2]
        img[y:y + h, x:x + w] = img[y:y + h, x:x + w] * (1 - a) + arr[..., :3] * a
    # TPDF dither (~1 LSB) to prevent banding in dark gradients after 8-bit quantisation
    rng = np.random.default_rng(i)
    noise = (rng.random(img.shape, dtype=np.float32) + rng.random(img.shape, dtype=np.float32) - 1.0) / 255.0
    out = np.clip(img + noise, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8).tobytes()


# --------------------------------------------------------------------------
class RenderManager:
    def __init__(self, hardware, log):
        self.hw = hardware
        self.log = log

    def video_encoder_args(self, encoder: str, preset: dict) -> list[str]:
        use_nvenc = encoder == "h264_nvenc" or (encoder == "auto" and self.hw.nvenc_usable)
        if use_nvenc:
            return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(preset["crf"] + 1),
                    "-b:v", "0", "-pix_fmt", "yuv420p"]
        return ["-c:v", "libx264", "-preset", preset["x264_preset"], "-crf", str(preset["crf"]),
                "-pix_fmt", "yuv420p", "-profile:v", "high"]

    def render(self, plan: VisualPlan, work_dir: Path, out_video: Path, preset: dict,
               encoder: str = "auto", workers: int = 0, watermark: dict | None = None,
               progress=None) -> dict:
        work_dir = Path(work_dir)
        chunk_dir = work_dir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        fps = plan.fps
        out_w, out_h = preset["width"], preset["height"]
        rw = max(64, int(round(out_w * preset["render_scale"] / 2)) * 2)
        rh = max(36, int(round(out_h * preset["render_scale"] / 2)) * 2)
        ss = int(preset.get("supersample", 1))
        total = int(round(plan.duration * fps))
        per_chunk = fps * CHUNK_SECONDS
        n_chunks = int(math.ceil(total / per_chunk))
        workers = workers or self.hw.recommended_workers()
        enc_args = self.video_encoder_args(encoder, preset)
        ffmpeg = find_tool("ffmpeg")

        controls_blob = {k: v.tolist() for k, v in plan.controls.items()}
        pj = plan.to_json_dict()
        initargs = (pj, controls_blob, (rw, rh), ss, watermark or {}, (out_w, out_h))

        t0 = time.time()
        rendered_chunks = skipped_chunks = 0
        pool = None
        if workers > 1:
            # 'spawn' (the only mode on Windows): workers must not inherit the encoder pipe, or
            # ffmpeg never sees end-of-input and the render hangs.
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                       initializer=_init_worker, initargs=initargs)
        else:
            _init_worker(*initargs)
        try:
            chunk_files = []
            for c in range(n_chunks):
                a, b = c * per_chunk, min(total, (c + 1) * per_chunk)
                final = chunk_dir / f"chunk_{c:05d}.mkv"
                chunk_files.append(final)
                if final.exists() and self._count_frames(final) == b - a:
                    skipped_chunks += 1
                    self.log.info("chunk %d/%d already complete - skipping", c + 1, n_chunks)
                    if progress:
                        progress(b / total)
                    continue
                part = chunk_dir / f"chunk_{c:05d}.part.mkv"
                if part.exists():
                    part.unlink()
                cmd = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{rw}x{rh}", "-r", str(fps), "-i", "-",
                       "-vf", f"scale={out_w}:{out_h}:flags=lanczos:out_color_matrix=bt709,format=yuv420p"
                       if (rw, rh) != (out_w, out_h) else "scale=out_color_matrix=bt709,format=yuv420p",
                       *enc_args, "-g", str(fps * 2), "-color_primaries", "bt709", "-color_trc", "bt709",
                       "-colorspace", "bt709", "-f", "matroska", str(part)]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    if pool is None:
                        for i in range(a, b):
                            proc.stdin.write(render_frame_rgb(i))
                    else:
                        window: deque = deque()
                        nxt = a
                        while nxt < b or window:
                            while nxt < b and len(window) < workers * 3:
                                window.append(pool.submit(render_frame_rgb, nxt))
                                nxt += 1
                            proc.stdin.write(window.popleft().result())
                    proc.stdin.close()
                    err = proc.stderr.read().decode(errors="replace")
                    proc.stderr.close()
                    if proc.wait() != 0:
                        raise RuntimeError(f"ffmpeg chunk encode failed: {err[-400:]}")
                except BaseException:
                    proc.kill()
                    proc.wait()
                    if proc.stderr:
                        proc.stderr.close()
                    raise
                os.replace(part, final)           # atomic: a chunk is either complete or absent
                rendered_chunks += 1
                if progress:
                    progress(b / total)
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)

        # concatenate chunks without re-encoding
        lst = work_dir / "chunks.txt"
        lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in chunk_files))
        part = out_video.with_suffix(".part.mkv")
        subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", str(part)], check=True)
        os.replace(part, out_video)
        return {"frames": total, "fps": fps, "render_size": [rw, rh], "output_size": [out_w, out_h],
                "chunks": n_chunks, "chunks_rendered": rendered_chunks, "chunks_resumed": skipped_chunks,
                "workers": workers, "encoder": enc_args[1], "seconds": round(time.time() - t0, 2)}

    @staticmethod
    def _count_frames(path: Path) -> int:
        try:
            out = subprocess.run(
                [find_tool("ffprobe"), "-v", "error", "-count_packets", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                capture_output=True, text=True, check=True).stdout.strip()
            return int(out)
        except Exception:
            return -1


# --------------------------------------------------------------------------
class AudioVideoMuxer:
    """Combine the silent video with the ORIGINAL audio file, stream-copied when possible."""

    MP4_COPY_OK = {"aac", "mp3", "alac", "ac3", "eac3", "opus"}

    def mux(self, video: Path, audio: Path, audio_codec: str, policy: str, out_base: Path) -> tuple[Path, dict]:
        ffmpeg = find_tool("ffmpeg")
        if audio_codec in self.MP4_COPY_OK:
            ext, acodec, mode = ".mp4", ["-c:a", "copy"], "copy"
        elif policy == "compat":
            ext, acodec, mode = ".mp4", ["-c:a", "aac", "-b:a", "320k"], "aac320k (compat policy, lossy)"
        else:
            ext, acodec, mode = ".mkv", ["-c:a", "copy"], "copy"
        final = out_base.with_suffix(ext)
        part = final.with_name(final.stem + ".part" + ext)
        cmd = [ffmpeg, "-v", "error", "-y", "-i", str(video), "-i", str(audio),
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *acodec, "-map_metadata", "1"]
        if ext == ".mp4":
            cmd += ["-movflags", "+faststart"]
        cmd += [str(part)]
        subprocess.run(cmd, check=True)
        return part, {"audio_mode": mode, "container": ext}
