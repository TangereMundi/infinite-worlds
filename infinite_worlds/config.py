"""ConfigurationManager: JSON-backed settings with a sensible default for everything."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

PRESETS = {
    # width/height = delivered video. render_scale = internal render size for the CPU
    # reference backend (upscaled with Lanczos by ffmpeg). A GPU backend renders at 1.0.
    "preview":         dict(width=640,  height=360,  fps=24, render_scale=1.0,   supersample=1, crf=26, x264_preset="veryfast"),
    "standard_1080p":  dict(width=1920, height=1080, fps=30, render_scale=0.5,   supersample=1, crf=18, x264_preset="medium"),
    "high_1080p":      dict(width=1920, height=1080, fps=30, render_scale=0.75,  supersample=1, crf=16, x264_preset="slow"),
    "experimental_4k": dict(width=3840, height=2160, fps=30, render_scale=0.375, supersample=1, crf=17, x264_preset="medium"),
}


@dataclass
class Config:
    input_dir: str = "input"
    output_dir: str = "output"
    work_dir: str = "work"
    direction: str = "auto"            # auto | mandelbrot | julia (more as engines arrive)
    style: str = "classic"             # classic: saturated colour bands, black interior | moody: lit, hazy
    quality: str = "standard_1080p"
    deterministic: bool = True
    generation_id: str = "g1"
    audio_policy: str = "preserve"     # preserve: bit-exact copy (mkv for wav/flac) | compat: AAC 320k in mp4
    overwrite: bool = False
    max_qc_retries: int = 2
    workers: int = 0                   # 0 = auto
    encoder: str = "auto"              # auto | libx264 | h264_nvenc
    watermark: dict = field(default_factory=lambda: {
        "enabled": False, "path": "", "position": "bottom-right",
        "width_fraction": 0.12, "opacity": 0.6, "margin_fraction": 0.03})
    extensions: tuple = (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".aiff")

    def save(self, path: Path) -> None:
        d = asdict(self)
        d["extensions"] = list(self.extensions)
        path.write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Config":
        d = json.loads(Path(path).read_text())
        if "extensions" in d:
            d["extensions"] = tuple(d["extensions"])
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def preset(self) -> dict:
        return dict(PRESETS[self.quality])
