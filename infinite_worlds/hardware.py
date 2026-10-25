"""Hardware detection and encoder selection. Detection only; nothing is assumed."""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, asdict

try:
    import psutil
except Exception:  # psutil is optional
    psutil = None


@dataclass
class Hardware:
    cpu_logical: int
    cpu_physical: int | None
    ram_total_gb: float | None
    ram_available_gb: float | None
    gpu_name: str | None
    vram_total_mb: int | None
    nvenc_usable: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def recommended_workers(self) -> int:
        phys = self.cpu_physical or max(1, self.cpu_logical // 2)
        w = max(1, phys - 1) if phys > 2 else max(1, phys)
        if self.ram_available_gb is not None:
            w = min(w, max(1, int(self.ram_available_gb // 2)))   # ~2 GB budget per worker
        return w


def _gpu() -> tuple[str | None, int | None]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        if out:
            name, mem = [p.strip() for p in out[0].split(",")]
            return name, int(float(mem))
    except Exception:
        pass
    return None, None


def _nvenc_works() -> bool:
    ff = shutil.which("ffmpeg")
    if not ff:
        return False
    try:
        r = subprocess.run([ff, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x144:d=0.2",
                            "-c:v", "h264_nvenc", "-f", "null", "-"],
                           capture_output=True, timeout=20)
        return r.returncode == 0
    except Exception:
        return False


def detect() -> Hardware:
    gpu, vram = _gpu()
    return Hardware(
        cpu_logical=os.cpu_count() or 1,
        cpu_physical=psutil.cpu_count(logical=False) if psutil else None,
        ram_total_gb=round(psutil.virtual_memory().total / 2**30, 1) if psutil else None,
        ram_available_gb=round(psutil.virtual_memory().available / 2**30, 1) if psutil else None,
        gpu_name=gpu, vram_total_mb=vram,
        nvenc_usable=bool(gpu) and _nvenc_works(),
    )
