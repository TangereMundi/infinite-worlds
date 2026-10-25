"""Pipeline: the end-to-end production flow for one track (see spec section 9)."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
from pathlib import Path

from . import __version__, ANALYSIS_VERSION
from .audio_analyzer import AudioAnalyzer, sha256_file
from .config import Config
from .engines.base import default_registry
from .hardware import Hardware, detect
from .quality import QualityController
from .render_manager import AudioVideoMuxer, RenderManager
from .seed_engine import SeedEngine, Seeds
from .visual_director import VisualDirector


def setup_logging(work_dir: Path) -> logging.Logger:
    """AuditLogger: rotating file log plus console."""
    from logging.handlers import RotatingFileHandler
    log = logging.getLogger("infinite_worlds")
    log.setLevel(logging.INFO)
    target = (work_dir / "logs" / "infinite_worlds.log").resolve()
    for h in list(log.handlers):                      # re-point handlers if the work dir changed
        if isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == target:
            return log
        if isinstance(h, RotatingFileHandler):
            h.close()
            log.removeHandler(h)
    (work_dir / "logs").mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(target, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
               for h in log.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(sh)
    return log


class PeakMemory:
    """Samples resident memory of this process tree; peak is recorded with every output."""

    def __init__(self):
        self.peak_mb = None
        self._stop = threading.Event()
        try:
            import psutil
            self._ps = psutil
        except Exception:
            self._ps = None

    def _run(self):
        me = self._ps.Process()
        while not self._stop.is_set():
            try:
                total = me.memory_info().rss + sum(c.memory_info().rss for c in me.children(recursive=True))
                self.peak_mb = max(self.peak_mb or 0, total / 2**20)
            except Exception:
                pass
            self._stop.wait(0.5)

    def __enter__(self):
        if self._ps:
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()


def safe_stem(p: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", p.stem)[:80] or "track"


class Pipeline:
    def __init__(self, config: Config, base_dir: Path = Path(".")):
        self.cfg = config
        self.base = Path(base_dir)
        self.work = self.base / config.work_dir
        self.out_dir = self.base / config.output_dir
        self.work.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = setup_logging(self.work)
        self.hw: Hardware = detect()
        self.registry = default_registry()
        self.analyzer = AudioAnalyzer(self.work / "cache")
        self.engine_version = "|".join(f"{k}={v}" for k, v in self.registry.versions().items())
        self.seed_engine = SeedEngine(self.engine_version)
        self.renderer = RenderManager(self.hw, self.log)

    # ------------------------------------------------------------------
    def existing_output(self, audio: Path, gid: str) -> Path | None:
        base = self.out_dir / f"{safe_stem(audio)}__{gid}"
        for ext in (".mp4", ".mkv"):
            p = base.with_suffix(ext)
            if p.exists():
                return p
        return None

    def produce(self, audio: Path, generation_id: str | None = None, deterministic: bool | None = None,
                progress=None, preset_override: dict | None = None) -> dict:
        cfg = self.cfg
        audio = Path(audio)
        gid = generation_id or cfg.generation_id
        det = cfg.deterministic if deterministic is None else deterministic
        base = self.out_dir / f"{safe_stem(audio)}__{gid}"

        exist = self.existing_output(audio, gid)
        if exist and not cfg.overwrite:
            self.log.info("output already exists, not overwriting: %s", exist)
            return {"status": "skipped_exists", "output": str(exist)}

        preset = cfg.preset()
        if preset_override:
            preset.update(preset_override)
        t_start = time.time()
        timings: dict = {}
        qc_history = []
        with PeakMemory() as mem:
            t = time.time()
            analysis = self.analyzer.analyse(audio)
            timings["analysis_s"] = round(time.time() - t, 2)
            hd = analysis.metadata.get("duration") or 0.0
            if abs(hd - analysis.metadata["analysed_duration"]) > 1.0:
                self.log.warning("%s: file header says %.1fs but the audio decodes to %.1fs - using the decoded length",
                                 audio.name, hd, analysis.metadata["analysed_duration"])
            self.log.info("analysed %s: %.1fs, tempo=%s (conf %.2f), %d sections",
                          audio.name, analysis.metadata["analysed_duration"], analysis.measured["tempo_bpm"],
                          analysis.measured["tempo_confidence"], len(analysis.sections))
            job_root = self.work / "jobs" / f"{analysis.content_hash[:16]}_{gid}"
            job_root.mkdir(parents=True, exist_ok=True)

            result = None
            for attempt in range(cfg.max_qc_retries + 1):
                eff_gid = gid if attempt == 0 else f"{gid}~r{attempt}"
                seeds = self._load_or_make_seeds(job_root, analysis, eff_gid, det)
                adir = job_root / f"attempt{attempt}"
                adir.mkdir(exist_ok=True)

                director = VisualDirector(self.registry, {"direction": cfg.direction, "style": cfg.style})
                t = time.time()
                plan = director.build_plan(analysis, seeds, preset["fps"])
                timings[f"plan_a{attempt}_s"] = round(time.time() - t, 2)
                self._check_plan_matches(adir, plan)

                vid = adir / "video_only.mkv"
                t = time.time()
                rinfo = self.renderer.render(
                    plan, adir, vid, preset, encoder=cfg.encoder, workers=cfg.workers,
                    watermark=cfg.watermark if cfg.watermark.get("enabled") else None, progress=progress)
                timings[f"render_a{attempt}_s"] = round(time.time() - t, 2)

                t = time.time()
                part, minfo = AudioVideoMuxer().mux(vid, audio, analysis.metadata["codec"], cfg.audio_policy,
                                                    adir / "muxed")
                timings[f"mux_a{attempt}_s"] = round(time.time() - t, 2)

                t = time.time()
                qc = QualityController(plan.fade_in, plan.fade_out).check(
                    part, audio, (preset["width"], preset["height"]), preset["fps"],
                    analysis.metadata["analysed_duration"])
                timings[f"qc_a{attempt}_s"] = round(time.time() - t, 2)
                qc_history.append({"attempt": attempt, "effective_generation_id": eff_gid,
                                   "passed": qc.passed, "issues": qc.issues, "warnings": qc.warnings})
                self.log.info("QC attempt %d: %s %s", attempt, "PASS" if qc.passed else "FAIL", qc.issues)
                result = (seeds, plan, rinfo, minfo, qc, part)
                if qc.passed:
                    break

            seeds, plan, rinfo, minfo, qc, part = result
            if not qc.passed:
                rej = self.work / "rejected"
                rej.mkdir(exist_ok=True)
                moved = rej / f"{base.name}{part.suffix}"
                os.replace(part, moved)
                rec = {"status": "failed_qc", "issues": qc.issues, "rejected_file": str(moved),
                       "qc_history": qc_history}
                (rej / f"{base.name}.iw.json").write_text(json.dumps(rec, indent=2))
                return rec

            final = base.with_suffix(minfo["container"])
            os.replace(part, final)                       # atomic finalisation
            peak = mem.peak_mb
        record = self._record(audio, final, analysis, seeds, plan, rinfo, minfo, qc, qc_history, preset,
                              timings, peak, time.time() - t_start)
        tmp = final.with_suffix(".iw.json.part")
        tmp.write_text(json.dumps(record, indent=2))
        os.replace(tmp, final.with_suffix(".iw.json"))
        shutil.rmtree(job_root, ignore_errors=True)       # chunks no longer needed
        return record

    # ------------------------------------------------------------------
    def _load_or_make_seeds(self, job_root: Path, analysis, gid: str, det: bool) -> Seeds:
        """Seeds are persisted on first creation so a resumed job always uses the same ones
        (important for non-deterministic mode, whose time component must not change)."""
        f = job_root / f"seeds_{re.sub(r'[^A-Za-z0-9_.~-]', '_', gid)}.json"
        if f.exists():
            d = json.loads(f.read_text())
            return Seeds(**d)
        s = self.seed_engine.make_seeds(analysis, gid, deterministic=det)
        f.write_text(json.dumps(s.to_dict(), indent=2))
        return s

    def _check_plan_matches(self, adir: Path, plan) -> None:
        ph = plan.plan_hash()
        f = adir / "plan.hash"
        if f.exists() and f.read_text() != ph:
            self.log.warning("plan changed since interrupted run - discarding partial chunks")
            shutil.rmtree(adir / "chunks", ignore_errors=True)
        f.write_text(ph)

    def _record(self, audio, final, analysis, seeds, plan, rinfo, minfo, qc, qc_history, preset,
                timings, peak_mb, total_s) -> dict:
        return {
            "status": "done",
            "application": {"name": "Infinite Worlds", "version": __version__,
                            "analysis_version": ANALYSIS_VERSION, "engines": self.registry.versions()},
            "source": {"file": audio.name, "sha256": analysis.content_hash, "metadata": analysis.metadata},
            "audio": {"measured": analysis.measured, "tempo": analysis.tempo,
                      "sections": analysis.sections, "beat_count": len(analysis.beats)},
            "seeds": seeds.to_dict(),
            "settings": {"quality": self.cfg.quality, "preset": preset, "direction": self.cfg.direction, "style": self.cfg.style,
                         "audio_policy": self.cfg.audio_policy, "encoder_setting": self.cfg.encoder,
                         "watermark_enabled": bool(self.cfg.watermark.get("enabled"))},
            "plan": {"plan_hash": plan.plan_hash(), "scenes": [
                {"index": s["index"], "engine": s["engine"], "start": s["start"], "end": s["end"],
                 "mode": s["scene"].get("mode"), "target": s["scene"].get("target"),
                 "novelty_distance": s["novelty_distance"], "climax": s["climax"],
                 "mean_intensity": s["mean_intensity"]} for s in plan.scenes],
                "transitions": plan.transitions, "events": plan.events},
            "render": rinfo, "mux": minfo,
            "quality_control": {**{k: v for k, v in qc.to_dict().items() if k != "fingerprints"},
                                "history": qc_history},
            "fingerprints_dhash_1fps": [format(f, "016x") for f in qc.fingerprints],
            "performance": {"timings_s": timings, "total_s": round(total_s, 2),
                            "peak_memory_mb": round(peak_mb, 1) if peak_mb else None,
                            "vram_peak_mb": None, "hardware": self.hw.to_dict()},
            "output": str(final),
        }
