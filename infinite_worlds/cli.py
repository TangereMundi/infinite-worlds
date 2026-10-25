"""Command line front end. (A simple GUI is planned for a later phase.)"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .audio_analyzer import AudioAnalyzer, sha256_file
from .config import Config, PRESETS
from .hardware import detect
from .job_queue import JobQueue
from .pipeline import Pipeline


def _cfg(args) -> Config:
    p = Path(args.config)
    cfg = Config.load(p) if p.exists() else Config()
    if getattr(args, "quality", None):
        cfg.quality = args.quality
    if getattr(args, "direction", None):
        cfg.direction = args.direction
    if getattr(args, "style", None):
        cfg.style = args.style
    if getattr(args, "workers", None) is not None:
        cfg.workers = args.workers
    if getattr(args, "out", None):
        cfg.output_dir = args.out
    if getattr(args, "work", None):
        cfg.work_dir = args.work
    if getattr(args, "gen", None):
        cfg.generation_id = args.gen
    if getattr(args, "nondeterministic", False):
        cfg.deterministic = False
    if getattr(args, "overwrite", False):
        cfg.overwrite = True
    if getattr(args, "audio_policy", None):
        cfg.audio_policy = args.audio_policy
    return cfg


def _override(args) -> dict | None:
    o = {}
    if getattr(args, "size", None):
        w, h = args.size.lower().split("x")
        o["width"], o["height"] = int(w), int(h)
    if getattr(args, "fps", None):
        o["fps"] = args.fps
    if getattr(args, "render_scale", None):
        o["render_scale"] = args.render_scale
    return o or None


def _bar(frac: float) -> None:
    n = int(frac * 30)
    sys.stderr.write(f"\r  [{'#' * n}{'.' * (30 - n)}] {frac:5.1%}")
    sys.stderr.flush()
    if frac >= 1:
        sys.stderr.write("\n")


def cmd_hw(args):
    print(json.dumps(detect().to_dict(), indent=2))


def cmd_analyse(args):
    cfg = _cfg(args)
    a = AudioAnalyzer(Path(cfg.work_dir) / "cache").analyse(Path(args.file))
    print(json.dumps({"metadata": a.metadata, "measured": a.measured, "tempo": a.tempo,
                      "sections": a.sections}, indent=2))


def cmd_render(args):
    cfg = _cfg(args)
    pipe = Pipeline(cfg)
    rec = pipe.produce(Path(args.file), progress=_bar, preset_override=_override(args))
    if rec["status"] == "done":
        print(f"\nDone: {rec['output']}")
        print(f"  render {rec['render']['seconds']}s, total {rec['performance']['total_s']}s, "
              f"peak memory {rec['performance']['peak_memory_mb']} MB, QC passed")
    else:
        print(json.dumps(rec, indent=2))
        sys.exit(1)


def cmd_scan(args):
    cfg = _cfg(args)
    q = JobQueue(Path(cfg.work_dir) / "queue.sqlite")
    added = 0
    for p in sorted(Path(cfg.input_dir).rglob("*")):
        if p.suffix.lower() in cfg.extensions and p.is_file():
            _, new = q.add(p, sha256_file(p), cfg.generation_id)
            added += new
    print(f"added {added} new job(s); queue: {q.counts()}")


def cmd_run(args):
    cfg = _cfg(args)
    pipe = Pipeline(cfg)
    q = JobQueue(Path(cfg.work_dir) / "queue.sqlite")
    rec = q.recover()
    if rec:
        pipe.log.info("recovered %d interrupted job(s)", rec)
    while True:
        job = q.claim_next()
        if job is None:
            if args.watch:
                time.sleep(30)
                cmd_scan(args)
                continue
            break
        pipe.log.info("job %d: %s (%s)", job["id"], job["path"], job["generation_id"])
        try:
            r = pipe.produce(Path(job["path"]), job["generation_id"],
                             progress=lambda f, j=job["id"]: q.progress(j, f),
                             preset_override=_override(args))
            if r["status"] == "done":
                q.finish(job["id"], "done", r["output"])
            elif r["status"] == "skipped_exists":
                q.finish(job["id"], "skipped", r["output"])
            else:
                q.finish(job["id"], "failed", error="; ".join(r.get("issues", [])) or r["status"])
        except KeyboardInterrupt:
            pipe.log.warning("interrupted - job stays resumable")
            q.recover()
            raise
        except Exception as e:                       # record and carry on with the queue
            pipe.log.exception("job %d failed", job["id"])
            q.finish(job["id"], "failed", error=f"{type(e).__name__}: {e}")
    print("queue:", q.counts())


def cmd_status(args):
    cfg = _cfg(args)
    q = JobQueue(Path(cfg.work_dir) / "queue.sqlite")
    print("queue:", q.counts())
    for j in q.list():
        print(f"  #{j['id']:<3} {j['status']:<8} {j['progress']:>6.1%}  {Path(j['path']).name}  "
              f"[{j['generation_id']}] {j['error'] or ''}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="infinite_worlds")
    ap.add_argument("--config", default="infinite_worlds.json")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--quality", choices=list(PRESETS))
        p.add_argument("--direction")
        p.add_argument("--style", choices=["classic", "moody"])
        p.add_argument("--workers", type=int)
        p.add_argument("--out")
        p.add_argument("--work")
        p.add_argument("--gen")
        p.add_argument("--nondeterministic", action="store_true")
        p.add_argument("--overwrite", action="store_true")
        p.add_argument("--audio-policy", choices=["preserve", "compat"])
        p.add_argument("--size", help="override delivered size, e.g. 640x360")
        p.add_argument("--fps", type=int)
        p.add_argument("--render-scale", type=float)

    sub.add_parser("hw").set_defaults(fn=cmd_hw)
    p = sub.add_parser("analyse"); p.add_argument("file"); p.add_argument("--work"); p.set_defaults(fn=cmd_analyse)
    p = sub.add_parser("render"); p.add_argument("file"); common(p); p.set_defaults(fn=cmd_render)
    p = sub.add_parser("scan"); common(p); p.add_argument("--input-dir"); p.set_defaults(fn=cmd_scan)
    p = sub.add_parser("run"); common(p); p.add_argument("--watch", action="store_true"); p.set_defaults(fn=cmd_run)
    p = sub.add_parser("status"); common(p); p.set_defaults(fn=cmd_status)
    args = ap.parse_args(argv)
    args.fn(args)
