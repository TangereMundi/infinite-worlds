# Architecture

```
 music file ─▶ AudioAnalyzer ─▶ SeedEngine ─▶ VisualDirector ─▶ VisualPlan ─▶ RenderManager ─▶ AudioVideoMuxer ─▶ QualityController ─▶ output + .iw.json
                 (features,       (AudioSeed,     (scenes, transitions,   (pure data:       (chunked, atomic,     (original audio      (measured checks,
                  cached by        GenerationSeed, events, controls,       identical after   resumable frames      stream-copied)       bounded retries,
                  hash+version)    sub-streams)    novelty memory)         a restart)        via engines)                               quarantine)
                                                          ▲
                                          VisualEngineRegistry (fractal today; tunnel, organic, space, assets, hybrid later)
 JobQueue (SQLite) drives Pipeline.produce() per track; AuditLogger writes work/logs; ConfigurationManager = Config + presets.
```

## Contracts that keep it extensible and crash-safe

* **Engines are pure.** `render(scene, t, controls, w, h)` depends only on its arguments. That is why frames can be rendered in any order, in parallel, and re-rendered after a crash with identical results. A new engine implements `sample_scene`, `render`, `fingerprint` (`engines/base.py`) and is registered in `default_registry()`.
* **The plan is data.** `VisualPlan` (scenes, transitions, events, control curves) is a pure function of (analysis, seeds, settings). Its hash is stored beside partial chunks; if it changes, stale chunks are discarded rather than silently mixed.
* **Randomness is portable and partitioned.** `DeterministicRNG` (SHA-256 counter mode) gives the same stream on any OS or numpy version; `fork(name)` creates independent streams so adding a draw in one place never changes another.
* **Nothing is visible until complete.** Chunks, the video, the muxed file and the JSON record are written as `.part` and atomically renamed. Existing finals are never overwritten unless `overwrite` is set.
* **Seeds are persisted** at first use, so a resumed non-deterministic job keeps its time component.
* **The soundtrack is never re-encoded** by default: the original file's audio stream is copied; QC verifies packets or decoded samples match.

## Where each spec module lives

| Spec | File |
|---|---|
| AudioAnalyzer | `audio_analyzer.py` |
| SeedEngine | `seed_engine.py` |
| VisualEngineRegistry, engine interface | `engines/base.py` |
| FractalEngine | `engines/fractal.py` |
| VisualDirector | `visual_director.py` |
| RenderManager, AudioVideoMuxer | `render_manager.py` |
| QualityController | `quality.py` |
| JobQueue | `job_queue.py` |
| ConfigurationManager | `config.py` |
| AuditLogger | `pipeline.py` (`setup_logging`) |
| Hardware detection | `hardware.py` |
| CLI | `cli.py`, `__main__.py` |
| Not yet built | TunnelEngine, OrganicWorldEngine, SpaceEngine, AssetLibrary, HybridComposer, MetadataGenerator, UploadManager, GUI |

## The per-output record (`<name>.iw.json`)
Source hash and metadata, measured audio features, both seeds, engine versions, settings, the full scene plan (targets, times, novelty distances), events, QC metrics and retry history, 1-fps perceptual fingerprints, timings, peak memory and hardware. VRAM is `null` until the GPU backend exists.
