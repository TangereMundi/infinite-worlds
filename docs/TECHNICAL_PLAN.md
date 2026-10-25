# Infinite Worlds — Technical plan (Phase 1) and status

Status legend: **✅ tested** (an automated test or measured run exists) · **🟡 partial** · **⬜ not built**.
Nothing is marked ✅ unless a test in `tests/` or a recorded run backs it.

## 1. What exists today (prototype 0.1.0)

| Spec module | Status | Notes |
|---|---|---|
| AudioAnalyzer | ✅ | numpy/scipy + ffmpeg. Loudness (EBU R128), spectral features, band energy, harmonic/percussive, chroma, tempo **with confidence + candidates**, beats, sections, intensity curve, absolute energy level. Cached by content hash + analysis version. |
| SeedEngine | ✅ | AudioSeed / GenerationSeed exactly as specified; portable SHA-256 counter-mode PRNG with independent named sub-streams (adding a draw to one consumer never shifts another). |
| FractalEngine (Engine A) | 🟡 | **Default "classic" style (slow zoom in → fade → zoom out cycle)** and the older "moody" style. Mandelbrot + animated Julia, smooth colouring, relief lighting, bloom, autonomous coarse-to-fine **discovery** of structured regions. **No** Mandelbulb / quaternion / distance-estimated 3D. CPU float64 only. |
| VisualDirector | 🟡 | Sections → scenes snapped to confident beats, transitions (crossfade / noise dissolve), climax event that forces a mode change, novelty memory (avoids repeating scenes inside a video), intensity-driven camera speed, smoothed band-driven controls. Event types implemented: discovery, section change, scene change, climax, resolution. |
| RenderManager / AudioVideoMuxer | ✅ | Parallel frame workers, chunked atomic encoding, resume, lossless concat, stream-copy mux of the **original** audio. |
| QualityController | 🟡 | Duration, resolution, fps, black frames, brightness, flicker, abrupt jumps, frozen image, in-video repetition (dHash), audio integrity, bounded retries, quarantine. Cross-video comparison helper exists. **No learned embeddings.** |
| JobQueue / AuditLogger / ConfigurationManager | ✅ | SQLite queue, crash recovery, dedupe by (content hash, generation), rotating log, JSON config. |
| Watermark | ✅ (still frame test) | PNG, position, size, opacity, margin. Composited in the frame compositor; never baked into assets. |
| TunnelEngine, OrganicWorldEngine, SpaceEngine | ⬜ | |
| Asset library (Engine E) + HybridComposer (Engine F) | ⬜ | Phase 4. |
| Title treatments, MetadataGenerator, UploadManager (YouTube) | ⬜ | Phase 6. Per-output JSON record (seeds, features, plan, QC, timings) already written. |
| GUI | ⬜ | CLI only today. |
| GPU renderer (OpenGL/GLSL) | ⬜ | Could **not** be built or tested here (no GPU in the build sandbox). See §3. |

## 2. Technology decisions

| Task | Choice | Why | Alternatives considered |
|---|---|---|---|
| Orchestration, analysis | Python 3.11+ | Fast to iterate; numpy/scipy are enough for every feature in spec §3.1. | C++ (more work, little gain for analysis). |
| Audio features | numpy + scipy (own implementations) | One small dependency set; behaviour is inspectable and versioned (`ANALYSIS_VERSION`). | librosa (good, but pulls numba/soundfile/audioread, a fragile stack on Windows); Essentia (AGPL). |
| Decode / loudness / mux | FFmpeg + ffprobe | De-facto standard; stream copy keeps the audio master bit-exact. | — |
| Video encode | libx264 (CPU) or h264_nvenc (auto-detected, test-encoded first) | Quality-first CRF; NVENC as speed option. | HEVC/AV1 later (YouTube accepts H.264 fine). |
| Frame rendering today | numpy, multiprocessing (`spawn`) | Portable, testable anywhere. Slow at high resolution. | numba (not installable in the build sandbox; optional later accelerator). |
| **Frame rendering target** | **OpenGL 4.3 fragment/compute shaders via `moderngl`** | Python-friendly, MIT-licensed (to be verified, §6), headless context, runs on a GTX 1660. | Vulkan (more work), CUDA (locks to NVIDIA), wgpu. |
| Deep zoom | float64 now; **perturbation theory** (high-precision reference orbit on CPU, per-pixel deltas on GPU in float32/float64) later | The standard technique for deep Mandelbrot zooms; fp32 deltas are fast on Turing, fp64 on GeForce is heavily rate-limited. | Arbitrary precision per pixel (far too slow). |
| Persistence | SQLite (stdlib) | Zero install, crash-safe (WAL). | — |
| Similarity | 64-bit dHash now; image embeddings (small ONNX model, offline) later | Hash catches near-duplicates only — documented limitation. | CLIP-class embeddings (adds model + runtime). |

## 3. Risks and how they are handled

| Risk | Handling today | Remaining work |
|---|---|---|
| **Deep-zoom precision** | Hard floor `MIN_WIDTH = 5e-10`; test asserts pixel step stays > 100× float64 resolution even at 4K. Scenes end (and hand over to a newly *discovered* region) instead of zooming forever. | Perturbation renderer for genuinely deep, continuous zooms. |
| **Long-video rendering** | Chunked, atomic, resumable; 236 s track rendered and verified; interrupted-render test passes. | Benchmark on target PC. |
| **CPU render speed** | Real and unavoidable: pure numpy is orders of magnitude slower than a GPU shader. Quality presets use an internal `render_scale` + Lanczos upscale on CPU. | The GPU backend is the plan for 1080p/4K quality. **No speed promise is made before benchmarking on the i9-9900 / GTX 1660.** |
| **Temporal repetition** | Per-video novelty memory; QC measures in-video repetition (dHash); long-track test asserts no recycled sequences. | Cross-library comparison, embeddings. |
| **GPU memory (6 GB)** | Design budget: ≤ 2 GB per render target set; tile 4K; no per-frame texture uploads. | Needs real GPU measurements. |
| Tempo/section errors | Tempo carries confidence; beats used only above a confidence threshold; sections are one input among several. | — |
| Colour banding in dark gradients | TPDF dither before 8-bit quantisation. | Optional 10-bit HEVC output. |
| MP3 in MP4 | Packets copy bit-exact; decoded start differs (~25 ms encoder-delay handling) — integrity check compares packets *or* decoded samples. | — |

## 4. Acceptance criteria and results (spec §15)

Run: `python -m unittest discover -s tests -v` (about 10 minutes on 2 cores). Test sizes are tiny; they check behaviour, not final image quality.

| # | Case | Check | Result |
|---|---|---|---|
| 1 | Two different tracks | different audio seeds, different features, little shared imagery | ✅ pass |
| 2 | Two generations of one track | same audio seed, different generation seed, different scene targets, little shared imagery | ✅ pass |
| 3 | Same track, identical deterministic settings | identical plan hash, identical scenes, identical decoded video | ✅ pass |
| 4 | Long quiet passages | lower visual energy than the dynamic track; QC passes | ✅ pass |
| 5 | Dynamic, strong transients | QC passes, climax event, multiple modes | ✅ pass |
| 6 | Long track (236 s) | full duration, ≥ 6 scenes, no recycled imagery, memory recorded | ✅ pass (tiny frames; peak ≈ 400 MB) |
| 7 | Interrupted render | killed mid-render → nothing published → rerun resumes ≥ 2 chunks → same plan hash | ✅ pass |
| 8 | Odd metadata | mono, 8-bit, 11 kHz, no tags, unicode filename | ✅ pass |
| 9/10 | No external assets / procedural only | all scenes from the fractal engine | ✅ pass |
| — | Audio integrity | WAV, FLAC, M4A, MP3 sources verified; lossy compat policy flagged | ✅ pass |
| — | QC negatives | black, frozen, truncated, wrong-resolution videos are rejected; failed QC is quarantined after bounded retries | ✅ pass |
| — | Queue | restart recovery, no duplicate jobs, one bad file does not stop the queue | ✅ pass |

36 automated tests, all passing on the last run (Linux sandbox, 2 cores). **Not yet run:** anything on Windows 11, the GTX 1660, NVENC, or a real music library. Test videos are tiny, so they prove behaviour, not 1080p image quality.

### Measured CPU render cost (2-core sandbox, single core per frame)
| Preset | Internal render size | Seconds per frame | Core-seconds per second of video (30 fps) |
|---|---|---|---|
| standard_1080p | 960×540 → upscaled | 1.4 | ≈ 43 |
| high_1080p | 1440×810 → upscaled | 4.3 | ≈ 130 |
| experimental_4k | 1440×810 → upscaled | 4.2 | ≈ 125 |

A 4-minute track at `standard_1080p` is therefore ≈ 10,000 core-seconds. Even if your i9-9900 runs 7 workers about 1.5× faster per core than this sandbox, expect roughly **half an hour per 4-minute track** — workable for a handful of tracks, **not** for a 20,000-track library. This is why the GPU shader backend is the next milestone, not an optional extra. These figures are from this sandbox, not your PC; run `render` on a 30-second track to get yours.


## 5. Roadmap

1. **Phase 3 (next):** richer event library (hidden-opening reveals, 2D→3D transitions), per-section engine choice.
2. **GPU backend** for the fractal engine (moderngl), then Mandelbulb/DE rendering and the Tunnel engine (SDF raymarching) — needs a machine with the GTX 1660 to test.
3. **Phase 4:** asset indexer (hash, ffprobe facts, dominant colours, motion energy, loop points, dHash similarity, provenance fields) and HybridComposer.
4. **Phase 5:** persistent cross-video fingerprint index, richer retry strategy.
5. **Phase 6:** titles, thumbnails, chapter timestamps, YouTube Data API uploader (OAuth, quota-aware, never uploads QC-failed output).
6. **Phase 7:** benchmarks on target hardware; GUI.

## 6. Licences — verify before distributing

I could not fetch authoritative licence text in the build environment. Treat the register in `DEPENDENCIES.md` as *my understanding*, and confirm each entry at its source before redistribution. Places to check: the [moderngl PyPI page](https://pypi.org/project/moderngl/), the [FFmpeg build provider page](https://www.gyan.dev/ffmpeg/builds/) (builds that include libx264 are GPL), and the [Ultra Fractal perturbation notes](https://ultrafractal.com/help/formulas/perturbationcalculations.html) for the technique.
