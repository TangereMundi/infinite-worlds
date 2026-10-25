# Infinite Worlds

Local-first engine that turns a folder of music into music-synchronised videos, with no cloud, no subscription and no per-track manual work. See `docs/TECHNICAL_PLAN.md` for exactly what is built and tested, and `docs/SETUP_WINDOWS.md` to run it.

```
python -m infinite_worlds render track.wav --quality preview
python -m infinite_worlds scan --input-dir MUSIC_FOLDER && python -m infinite_worlds run
```

Package layout (one module per spec component): `audio_analyzer`, `seed_engine`, `engines/` (registry + `fractal`), `visual_director`, `render_manager` (RenderManager + AudioVideoMuxer), `quality`, `job_queue`, `pipeline`, `config`, `hardware`, `cli`.

New visual engines subclass `engines.base.VisualEngine` (three methods: `sample_scene`, `render`, `fingerprint`) and are registered in `default_registry()`. An engine must be a pure function of (scene, time, controls) — that is what makes resume and parallel rendering safe.

Tests: `python -m unittest discover -s tests -v`
