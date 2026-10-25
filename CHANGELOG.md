# Changelog

## 0.2.0 — "classic zoom" look (requested after reviewing the Maths Town Mandelbrot zoom playlist)
* **New default style `classic`** (`--style classic|moody`): saturated colour bands that cycle smoothly with log(iterations), black interior, no haze or relief lighting. The previous lit, hazy look remains as `moody`.
* **Zoom cycle:** scenes alternate **zoom in → slow fade → different place and palette → zoom out → fade → zoom in …**. A zoom-out scene travels the same discovered path backwards, deep → wide.
* **Slower, longer pacing for `classic`:** scenes of roughly 40–60 s (never under 30 s), 6–10 s fades, zoom speed about 0.2–0.34 octaves per second scaled by the music's energy.
* Depth is still limited by float64 (about 30 octaves, ≈ 1e9×). Zooms to 1e50+ like the long reference videos need perturbation rendering (planned with the GPU backend).
* New tests for scene length, in/out alternation, path reversal, distinct places and colour character.
* **Path validation:** before accepting a scene the director probes the whole camera path (9 low-res views) and rejects scenes that pass through solid-black or featureless views. This fixed a real case (a black stretch inside a crossfade between two Julia scenes) that QC had caught.
* **Fixed:** the plan and QC trusted the file header's duration. A stream-copied excerpt whose header still claimed the original length crashed the planner; the decoded length is now authoritative (with a log warning).

## 0.1.1 — fixes from the first full test run
* **Fixed:** a featureless "flat colour" stretch could appear deep in a zoom when discovery picked a bare region; the colour stage then amplified noise, causing a luminance jump that QC caught (`luminance flicker spike 0.098` on the 236 s track).
  * Discovery now also tries well-known boundary regions, retries when the best target has little structure, and makes the scene shallower if none is found.
  * Colour normalisation has floors, so an empty frame can no longer be amplified.
  * The director treats bare targets as a last resort.
  * New QC check: share of featureless frames.
* **Fixed:** MP3-in-MP4 audio check compared decoded samples, which differ at the first ~25 ms because of encoder-delay signalling even though the packets are identical. The check now accepts identical packets *or* identical decoded samples.
* **Added:** absolute `energy_level` so a quiet piece and a driving piece get different overall visual energy (the old curve only showed where music builds). Analysis version bumped to 2.
* **Fixed:** unreadable input is reported as "not a readable audio file"; unclosed pipes/DB handles.
* Test for interrupted-render resume now checks the first attempt explicitly.

## 0.1.0 — first prototype (Phases 1–3 partial, 5 partial)
Audio analysis, SeedEngine, Mandelbrot/Julia engine (CPU) with autonomous discovery, VisualDirector, chunked resumable renderer, lossless audio mux, quality control, SQLite queue, CLI, watermark, acceptance tests, docs.
