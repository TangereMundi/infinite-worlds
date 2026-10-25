# Dependency and asset register

Licences below are **my understanding, not verified against upstream text in this environment**.
Confirm each at its source before redistributing or using commercially.

| Component | Version tested | Licence (to verify) | Source | Use | Redistribution / commercial notes |
|---|---|---|---|---|---|
| Python | 3.13 (3.11+ intended) | PSF | python.org | runtime | Free. |
| numpy | 2.5.x | BSD-3 | numpy.org | all numerics | Free. |
| scipy | 1.18.x | BSD-3 | scipy.org | filters, interpolation | Free. |
| Pillow | 12.x | HPND | python-pillow.org | watermark PNG loading; tests | Free. |
| psutil | 7.x (optional) | BSD-3 | github.com/giampaolo/psutil | RAM/CPU detection, peak-memory sampling | Optional; the app runs without it (fewer metrics). |
| FFmpeg / ffprobe | system build with libx264, aac, h264_nvenc | LGPL or GPL **depending on the build** | ffmpeg.org; Windows builds e.g. gyan.dev | decode, loudness, encode, mux | Builds containing libx264 are GPL. Using it as a separate program is normally fine for personal use; if you ever *redistribute* the application with FFmpeg, GPL terms apply. |
| x264 (inside FFmpeg) | — | GPL | videolan.org | H.264 encoding | As above. |
| NVENC (via FFmpeg) | — | NVIDIA driver licence | nvidia.com | optional hardware encode | Requires a supported NVIDIA GPU + driver. |
| moderngl *(planned, not yet used)* | — | MIT (to verify) | pypi.org/project/moderngl | GPU shader backend | — |
| SQLite (stdlib) | — | public domain | — | job queue | Free. |

## Not used (deliberately)
* No code, assets, or settings from BiepBop, Vortexia, Adobe products, or any other proprietary tool.
* No online services, AI APIs or subscriptions.
* Fractal formulas (Mandelbrot, Julia, smooth iteration count, cosine palettes) are standard published mathematics, implemented from scratch.

## Visual assets
None bundled. The CC0 tunnel/fractal library is not yet indexed (Phase 4). When it is, each asset's source URL and licence text will be stored with its hash; a "CC0" label will be treated as a claim to record, not a guarantee about every element inside the footage.

## Test material
`tests/make_test_audio.py` synthesises its own audio (sines, noise, simple drum shapes). No third-party music is shipped.
