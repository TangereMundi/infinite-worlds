# Setup on Windows 11

1. **Python 3.11 or newer** (python.org, tick "Add python.exe to PATH").
2. **FFmpeg**: download a Windows build that includes `libx264` and `aac` (and `h264_nvenc` if you want GPU encoding), unzip, and add its `bin` folder to PATH. Check with `ffmpeg -version` and `ffprobe -version` in a new terminal.
3. **Install the app** (offline after this step):
   ```
   cd infinite-worlds
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. **Check your hardware is seen correctly:**
   ```
   python -m infinite_worlds hw
   ```
   You should see your CPU count, RAM, GPU name/VRAM (via `nvidia-smi`), and whether NVENC works.

## First render
```
python -m infinite_worlds render "C:\Music\mytrack.wav" --quality preview
```
The video lands in `output\`, next to a `.iw.json` record (seeds, measured audio features, plan, QC, timings, peak memory).

* WAV/FLAC sources are muxed **bit-exact into .mkv**. MP3/AAC/M4A sources are muxed bit-exact into .mp4.
* Want .mp4 for everything? Add `--audio-policy compat` (re-encodes lossless audio to AAC 320k; flagged as lossy in the record).
* A second generation of the same track: `--gen g2`. Same settings + same generation id reproduces the same choices.

## Whole library, unattended
```
python -m infinite_worlds scan --input-dir "C:\Music"      # add new files to the queue
python -m infinite_worlds run --quality standard_1080p       # process the queue
python -m infinite_worlds run --watch                        # keep watching the input folder
python -m infinite_worlds status
```
Close the window or reboot at any time: the next `run` resumes where it stopped (finished 10-second chunks are kept).

## Quality presets
`preview` (640×360), `standard_1080p`, `high_1080p`, `experimental_4k`.
**Important:** the current CPU renderer draws internally at a fraction of the delivered size and upscales; it is a reference implementation, not a fast one. Do not expect real-time speed. Measure with a short track first (`render` prints render time and peak memory).
