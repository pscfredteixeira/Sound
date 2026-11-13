# SampleForge AI — MVP 0.1

Desktop app for **real, local four-stem separation** (Drums, Bass, Vocals, Other). Source code lives in this repository; the GitHub Pages site cannot run local PyTorch inference.

## What works in Stage 1

- Import/drag WAV, MP3, FLAC, play/pause/stop and view waveform.
- Separate four stems with the **real** `audio-separator` Python API and `htdemucs.yaml` or `htdemucs_ft.yaml` models.
- Model files are downloaded **on first use** to `%LOCALAPPDATA%\SampleForgeAI\models`; no audio is uploaded to any service.
- Background worker process, logs, cancel button and basic error handling.
- Listen to each stem and export 44.1/48 kHz 16/24-bit WAV files plus JSON manifest.
- Original music is never overwritten. Work-in-progress stems are in `%LOCALAPPDATA%\SampleForgeAI\jobs`.

**Not yet implemented:** automatic BPM/key, transient detection, one-shots, loops, sample classification, sample-pack generator, waveform editing, installer. These are planned for stages 2–4, not simulated in this MVP.

## Windows 11 installation (Python 3.11 x64)

1. Install Python 3.11 (64-bit) and FFmpeg. In PowerShell: `winget install Gyan.FFmpeg`; restart the terminal, then verify `ffmpeg -version`.
2. Download/clone the repository and open PowerShell in the project directory.
3. Run `powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1` (or follow the manual commands below).
4. Launch: `.\.venv\Scripts\python.exe -m sampleforge`
5. Import a **short, authorized** song, click **Separate 4 Stems**, wait for the model download and processing, preview the four stems, then **Export WAV Stems**.

Manual setup:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m sampleforge
```

Default installation uses CPU (slower). For NVIDIA CUDA, use the PyTorch [official install selector](https://pytorch.org/get-started/locally/) for your driver, then install `audio-separator[gpu]` inside the same virtual environment. Check `python -c "import torch; print(torch.cuda.is_available())"` with the venv Python. CUDA is **not** guaranteed by merely having an NVIDIA GPU. Demucs uses 7-second segments and 1 shift to limit VRAM usage. In **Advanced → Device**, choose **Force CPU (low VRAM)** to process locally without CUDA if your GPU runs out of memory; this does not require switching environments.

## Architecture

```text
sampleforge/
  __main__.py   # application entry point
  gui.py        # PySide6 player, waveform, stems, async UI
  worker.py     # isolated AI process, killable
  core.py       # audio validation, waveform decoding, export and manifest
scripts/setup_windows.ps1
requirements.txt
tests/test_core.py
.github/workflows/test.yml
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
```

Tests generate synthetic WAV audio and verify waveform reading, four-stem validation, 48 kHz/24-bit export and JSON. **End-to-end model inference on a real song must be tested on a Windows machine with the dependencies installed; GitHub CI does not download model weights or assert separation quality.**

## Technical limits, licenses and privacy

- Four-stem separation is imperfect; cross-bleed and artifacts are expected. Only models actually supporting four stems are selectable.
- `audio-separator` is MIT-licensed; underlying model weights and bundled dependencies may have **separate** license/usage conditions. Review them before redistribution of weights or commercial packaging. Credits: [Audio Separator](https://github.com/nomadkaraoke/python-audio-separator), [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui), [Demucs](https://github.com/facebookresearch/demucs).
- This program makes no audio uploads, login requests or cloud inference. The **first model download** does require an internet connection; afterward model inference is local.
- Export files are new files. Intermediate jobs are kept until manually removed. You are responsible for using audio you own or are licensed to process.
- FFmpeg and its build may have LGPL/GPL obligations depending on distribution; consult its licensing before bundling binaries.

## Sample Recovery & Sample Playground

The future **Sample Recovery** mode will help reconstruct usable one-shots and phrases from previously finished tracks, with clear confidence limits. A future **Sample Playground** will provide a real step sequencer and multitrack timeline to arrange recovered audio, preview patterns and export WAV mixdowns. Neither feature is simulated as functional in Stage 1.

See [the Portuguese implementation roadmap](docs/ROADMAP_RECOVERY_PLAYGROUND.md) for requirements and acceptance criteria.

## Roadmap

- Stage 2: beat/transient analysis, slicer, sample preview/edit and loop exports.
- Stage 3: evidence-based classification, ranking, presets and automatic sample-pack generation.
- Stage 4: installer, deeper testing, diagnostics and performance optimization.
