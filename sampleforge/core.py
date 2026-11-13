"""Pure helpers for SampleForge's audio IO, metadata and export."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

STEMS = ("drums", "bass", "vocals", "other")
MODELS = ("htdemucs.yaml", "htdemucs_ft.yaml")
EXTENSIONS = {".wav", ".mp3", ".flac"}


def app_data() -> Path:
    base = Path(os.getenv("LOCALAPPDATA") or (Path.home() / ".local" / "share"))
    result = base / "SampleForgeAI"
    result.mkdir(parents=True, exist_ok=True)
    return result


def check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg não encontrado no PATH. Instale FFmpeg e reinicie o programa.")


def validate_audio(path: Path) -> Path:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
        raise ValueError("Selecione um arquivo WAV, MP3 ou FLAC.")
    if path.stat().st_size == 0:
        raise ValueError("O arquivo de áudio está vazio.")
    return path


def waveform_peaks(path: Path, points: int = 1400) -> np.ndarray:
    """Decode to low-rate mono in a worker thread; return normalized display peaks."""
    check_ffmpeg()
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
               "-map", "0:a:0", "-ac", "1", "-ar", "2000", "-f", "f32le", "pipe:1"]
    proc = subprocess.run(command, capture_output=True, timeout=240)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-900:])
    audio = np.frombuffer(proc.stdout, dtype="<f4")
    if audio.size == 0:
        raise ValueError("Nenhum áudio foi decodificado.")
    edges = np.linspace(0, audio.size, points + 1, dtype=int)
    peaks = np.zeros(points, dtype=np.float32)
    for i in range(points):
        if edges[i] < edges[i + 1]:
            peaks[i] = float(np.max(np.abs(audio[edges[i]:edges[i + 1]])))
    return np.clip(peaks, 0, 1)


def verify_stems(job: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for stem in STEMS:
        path = job / f"{stem}.wav"
        if not path.is_file() or path.stat().st_size < 100:
            raise RuntimeError(f"Stem ausente ou vazio: {stem}.wav")
        info = sf.info(path)
        if info.frames == 0 or info.channels < 1:
            raise RuntimeError(f"Stem inválido: {stem}.wav")
        result[stem] = path
    return result


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")[:70] or "track"


def export_stems(stems: dict[str, Path], source: Path, model: str, destination: Path,
                 sample_rate: int, bit_depth: int, cancel: threading.Event,
                 progress=None) -> Path:
    """Create a new pack transactionally; never overwrite user originals."""
    check_ffmpeg()
    if sample_rate not in (44100, 48000) or bit_depth not in (16, 24):
        raise ValueError("Formato WAV inválido.")
    if set(stems) != set(STEMS):
        raise ValueError("É necessário ter os quatro stems antes da exportação.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = f"SampleForge_{safe_name(source.stem)}_{stamp}_{uuid.uuid4().hex[:5]}"
    destination.mkdir(parents=True, exist_ok=True)
    partial = destination / ("." + base + ".partial")
    final = destination / base
    partial.mkdir(parents=False, exist_ok=False)
    active = None
    try:
        out_dir = partial / "Stems"
        out_dir.mkdir()
        records = []
        for i, stem in enumerate(STEMS, 1):
            if cancel.is_set():
                raise InterruptedError("Exportação cancelada.")
            name = f"SF_{stem.capitalize()}_{i:02d}.wav"
            output = out_dir / name
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-i", str(stems[stem]), "-map", "0:a:0", "-ar", str(sample_rate),
                   "-c:a", f"pcm_s{bit_depth}le", str(output)]
            active = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            while active.poll() is None:
                if cancel.wait(0.1):
                    active.kill()
                    active.communicate()
                    raise InterruptedError("Exportação cancelada.")
            _, stderr = active.communicate()
            if active.returncode:
                raise RuntimeError(f"FFmpeg ({stem}): {stderr.decode('utf-8', 'replace')[-600:]}")
            info = sf.info(output)
            records.append({"name": name, "category": stem, "path": f"Stems/{name}",
                            "duration_seconds": round(info.duration, 4), "sample_rate": info.samplerate,
                            "bit_depth": bit_depth, "source": source.name, "start_seconds": 0,
                            "end_seconds": round(info.duration, 4), "bpm": None, "key": None})
            if progress:
                progress(i, len(STEMS))
        metadata = {"app": "SampleForge AI", "schema_version": 1,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "source": source.name, "model": model, "samples": records,
                    "notice": "Stem separation is imperfect; audible artifacts may remain."}
        (partial / "manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        if cancel.is_set():
            raise InterruptedError("Exportação cancelada.")
        partial.rename(final)
        return final
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
