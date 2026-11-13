"""Real SampleForge library generation from separated stems.

The drum sub-stem step uses DrumSep's *DSP*, not instrument-perfect AI.
Every generated WAV contains real audio; there are no simulated samples.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .core import STEMS, safe_name, verify_stems

DRUM_FOLDERS = {
    "kick": "Drums/Kicks",
    "snare": "Drums/Snares",
    "hihat": "Drums/HiHats",
    "cymbals": "Drums/Cymbals",
    "toms": "Drums/Toms",
}

ONESHOT_LIMITS = {"kick": .9, "snare": .9, "hihat": .6, "cymbals": 2.0, "toms": 1.3}
Record = dict[str, object]


def _mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim == 2 else audio


def _read(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr < 8000 or len(data) == 0:
        raise ValueError(f"Áudio inválido ou vazio: {path}")
    return data, sr


def _fade_out(audio: np.ndarray, sr: int, ms: float = 9) -> np.ndarray:
    """Apply tail fade only; never blur the attack."""
    audio = audio.copy()
    n = min(len(audio), max(1, int(sr * ms / 1000)))
    if n:
        audio[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)[:, None]
    return audio


def _write_clip(destination: Path, audio: np.ndarray, sr: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, np.clip(audio, -1, 1), sr, subtype="PCM_24")


def estimate_tempo(stem_path: Path) -> tuple[float | None, float]:
    """Approximate BPM and beat anchor. A beat is NOT necessarily a downbeat."""
    import librosa

    y, sr = librosa.load(str(stem_path), sr=11025, mono=True, duration=150)
    if len(y) < sr * 2 or float(np.max(np.abs(y))) < .003:
        return None, 0.0
    tempo, frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    value = float(np.asarray(tempo).reshape(-1)[0])
    if not math.isfinite(value) or not 35 <= value <= 280:
        return None, 0.0
    # Report the algorithm's tempo without claiming a verified musical meter.
    beats = librosa.frames_to_time(frames, sr=sr)
    return round(value, 2), float(beats[0]) if len(beats) else 0.0


def _record(path: Path, root: Path, category: str, stem: str,
            start: float, duration: float, bpm: float | None, kind: str) -> Record:
    return {
        "name": path.name,
        "path": path.relative_to(root).as_posix(),
        "category": category,
        "source_stem": stem,
        "type": kind,
        "start_seconds": round(start, 5),
        "end_seconds": round(start + duration, 5),
        "duration_seconds": round(duration, 5),
        "bpm_estimated": bpm,
        "key_estimated": None,
        "confidence_note": "Separação e pontos de corte estimados; ouça antes de usar.",
    }


def detect_events(audio: np.ndarray, sr: int) -> np.ndarray:
    """Find backtracked transient onsets on actual audio, in sample indices."""
    import librosa

    mono = _mono(audio)
    hop = 256 if sr <= 48000 else 512
    strength = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    positions = librosa.onset.onset_detect(
        onset_envelope=strength, sr=sr, hop_length=hop,
        units="samples", backtrack=True,
    )
    positions = np.asarray(positions, dtype=int)
    if len(positions) == 0 and float(np.max(np.abs(mono))) > .006:
        positions = np.array([0], dtype=int)
    # At most one candidate within 60 ms, avoids duplicate attacks.
    spacing = int(.06 * sr)
    clean: list[int] = []
    for pos in positions:
        pos = max(0, min(int(pos), len(audio) - 1))
        if not clean or pos - clean[-1] >= spacing:
            clean.append(pos)
    return np.asarray(clean, dtype=int)


def _fingerprint(segment: np.ndarray) -> bytes:
    """Exact deduplication on quantized content only; not misleading timbre matching."""
    return hashlib.blake2b(segment.tobytes(), digest_size=12).digest()


def cut_events(path: Path, root: Path, folder: str, stem: str,
               *, max_duration: float, max_count: int = 40,
               min_duration: float = .065) -> list[Record]:
    """Produce separate playable WAVs from real transients."""
    audio, sr = _read(path)
    positions = detect_events(audio, sr)
    candidates: list[tuple[float, int, np.ndarray]] = []
    for i, position in enumerate(positions):
        start = max(0, int(position) - int(sr * .009))
        boundary = int(positions[i + 1]) - int(sr * .008) if i + 1 < len(positions) else len(audio)
        end = min(len(audio), boundary, start + int(sr * max_duration))
        if end - start < int(sr * min_duration):
            continue
        segment = audio[start:end]
        peak = float(np.max(np.abs(segment)))
        rms = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))
        if peak < .012 or rms < .003:
            continue
        # Attacks are preserved; only tails are faded. Rank by usable energy.
        score = rms * np.sqrt((end - start) / sr)
        candidates.append((float(score), start, segment))
    candidates.sort(key=lambda row: row[0], reverse=True)
    records: list[Record] = []
    seen: set[bytes] = set()
    for _, start, segment in candidates:
        if len(records) >= max_count:
            break
        key = _fingerprint(segment)
        if key in seen:
            continue
        seen.add(key)
        number = len(records) + 1
        name = f"SF_{stem}_{number:03d}.wav"
        output = root / folder / name
        _write_clip(output, _fade_out(segment, sr), sr)
        records.append(_record(output, root, stem, stem, start / sr,
                               len(segment) / sr, None, "one_shot"))
    return records


def cut_loops(path: Path, root: Path, folder: str, stem: str,
              bpm: float | None, anchor: float,
              bar_options: tuple[int, ...] = (1, 2, 4, 8, 16),
              per_size: int = 2) -> list[Record]:
    """Cut beat-quantized *candidates*. Meter/downbeat accuracy is not guaranteed."""
    if bpm is None:
        return []
    audio, sr = _read(path)
    bar_samples = sr * 240.0 / bpm  # 4 beats at estimated BPM
    if not 0 < bar_samples <= len(audio):
        return []
    records: list[Record] = []
    for bars in bar_options:
        n = round(bar_samples * bars)
        if n < 64 or n > len(audio):
            continue
        starts: list[tuple[float, int]] = []
        initial = max(0, int(anchor * sr))
        step = max(1, round(bar_samples))
        for start in range(initial, len(audio) - n + 1, step):
            clip = audio[start:start + n]
            rms = float(np.sqrt(np.mean(clip.astype(np.float64) ** 2)))
            if rms < .008:
                continue
            # Prefer audible, non-clipped sections; candidates can be auditioned.
            overload = float(np.mean(np.abs(clip) > .999))
            starts.append((rms * (1.0 - overload), start))
        starts.sort(reverse=True)
        accepted: list[int] = []
        for _, start in starts:
            if len(accepted) >= per_size:
                break
            if any(abs(start - old) < n for old in accepted):
                continue
            accepted.append(start)
            number = len(accepted)
            output = root / folder / f"SF_{stem}_{bars}Bars_{bpm:.0f}BPM_{number:02d}.wav"
            _write_clip(output, _fade_out(audio[start:start + n], sr, ms=4), sr)
            records.append(_record(output, root, f"{stem} loop", stem, start / sr,
                                   n / sr, bpm, "loop"))
    return records


def build_library(source: Path, job: Path, destination: Path,
                  status: Callable[[str], None] | None = None) -> tuple[Path, list[Record]]:
    """Build a pack atomically from an existing four-stem separation.

    The calling GUI runs this in a killable child process. Failed jobs clean up.
    """
    from drumsep import DrumSeparator

    def announce(message: str) -> None:
        if status:
            status(message)

    stems = verify_stems(job)
    destination.mkdir(parents=True, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pack_name = f"SampleForge_{safe_name(source.stem)}_{tag}_{uuid.uuid4().hex[:6]}"
    temporary = destination / ("." + pack_name + ".partial")
    finished = destination / pack_name
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        announce("Separando kick, snare, hats, pratos e toms por análise espectral...")
        substem_folder = temporary / "Drum_Substems"
        separator = DrumSeparator(enhanced=True)
        result = separator.separate(str(stems["drums"]), str(substem_folder),
                                    bass_path=str(stems["bass"]))
        files = {name: Path(filename) for name, filename in result.stems.items()}
        if not set(DRUM_FOLDERS).issubset(files):
            raise RuntimeError("O separador não retornou os cinco elementos de bateria.")
        announce("Estimando BPM e pontos de início dos eventos...")
        bpm, beat_anchor = estimate_tempo(stems["drums"])
        records: list[Record] = []
        for name, folder in DRUM_FOLDERS.items():
            announce(f"Recortando {name} em WAVs individuais...")
            records.extend(cut_events(files[name], temporary, folder, name,
                                      max_duration=ONESHOT_LIMITS[name]))
        announce("Recortando eventos e frases musicais...")
        records.extend(cut_events(stems["bass"], temporary, "Bass/Events", "bass",
                                  max_duration=1.8, max_count=36, min_duration=.12))
        records.extend(cut_events(stems["vocals"], temporary, "Vocals/Chops", "vocal",
                                  max_duration=2.8, max_count=36, min_duration=.15))
        for name, folder in (
            ("drums", "Loops/Drums"), ("bass", "Loops/Bass"),
            ("vocals", "Loops/Vocals"), ("other", "Unclassified/Loops"),
        ):
            announce(f"Recortando loops: {name}...")
            records.extend(cut_loops(stems[name], temporary, folder, name, bpm, beat_anchor))
        if not records:
            raise RuntimeError("Nenhum recorte aproveitável foi encontrado nesta faixa.")
        manifest = {
            "app": "SampleForge AI",
            "schema_version": 2,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_name": source.name,
            "bpm_estimated": bpm,
            "beat_grid_note": "Grade aproximada. O primeiro beat não é necessariamente o primeiro tempo do compasso.",
            "method": "Demucs 4-stem + DrumSep DSP 5-substem + onset slicing",
            "limitations": [
                "Substems estimados via máscaras espectrais, não isolamento perfeito por IA.",
                "One-shots podem conter interferências, compressão e reverberação da mix original.",
                "Loops seguem BPM aproximado e devem ser revisados antes de uso.",
                "A categoria 'Other' da separação não equivale a sintetizador isolado.",
            ],
            "samples": records,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.rename(finished)
        for record in records:
            record["absolute_path"] = str(finished / str(record["path"]))
        return finished, records
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
