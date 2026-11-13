"""Synthetic audio tests: verify that generated samples are actual WAV audio."""
import json

import numpy as np
import soundfile as sf

from sampleforge.library import cut_events, cut_loops, detect_events


def drum_pattern(sr=11025, seconds=5):
    samples = np.zeros((sr * seconds, 2), dtype=np.float32)
    for second in range(seconds):
        for beat in (0.0, 0.5):
            offset = int((second + beat) * sr)
            n = min(900, len(samples) - offset)
            attack = np.sin(2 * np.pi * 90 * np.arange(n) / sr)
            envelope = np.exp(-np.arange(n) / (sr * .04))
            samples[offset:offset + n] += (.65 * attack * envelope)[:, None]
    return samples


def test_find_actual_transient_events():
    x = drum_pattern()
    positions = detect_events(x, 11025)
    assert len(positions) >= 4
    assert all(np.diff(positions) > 0)


def test_export_individual_wavs(tmp_path):
    original = tmp_path / "kick.wav"
    sf.write(original, drum_pattern(), 11025)
    records = cut_events(original, tmp_path / "pack", "Drums/Kicks", "kick",
                         max_duration=.45, max_count=12)
    assert len(records) >= 4
    for record in records:
        dest = tmp_path / "pack" / record["path"]
        assert dest.is_file()
        data, sr = sf.read(dest)
        assert sr == 11025
        assert data.ndim == 2
        assert float(np.max(np.abs(data))) > .01


def test_cut_real_loops_without_fake_bpm(tmp_path):
    path = tmp_path / "bass.wav"
    sf.write(path, drum_pattern(seconds=12), 11025)
    root = tmp_path / "pack"
    assert cut_loops(path, root, "Loops/Bass", "bass", None, 0) == []
    records = cut_loops(path, root, "Loops/Bass", "bass", 120.0, 0,
                        bar_options=(1, 2), per_size=1)
    assert len(records) == 2
    for record in records:
        data, sr = sf.read(root / record["path"])
        assert data.size > 0
        assert sr == 11025
        assert record["bpm_estimated"] == 120.0
