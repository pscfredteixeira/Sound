import json
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sampleforge.core import STEMS, export_stems, safe_name, validate_audio, verify_stems, waveform_peaks


def test_safe_name():
    assert safe_name('A Música / Teste') == 'A_M_sica_Teste'


def test_import_and_waveform(tmp_path):
    path = tmp_path / 'test.wav'
    sf.write(path, np.sin(np.linspace(0, 200, 44100)), 44100)
    assert validate_audio(path) == path.resolve()
    peaks = waveform_peaks(path, 128)
    assert len(peaks) == 128
    assert np.max(peaks) > 0.5


def test_verify_and_export(tmp_path):
    job = tmp_path / 'job'
    job.mkdir()
    t = np.arange(44100) / 44100
    for i, stem in enumerate(STEMS):
        sf.write(job / f'{stem}.wav', 0.1 * np.sin(2 * np.pi * (220 + i * 100) * t), 44100)
    stems = verify_stems(job)
    exported = export_stems(stems, job / 'track.wav', 'htdemucs.yaml', tmp_path / 'exports',
                            48000, 24, threading.Event())
    manifest = json.loads((exported / 'manifest.json').read_text(encoding='utf-8'))
    assert len(manifest['samples']) == 4
    assert all(sf.info(exported / 'Stems' / x['name']).samplerate == 48000
               for x in manifest['samples'])
    assert all(sf.info(exported / 'Stems' / x['name']).subtype == 'PCM_24'
               for x in manifest['samples'])


def test_cancel_export(tmp_path):
    with pytest.raises(ValueError):
        export_stems({}, tmp_path / 'a.wav', 'htdemucs.yaml', tmp_path,
                     44100, 16, threading.Event())
