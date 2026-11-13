"""Separate audio in an isolated subprocess so the UI can cancel safely."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .core import MODELS, STEMS, app_data, validate_audio, verify_stems


def event(kind: str, **data) -> None:
    print("SF_EVENT:" + json.dumps({"type": kind, **data}, ensure_ascii=False), flush=True)


def run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    args = parser.parse_args()
    try:
        from audio_separator.separator import Separator
        import torch
        source = validate_audio(Path(args.input))
        job = Path(args.job).resolve()
        job.mkdir(parents=True, exist_ok=True)
        event("status", message="CUDA disponível" if torch.cuda.is_available() else "Modo CPU (mais lento)")
        event("status", message="Carregando modelo; o primeiro uso pode exigir download...")
        separator = Separator(output_dir=str(job), model_file_dir=str(app_data() / "models"),
                              output_format="WAV", sample_rate=44100, use_soundfile=True,
                              demucs_params={"segment_size": 7, "shifts": 1,
                                             "overlap": 0.25, "segments_enabled": True})
        separator.load_model(model_filename=args.model)
        event("status", message="Separando quatro stems...")
        separator.separate(str(source), {s.capitalize(): s for s in STEMS})
        result = verify_stems(job)
        event("result", stems={k: str(v) for k, v in result.items()})
        return 0
    except Exception as exc:
        event("error", message=str(exc))
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run())
