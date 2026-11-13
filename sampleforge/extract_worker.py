"""Isolated process: turn a four-stem job into an audible sample library."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .core import validate_audio
from .library import build_library


def announce(kind: str, **data: object) -> None:
    print("SF_EVENT:" + json.dumps({"type": kind, **data}, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    try:
        source = validate_audio(Path(args.source))
        final, records = build_library(
            source, Path(args.job), Path(args.destination),
            status=lambda message: announce("status", message=message),
        )
        announce("result", folder=str(final),
                 samples=[{"path": rec["absolute_path"], "name": rec["name"],
                           "category": rec["category"], "kind": rec["type"]}
                          for rec in records])
        return 0
    except Exception as exc:
        announce("error", message=str(exc))
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
