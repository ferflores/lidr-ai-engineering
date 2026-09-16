#!/usr/bin/env bash
# Genera (o regenera) transcripciones/*.json a partir de cada transcripciones/*.txt,
# listos para `curl -d @transcripciones/<nombre>.json`.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python - <<'PY'
import json
from pathlib import Path

for txt in sorted(Path("transcripciones").glob("*.txt")):
    body = {"transcription": txt.read_text(encoding="utf-8").strip()}
    out = txt.with_suffix(".json")
    out.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{out}  ({len(body['transcription'])} caracteres)")
PY
