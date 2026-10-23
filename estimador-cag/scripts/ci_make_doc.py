#!/usr/bin/env python
"""Imprime el body de `POST /embeddings/ingest` para el primer presupuesto del corpus (lo usa el CI).

    docker compose exec -T api python < scripts/ci_make_doc.py > /tmp/doc.json

Vive en un fichero y no como `python -c "..."` dentro del workflow por la misma razón que
`ci_check_chunker.py`: un script multilínea embebido en un bloque `run: |` de YAML rompe el bloque en
cuanto una línea empieza en la columna 0. Por stdin, la indentación del script no depende del YAML.
"""

import json

SOURCE_PREFIX = "ci/"

with open("data/budgets_sample.json", encoding="utf-8") as handle:
    budget = json.load(handle)[0]

print(
    json.dumps(
        {
            "source_path": f"{SOURCE_PREFIX}{budget['budget_id']}.json",
            "document_type": "historical_budget",
            "content": budget,
        }
    )
)
