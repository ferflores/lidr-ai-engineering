"""Comprobación del CI: el chunker funciona dentro del contenedor, sin red ni API key.

Se ejecuta con `docker compose exec -T api python < scripts/ci_check_chunker.py` (por stdin, para que la
indentación del script no dependa del bloque YAML del workflow). Solo trocea: no llama a la API de embeddings.
"""

import json
import sys
from pathlib import Path

# Dentro del contenedor el proyecto vive en /app; en local, en la raíz del repo del proyecto. El script llega
# por stdin, así que no puede deducir su propia ruta: se prueban las dos.
ROOT = Path("/app") if Path("/app/data/budgets_sample.json").exists() else Path.cwd()
sys.path.insert(0, str(ROOT))

from app.embedding_pipeline.chunker import JSONStructuralChunker  # noqa: E402
from app.embedding_pipeline.schemas import Budget  # noqa: E402

budgets = [Budget.model_validate(b) for b in json.loads((ROOT / "data/budgets_sample.json").read_text())]
chunks = JSONStructuralChunker().chunk(budgets)

assert len(budgets) == 15, f"se esperaban 15 presupuestos, hay {len(budgets)}"
assert len(chunks) == sum(len(b.components) for b in budgets) == 73, f"se esperaban 73 chunks, hay {len(chunks)}"
assert chunks[0].chunk_id == "BUD-2024-014::AUTH-001", chunks[0].chunk_id
assert chunks[0].text.startswith("[Project: "), "falta la cabecera de contexto del presupuesto padre"
assert chunks[0].token_count > 0, "tiktoken no contó tokens (¿falta el tokenizador en la imagen?)"
assert chunks[0].metadata["client_sector"] == "finance", chunks[0].metadata

print(f"{len(budgets)} presupuestos -> {len(chunks)} chunks; el más largo: "
      f"{max(chunks, key=lambda c: c.token_count).chunk_id} ({max(c.token_count for c in chunks)} tokens)")
