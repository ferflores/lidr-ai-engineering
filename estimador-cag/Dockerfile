# Imagen oficial de uv con Python 3.11 (la misma versión fijada en .python-version)
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# UV_COMPILE_BYTECODE: precompila los .pyc para arrancar más rápido.
# UV_LINK_MODE=copy: copia los paquetes desde la caché en vez de enlazarlos (imagen autocontenida).
# UV_PYTHON_DOWNLOADS=never: usa el Python de la imagen, no descarga otro.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken

WORKDIR /app

# 1) Solo las dependencias, para que esta capa se cachee mientras no cambie uv.lock
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 1b) El tokenizador que usa el chunker de embeddings (tiktoken, cl100k_base) se descarga en el build y queda
#     en la imagen: el contenedor no necesita red para contar tokens ni falla si la descarga no está disponible.
RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.encoding_for_model('text-embedding-3-small')"

# 2) El código: API (app/), plantillas de prompt (prompts/), interfaz (streamlit_app.py + .streamlit/config.toml),
#    scripts/ y data/ (pre-work sesión 07)
#    y las transcripciones y adjuntos de ejemplo que carga la interfaz.
#    El .env NO se copia (ver .dockerignore): llega como variables de entorno por docker compose.
COPY app ./app
COPY prompts ./prompts
COPY streamlit_app.py ./
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY transcripciones ./transcripciones
# Pre-work sesión 07: scripts/compare.py se ejecuta con `docker compose exec api ...` y data/ trae el sample de presupuestos.
COPY scripts ./scripts
COPY data ./data

# Usuario sin privilegios
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
