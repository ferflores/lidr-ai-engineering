# Imagen oficial de uv con Python 3.11 (la misma versión fijada en .python-version)
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# UV_COMPILE_BYTECODE: precompila los .pyc para arrancar más rápido.
# UV_LINK_MODE=copy: copia los paquetes desde la caché en vez de enlazarlos (imagen autocontenida).
# UV_PYTHON_DOWNLOADS=never: usa el Python de la imagen, no descarga otro.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Solo las dependencias, para que esta capa se cachee mientras no cambie uv.lock
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2) El código de la aplicación (el .env NO se copia: ver .dockerignore; llega por docker compose)
COPY app ./app

# Usuario sin privilegios
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
