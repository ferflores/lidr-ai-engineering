"""Cliente HTTP del servicio IA para la interfaz (Streamlit).

En la sesión 04 el formulario llamaba al servicio LLM en el mismo proceso. Con sesiones eso ya no vale:
el historial y la memoria viven en el diccionario del proceso de la API, así que el cliente tiene que
hablar con la API por HTTP para que `POST /sessions` y `POST /sessions/{id}/estimate` compartan estado.
Es la arquitectura real (frontend ↔ servicio IA) y lo que hace que el `session_id` signifique algo.

`make_http_client` está separado para que los tests lo sustituyan por un `TestClient` de FastAPI y
ejerciten la interfaz contra la API real en el mismo proceso, sin levantar un servidor.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 180.0  # un turno con adjuntos largos puede tardar bastante más que una estimación suelta

Attachment = tuple[str, bytes, str | None]  # (nombre, contenido, content-type)


class EstimatorAPIError(Exception):
    """La API ha respondido con un error HTTP, o no ha respondido."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def make_http_client(base_url: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=timeout)


def format_detail(detail: object) -> str:
    """El campo `detail` de un error de FastAPI como texto legible (los 422 llegan como lista de errores)."""
    if isinstance(detail, list):
        lines = []
        for error in detail:
            if isinstance(error, dict):
                loc = ".".join(str(p) for p in error.get("loc", ()) if p != "body")
                lines.append(f"{loc}: {error.get('msg')}" if loc else str(error.get("msg")))
            else:
                lines.append(str(error))
        return "\n".join(lines)
    return str(detail)


class EstimatorAPI:
    """Las operaciones que necesita la interfaz, una por endpoint."""

    def __init__(self, base_url: str | None = None, timeout: float = DEFAULT_TIMEOUT_SECONDS):
        self.base_url = (base_url or os.environ.get("API_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._client = make_http_client(self.base_url, timeout)

    # --- Sistema ---
    def health(self) -> dict:
        return self._request("GET", "/health")

    def prompts(self) -> dict:
        return self._request("GET", "/api/v1/prompts")

    # --- Sesiones ---
    def create_session(self) -> dict:
        return self._request("POST", "/api/v1/sessions")

    def get_session(self, session_id: str) -> dict:
        return self._request("GET", f"/api/v1/sessions/{session_id}")

    def delete_session(self, session_id: str) -> None:
        self._request("DELETE", f"/api/v1/sessions/{session_id}")

    def estimate(
        self,
        session_id: str,
        transcript: str,
        params: dict[str, object] | None = None,
        attachments: Sequence[Attachment] = (),
    ) -> dict:
        """Un turno: transcripción + parámetros tipados + adjuntos, en multipart/form-data."""
        data = {"transcript": transcript}
        for name, value in (params or {}).items():
            if value is not None and value != "":
                data[name] = str(value)
        files = [("attachments", (name, content, content_type or "application/octet-stream")) for name, content, content_type in attachments]
        return self._request("POST", f"/api/v1/sessions/{session_id}/estimate", data=data, files=files or None)

    # --- Interno ---
    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise EstimatorAPIError(
                f"No se ha podido conectar con la API en {self.base_url} ({exc.__class__.__name__}). "
                "¿Está arrancada? `uv run uvicorn app.main:app --reload`"
            ) from exc
        if response.status_code >= 400:
            try:
                detail = format_detail(response.json().get("detail"))
            except ValueError:
                detail = response.text
            raise EstimatorAPIError(f"HTTP {response.status_code}: {detail}", status_code=response.status_code)
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()
