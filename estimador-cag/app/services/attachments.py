"""Adjuntos: extracción local de texto (camino B) de PDF, Word y texto plano.

Dos caminos canónicos para que un LLM "lea" un documento:

- **A, multimodal directo**: subir el fichero al proveedor (Files API) y referenciarlo en el mensaje. Menos
  código, pero acoplado a un proveedor con soporte multimodal, con el fichero viajando (y facturándose)
  en cada turno y sin control sobre qué texto ve el modelo.
- **B, extracción local** (este módulo): sacar el texto aquí con `pypdf` y `python-docx` y concatenarlo al
  transcript. Funciona igual con OpenAI y Anthropic (el proyecto soporta ambos), el texto es inspeccionable
  y medible en tokens, se puede recortar y se prueba sin llamar a ningún LLM. Y es el primer paso del
  chunking de RAG: una vez el documento es texto, trocearlo e indexarlo es el siguiente módulo.

El texto extraído se concatena a la transcripción con separadores claros:

    <transcripción>

    --- attachment: especificacion.pdf ---
    <texto del documento>
    --- fin del adjunto: especificacion.pdf ---

Límites (configurables en `Settings`): número de ficheros, tamaño en bytes y caracteres extraídos por
fichero. Lo que no se puede leer (PDF escaneado sin texto, fichero corrupto, formato no soportado) se
reporta como error o aviso en vez de pasar en silencio al prompt.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

import pypdf
from docx import Document as load_docx
from docx.opc.exceptions import PackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "texto",
    ".md": "texto",
}
CONTENT_TYPE_KINDS: dict[str, str] = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "texto",
    "text/markdown": "texto",
}
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_CHARS = 20_000

_BLANK_LINES = re.compile(r"\n{3,}")


class AttachmentError(Exception):
    """Error base de los adjuntos."""


class UnsupportedAttachment(AttachmentError):
    """Formato no soportado (solo PDF, Word .docx y texto plano)."""


class AttachmentTooLarge(AttachmentError):
    """El fichero supera el tamaño máximo configurado."""


class AttachmentUnreadable(AttachmentError):
    """El fichero no se pudo abrir o decodificar (corrupto, cifrado, codificación desconocida)."""


@dataclass(frozen=True)
class ExtractedAttachment:
    """Texto extraído de un adjunto y lo que conviene saber de la extracción."""

    filename: str
    kind: str              # "pdf" | "docx" | "texto"
    text: str
    size_bytes: int
    pages: int | None      # solo PDF
    truncated: bool        # el texto superaba max_chars y se recortó
    warning: str | None    # p. ej. PDF sin texto extraíble (escaneado)

    @property
    def chars(self) -> int:
        return len(self.text)


# --- Detección y extracción ---------------------------------------------------------------


def detect_kind(filename: str, content_type: str | None = None) -> str:
    """Tipo de adjunto por extensión (o por content-type si no hay extensión reconocible)."""
    suffix = PurePosixPath(filename).suffix.lower()
    if suffix in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[suffix]
    if content_type and content_type.split(";")[0].strip() in CONTENT_TYPE_KINDS:
        return CONTENT_TYPE_KINDS[content_type.split(";")[0].strip()]
    hint = " Convierte el .doc a .docx antes de adjuntarlo." if suffix == ".doc" else ""
    raise UnsupportedAttachment(
        f"Adjunto '{filename}' no soportado: se aceptan {', '.join(SUPPORTED_EXTENSIONS)}.{hint}"
    )


def extract_text(
    filename: str,
    data: bytes,
    *,
    content_type: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> ExtractedAttachment:
    """Extrae el texto de un adjunto aplicando los límites de tamaño y de caracteres."""
    filename = PurePosixPath(filename.replace("\\", "/")).name or "adjunto"
    kind = detect_kind(filename, content_type)
    if len(data) > max_bytes:
        raise AttachmentTooLarge(
            f"Adjunto '{filename}' demasiado grande: {len(data):,} bytes (máximo {max_bytes:,})".replace(",", ".")
        )
    if not data:
        raise AttachmentUnreadable(f"Adjunto '{filename}' vacío (0 bytes)")

    pages: int | None = None
    if kind == "pdf":
        text, pages = _extract_pdf(filename, data)
    elif kind == "docx":
        text = _extract_docx(filename, data)
    else:
        text = _extract_plain_text(filename, data)

    text = _normalize(text)
    warning = None
    if not text:
        warning = (
            f"'{filename}' no contiene texto extraíble"
            + (" (¿PDF escaneado? haría falta OCR)" if kind == "pdf" else "")
        )
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars].rstrip() + f"\n[... adjunto truncado a {max_chars:,} caracteres ...]".replace(",", ".")
    return ExtractedAttachment(
        filename=filename, kind=kind, text=text, size_bytes=len(data), pages=pages, truncated=truncated, warning=warning
    )


def _extract_pdf(filename: str, data: bytes) -> tuple[str, int]:
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # contraseña desconocida o cifrado no soportado
                raise AttachmentUnreadable(f"El PDF '{filename}' está cifrado y no se puede leer") from exc
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except AttachmentError:
        raise
    except Exception as exc:  # pypdf lanza varias familias de errores con ficheros corruptos
        raise AttachmentUnreadable(f"No se ha podido leer el PDF '{filename}': {exc}") from exc
    return "\n\n".join(page for page in pages if page), len(pages)


def _extract_docx(filename: str, data: bytes) -> str:
    try:
        document = load_docx(io.BytesIO(data))
    except (PackageNotFoundError, zipfile.BadZipFile, KeyError, ValueError) as exc:  # un .docx es un zip con XML dentro
        raise AttachmentUnreadable(f"No se ha podido leer el documento Word '{filename}': {exc}") from exc
    parts: list[str] = []
    # iter_inner_content() recorre párrafos y tablas en el orden del documento
    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            if block.text.strip():
                parts.append(block.text.strip())
        elif isinstance(block, Table):
            for row in block.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
    return "\n".join(parts)


def _extract_plain_text(filename: str, data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AttachmentUnreadable(f"No se ha podido decodificar el texto de '{filename}'")


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()


# --- Concatenación al transcript ---------------------------------------------------------------


def render_attachment_block(attachment: ExtractedAttachment) -> str:
    """Un adjunto delimitado para el prompt, con el separador `--- attachment: <fichero> ---`."""
    body = attachment.text or f"(sin texto extraíble: {attachment.warning})"
    return (
        f"--- attachment: {attachment.filename} ---\n"
        f"{body}\n"
        f"--- fin del adjunto: {attachment.filename} ---"
    )


def enrich_transcript(transcript: str, attachments: Sequence[ExtractedAttachment]) -> str:
    """Transcripción más el texto de los adjuntos, cada uno con su separador. Sin adjuntos, la transcripción tal cual."""
    if not attachments:
        return transcript
    return "\n\n".join([transcript.strip(), *(render_attachment_block(a) for a in attachments)])
