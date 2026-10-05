"""Generadores de documentos de prueba (PDF y Word) sin dependencias de generación: pypdf solo lee.

Los usan los tests (`tests/conftest.py` los reexporta) y `scripts/generar-adjuntos.py` para crear los
adjuntos de ejemplo de `transcripciones/adjuntos/`.
"""

from __future__ import annotations

import io

from docx import Document


def make_pdf(*pages: list[str]) -> bytes:
    """Un PDF mínimo y válido (texto Helvetica, codificación WinAnsi) con una página por lista de líneas.

    Cada elemento de la lista es una línea (sin ajuste automático: conviene no pasar de ~85 caracteres).
    """
    pages = pages or (["Documento de prueba"],)

    def escape(line: str) -> bytes:
        raw = line.encode("cp1252", errors="replace")
        return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")

    objects: list[bytes] = []
    n_pages = len(pages)
    # 1: catálogo, 2: árbol de páginas, 3: fuente; después, por página: objeto página + objeto contenido
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n_pages))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    for i, lines in enumerate(pages):
        page_obj = 4 + 2 * i
        content_obj = page_obj + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_obj} 0 R >>".encode()
        )
        stream = b"BT /F1 11 Tf 56 740 Td 15 TL\n" + b"".join(b"(" + escape(line) + b") Tj T*\n" for line in lines) + b"ET"
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def make_docx(paragraphs: list[str], table: list[list[str]] | None = None, title: str | None = None) -> bytes:
    """Un documento Word con un título opcional, párrafos y, opcionalmente, una tabla."""
    document = Document()
    if title:
        document.add_heading(title, level=1)
    for text in paragraphs:
        document.add_paragraph(text)
    if table:
        rows = document.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, value in enumerate(row):
                rows.cell(r, c).text = value
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()
