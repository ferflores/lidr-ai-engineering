"""Corpus de adjuntos sintéticos para el stress (sesión 06, bloque 3).

    uv run python -m evals.stress.fixtures.build_pdfs            # genera attach_5kb.pdf ... attach_100kb.pdf aquí al lado

Produce PDFs de texto plano calibrados por **caracteres extraíbles** (lo que acaba en el prompt), no por
bytes: 5.000, 20.000, 50.000 y 100.000 caracteres (≈ 2, 8, 20 y 40 páginas). El de 100 KB supera el tope
`ATTACHMENT_MAX_CHARS=60.000`, así que mide cómo trunca el sistema. `attach_0kb` no existe: es la ausencia de adjunto.

El texto es una especificación ficticia de un proyecto, repetida por secciones, con cinco **marcadores**
colocados al principio, al 25 %, al 50 %, al 75 % y al final (`MARKERS`). Sirven para medir el recall del
contenido del adjunto en la estimación: si la estimación menciona "Redsys" (principio) pero no "catalán"
(final), el adjunto llegó truncado o el modelo no leyó hasta el final. Cuatro de los cinco son tecnologías del
vocabulario de `metadata_extractor`, así que también se puede ver qué llega a `project_metadata`.

Se genera con `tests/documentos.make_pdf` (PDF mínimo sin dependencias de maquetación): determinista, así
que los PDFs no se versionan (ver `.gitignore` de esta carpeta); el runner los regenera si faltan.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.documentos import make_pdf  # noqa: E402

FIXTURES_DIR = Path(__file__).resolve().parent
SIZES_KB: tuple[int, ...] = (5, 20, 50, 100)
LINES_PER_PAGE = 46
MAX_LINE_CHARS = 85

# (posición relativa en el documento, frase con el marcador, marcador a buscar en la estimación)
MARKERS: tuple[tuple[float, str, str], ...] = (
    (0.02, "La pasarela de pago obligatoria es Redsys, por exigencia del banco del cliente.", "Redsys"),
    (0.25, "Los informes de direccion se publicaran en Power BI con refresco diario.", "Power BI"),
    (0.50, "El despliegue se hara en Azure, en la suscripcion corporativa existente.", "Azure"),
    (0.75, "La integracion contable se hara contra SAP mediante ficheros nocturnos.", "SAP"),
    (0.98, "La interfaz debe estar disponible en castellano y en catalan desde el primer dia.", "catalán"),
)
MARKER_TERMS: tuple[str, ...] = tuple(term for _, _, term in MARKERS)

_SECTION_TEMPLATES: tuple[str, ...] = (
    "Seccion {n}. Requisitos funcionales del modulo {n}. El sistema debe permitir dar de alta, consultar y "
    "modificar los registros del modulo {n} con validacion de campos obligatorios, historico de cambios y "
    "exportacion de listados. Los usuarios con rol gestor pueden aprobar o rechazar solicitudes pendientes.",
    "Seccion {n}. Reglas de negocio. Cada operacion del modulo {n} queda registrada con usuario, fecha y motivo. "
    "Las solicitudes caducan a los treinta dias si nadie las atiende y se notifica al responsable por correo. "
    "Los importes se calculan con dos decimales y se redondean al alza en caso de empate.",
    "Seccion {n}. Requisitos no funcionales. Tiempo de respuesta inferior a dos segundos en el percentil 95 "
    "para las consultas del modulo {n}. Copias de seguridad diarias con retencion de noventa dias. Registro de "
    "accesos conforme a la politica de seguridad corporativa y revision trimestral de permisos.",
    "Seccion {n}. Integraciones. El modulo {n} intercambia datos con el resto de modulos mediante eventos "
    "internos y expone un servicio de consulta para terceros con autenticacion por token. Los errores de "
    "integracion se reintentan tres veces antes de escalar a soporte.",
)


def _wrap(text: str, width: int = MAX_LINE_CHARS) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def build_text_lines(target_chars: int) -> list[str]:
    """Líneas de texto (≤ 85 caracteres) hasta sumar ~target_chars, con los marcadores en su posición."""
    lines: list[str] = ["ESPECIFICACION FUNCIONAL DEL PROYECTO (documento sintetico para pruebas de carga)", ""]
    marker_at = {int(target_chars * position): sentence for position, sentence, _ in MARKERS}
    pending = sorted(marker_at)
    chars = sum(len(line) + 1 for line in lines)
    n = 1
    while chars < target_chars:
        paragraph = _SECTION_TEMPLATES[(n - 1) % len(_SECTION_TEMPLATES)].format(n=n)
        while pending and chars >= pending[0]:
            for line in _wrap(marker_at[pending.pop(0)]):
                lines.append(line)
                chars += len(line) + 1
            lines.append("")
            chars += 1
        for line in _wrap(paragraph):
            lines.append(line)
            chars += len(line) + 1
        lines.append("")
        chars += 1
        n += 1
    for position in pending:  # los marcadores que no cupieron van al final
        for line in _wrap(marker_at[position]):
            lines.append(line)
    return lines


def build_pdf(size_kb: int) -> bytes:
    lines = build_text_lines(size_kb * 1000)
    pages = [lines[i : i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]
    return make_pdf(*pages)


def pdf_path(size_kb: int) -> Path:
    return FIXTURES_DIR / f"attach_{size_kb}kb.pdf"


def ensure_pdfs(sizes: tuple[int, ...] = SIZES_KB, force: bool = False) -> list[Path]:
    """Genera los PDFs que falten (o todos con `force`) y devuelve sus rutas."""
    paths = []
    for size in sizes:
        if size == 0:
            continue
        path = pdf_path(size)
        if force or not path.exists():
            path.write_bytes(build_pdf(size))
        paths.append(path)
    return paths


def main() -> None:
    for path in ensure_pdfs(force=True):
        import pypdf  # noqa: PLC0415

        reader = pypdf.PdfReader(str(path))
        chars = sum(len(page.extract_text() or "") for page in reader.pages)
        print(f"{path.name}: {path.stat().st_size:,} bytes, {len(reader.pages)} páginas, {chars:,} caracteres extraíbles".replace(",", "."))


if __name__ == "__main__":
    main()
