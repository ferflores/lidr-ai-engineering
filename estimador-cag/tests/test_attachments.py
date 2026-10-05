"""Tests de la extracción local de texto de adjuntos (camino B): PDF, Word, texto, límites y errores."""

import pytest

from app.services.attachments import (
    AttachmentTooLarge,
    AttachmentUnreadable,
    ExtractedAttachment,
    UnsupportedAttachment,
    detect_kind,
    enrich_transcript,
    extract_text,
    render_attachment_block,
)
from tests.documentos import make_docx, make_pdf


def test_pdf_de_dos_paginas():
    pdf = make_pdf(
        ["Especificación técnica v2", "Pasarela de pago: Stripe (tarjeta y Bizum)."],
        ["Página dos: integración con el TPV del obrador."],
    )

    adjunto = extract_text("especificacion.pdf", pdf)

    assert adjunto.kind == "pdf"
    assert adjunto.pages == 2
    assert adjunto.size_bytes == len(pdf)
    assert "Especificación técnica v2" in adjunto.text
    assert "Stripe" in adjunto.text
    assert "Página dos" in adjunto.text
    assert adjunto.truncated is False
    assert adjunto.warning is None
    assert adjunto.chars == len(adjunto.text) > 0


def test_docx_con_parrafos_y_tabla_en_orden():
    docx = make_docx(
        ["Propuesta anterior (marzo).", "Hosting en AWS y pagos con Stripe."],
        table=[["Módulo", "Horas"], ["Catálogo", "40"]],
        title="Propuesta",
    )

    adjunto = extract_text("propuesta.docx", docx)

    assert adjunto.kind == "docx"
    assert adjunto.pages is None
    assert adjunto.text.startswith("Propuesta\nPropuesta anterior (marzo).")
    assert "Hosting en AWS y pagos con Stripe." in adjunto.text
    assert "Módulo | Horas\nCatálogo | 40" in adjunto.text


def test_texto_plano_utf8_y_windows():
    assert extract_text("notas.txt", "Reunión con el cliente: integración con HubSpot.".encode("utf-8")).text == (
        "Reunión con el cliente: integración con HubSpot."
    )
    assert extract_text("notas.md", "Año: 2026 — reparto".encode("cp1252")).kind == "texto"
    assert "Año" in extract_text("notas.md", "Año: 2026".encode("cp1252")).text


def test_el_texto_se_recorta_al_maximo_de_caracteres_y_se_avisa():
    adjunto = extract_text("largo.txt", ("palabra " * 100).encode(), max_chars=50)

    assert adjunto.truncated is True
    assert adjunto.text.startswith("palabra palabra")
    assert "[... adjunto truncado a 50 caracteres ...]" in adjunto.text


def test_demasiado_grande():
    with pytest.raises(AttachmentTooLarge, match="demasiado grande"):
        extract_text("grande.txt", b"x" * 101, max_bytes=100)


def test_vacio():
    with pytest.raises(AttachmentUnreadable, match="vacío"):
        extract_text("vacio.pdf", b"")


@pytest.mark.parametrize("filename", ["imagen.png", "programa.exe", "hoja.xlsx", "sin_extension"])
def test_formato_no_soportado(filename):
    with pytest.raises(UnsupportedAttachment, match="no soportado"):
        extract_text(filename, b"datos")


def test_word_antiguo_sugiere_convertir_a_docx():
    with pytest.raises(UnsupportedAttachment, match="Convierte el .doc a .docx"):
        detect_kind("propuesta.doc")


def test_sin_extension_reconocible_usa_el_content_type():
    assert detect_kind("adjunto", "application/pdf") == "pdf"
    assert detect_kind("adjunto", "text/plain; charset=utf-8") == "texto"
    assert detect_kind("adjunto", "application/vnd.openxmlformats-officedocument.wordprocessingml.document") == "docx"


def test_pdf_corrupto():
    with pytest.raises(AttachmentUnreadable, match="No se ha podido leer el PDF"):
        extract_text("roto.pdf", b"%PDF-1.4\nesto no es un pdf de verdad")


def test_docx_corrupto():
    with pytest.raises(AttachmentUnreadable, match="Word"):
        extract_text("roto.docx", b"PK\x03\x04 esto no es un docx")


def test_pdf_sin_texto_extraible_avisa_en_vez_de_fallar():
    adjunto = extract_text("escaneado.pdf", make_pdf([]))

    assert adjunto.text == ""
    assert adjunto.chars == 0
    assert "no contiene texto extraíble" in adjunto.warning
    assert "OCR" in adjunto.warning
    assert "(sin texto extraíble" in render_attachment_block(adjunto)


def test_el_nombre_de_fichero_se_normaliza():
    assert extract_text("../../etc/passwd.txt", b"hola").filename == "passwd.txt"
    assert extract_text("C:\\Users\\fer\\propuesta.txt", b"hola").filename == "propuesta.txt"


def test_enrich_transcript_concatena_con_separadores_en_orden():
    a = ExtractedAttachment("spec.pdf", "pdf", "Texto del PDF", 10, 1, False, None)
    b = ExtractedAttachment("propuesta.docx", "docx", "Texto del Word", 10, None, False, None)

    enriched = enrich_transcript("  La transcripción.  ", [a, b])

    assert enriched == (
        "La transcripción.\n\n"
        "--- attachment: spec.pdf ---\nTexto del PDF\n--- fin del adjunto: spec.pdf ---\n\n"
        "--- attachment: propuesta.docx ---\nTexto del Word\n--- fin del adjunto: propuesta.docx ---"
    )
    assert enrich_transcript("Sin adjuntos", []) == "Sin adjuntos"
