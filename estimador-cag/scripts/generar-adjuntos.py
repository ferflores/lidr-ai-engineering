#!/usr/bin/env -S uv run python
"""Genera los adjuntos de ejemplo de `transcripciones/adjuntos/` para probar la conversación de tres turnos.

    uv run scripts/generar-adjuntos.py

Crea un PDF (especificación técnica del cliente) y un Word (propuesta anterior de otra consultora) para la
tienda online de la panadería. Son ficticios y se generan sin librerías de maquetación (ver tests/documentos.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.documentos import make_docx, make_pdf  # noqa: E402

DESTINO = ROOT / "transcripciones" / "adjuntos"

ESPECIFICACION_PDF = [
    [
        "PANADERIA ARTESANA LA ESPIGA - Especificacion tecnica de la tienda online (v2)",
        "",
        "1. Catalogo",
        "- 60 referencias en 4 familias (panes, bolleria, pasteleria, encargos).",
        "- Stock diario por producto con cierre de ventas al agotarse; el catalogo actual esta en Excel.",
        "- Fotos y descripciones las aporta la panaderia.",
        "",
        "2. Pedidos y pago",
        "- Pasarela de pago: Stripe (tarjeta) y Bizum. No se admite pago contra reembolso.",
        "- Pedido minimo de 10 EUR. Franjas de recogida en tienda de 30 minutos.",
        "- Reparto a domicilio solo dentro de un radio de 3 km, con coste fijo de 2,50 EUR.",
        "- Pedidos recurrentes semanales (suscripcion de pan) con pausa y cancelacion por el cliente.",
        "",
        "3. Panel del obrador",
        "- Vista de produccion del dia siguiente agrupada por producto y franja.",
        "- Exportacion a Excel de los pedidos del dia.",
        "- Avisos por WhatsApp al cliente cuando el pedido esta listo.",
    ],
    [
        "4. Requisitos no funcionales",
        "- Web en castellano y catalan.",
        "- Hosting en AWS (ya tienen cuenta). Alta disponibilidad no requerida.",
        "- Cumplimiento RGPD: consentimiento de cookies y politica de privacidad.",
        "- Integracion con la impresora de tickets del obrador: fuera de alcance en esta fase.",
        "",
        "5. Plazo y presupuesto",
        "- Lanzamiento deseado antes de la campana de Navidad: 8 semanas desde el inicio.",
        "- Presupuesto orientativo: 12.000 EUR.",
    ],
]

PROPUESTA_DOCX_PARRAFOS = [
    "Propuesta economica presentada en marzo por otra consultora para la tienda online de la panaderia.",
    "Planteaban una tienda sobre WordPress con WooCommerce, pagos con Stripe y hosting gestionado propio.",
    "No incluia pedidos recurrentes, panel del obrador ni reparto por radio. Plazo propuesto: 10 semanas.",
    "El cliente la descarto por el coste del mantenimiento mensual y por no cubrir la produccion diaria.",
]
PROPUESTA_DOCX_TABLA = [
    ["Bloque", "Horas", "Importe"],
    ["Instalacion y tema WooCommerce", "40", "2.000 EUR"],
    ["Catalogo y stock basico", "35", "1.750 EUR"],
    ["Pagos con Stripe", "20", "1.000 EUR"],
    ["Recogida en tienda (franjas)", "30", "1.500 EUR"],
    ["Formacion y despliegue", "15", "750 EUR"],
    ["Total", "140", "7.000 EUR"],
]


def main() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    pdf = DESTINO / "panaderia-especificacion-tecnica.pdf"
    docx = DESTINO / "panaderia-propuesta-anterior.docx"
    pdf.write_bytes(make_pdf(*ESPECIFICACION_PDF))
    docx.write_bytes(make_docx(PROPUESTA_DOCX_PARRAFOS, PROPUESTA_DOCX_TABLA, title="Propuesta anterior (marzo)"))
    for path in (pdf, docx):
        print(f"{path.relative_to(ROOT)}  ({path.stat().st_size:,} bytes)".replace(",", "."))


if __name__ == "__main__":
    main()
