#!/usr/bin/env python3
"""Compara varias transcripciones contra el endpoint: tokens, coste y resultado de la estimación.

Uso (con el servicio levantado, con uv o con Docker):
    uv run scripts/comparar.py                          # pobre vs. detallada (panadería)
    uv run scripts/comparar.py a.json b.json [c.json]   # cualquier lista de cuerpos JSON
    BASE_URL=http://localhost:9000 uv run scripts/comparar.py

Cada transcripción supone una llamada real al LLM (y su coste).
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
POR_DEFECTO = [
    "transcripciones/panaderia-descripcion-pobre.json",
    "transcripciones/panaderia-descripcion-detallada.json",
]


def estimar(path: Path) -> dict:
    request = urllib.request.Request(
        f"{BASE_URL}/api/v1/estimate",
        data=path.read_bytes(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detalle = exc.read().decode("utf-8", errors="replace")
        sys.exit(f"Error {exc.code} al estimar {path}: {detalle}")
    except urllib.error.URLError as exc:
        sys.exit(f"No se pudo conectar con {BASE_URL}: {exc.reason}. ¿Está levantado el servicio?")


def horas_estimadas(estimation: str) -> int | None:
    match = re.search(r"Total estimado:\s*([\d.,]+)\s*horas", estimation)
    return int(re.sub(r"[.,]", "", match.group(1))) if match else None


def contar_tareas(estimation: str) -> int:
    return len(re.findall(r"^\s*\d+\.\s", estimation, flags=re.MULTILINE))


def contar_preguntas(estimation: str) -> int:
    seccion = estimation.split("### Preguntas pendientes", 1)
    return len(re.findall(r"^\s*-\s", seccion[1], flags=re.MULTILINE)) if len(seccion) == 2 else 0


def fmt_int(valor: int | None, signo: bool = False) -> str:
    if valor is None:
        return "?"
    texto = f"{abs(valor):,}".replace(",", ".")
    if signo:
        return ("+" if valor >= 0 else "-") + texto
    return texto


def fmt_usd(valor: float | None, signo: bool = False) -> str:
    if valor is None:
        return "?"
    texto = f"{abs(valor):.6f}"
    return (("+" if valor >= 0 else "-") + texto) if signo else texto


def main(argv: list[str]) -> None:
    rutas = [Path(p) for p in (argv or POR_DEFECTO)]
    filas = []
    for ruta in rutas:
        print(f"Estimando {ruta} ...", file=sys.stderr)
        d = estimar(ruta)
        usage = d.get("usage") or {}
        cost = usage.get("cost") or {}
        filas.append(
            {
                "caso": ruta.stem,
                "chars": (d.get("prompt") or {}).get("transcription_chars"),
                "system_chars": (d.get("prompt") or {}).get("system_prompt_chars"),
                "in": usage.get("input_tokens"),
                "out": usage.get("output_tokens"),
                "total": usage.get("total_tokens"),
                "usd": cost.get("total_usd"),
                "horas": horas_estimadas(d["estimation"]),
                "tareas": contar_tareas(d["estimation"]),
                "preguntas": contar_preguntas(d["estimation"]),
                "modelo": d["model"],
                "proveedor": d["provider"],
                "precio_in": cost.get("input_price_per_mtok"),
                "precio_out": cost.get("output_price_per_mtok"),
            }
        )

    primera = filas[0]
    precios = (
        f"precios: {primera['precio_in']} / {primera['precio_out']} USD por millón de tokens (entrada / salida)"
        if primera["precio_in"] is not None
        else "sin precio conocido para este modelo (coste no calculado)"
    )
    print(f"\nComparativa de transcripciones · {primera['modelo']} ({primera['proveedor']}) · {precios}\n")

    ancho = max(len(f["caso"]) for f in filas) + 2
    cab = f"{'Caso':<{ancho}}{'Caract.':>9}{'Tok. entrada':>14}{'Tok. salida':>13}{'Tok. total':>12}{'Coste USD':>12}   Estimación"
    print(cab)
    print("-" * len(cab))
    for f in filas:
        estimacion = f"{fmt_int(f['horas'])} h · {f['tareas']} tareas · {f['preguntas']} preguntas"
        print(
            f"{f['caso']:<{ancho}}{fmt_int(f['chars']):>9}{fmt_int(f['in']):>14}{fmt_int(f['out']):>13}"
            f"{fmt_int(f['total']):>12}{fmt_usd(f['usd']):>12}   {estimacion}"
        )

    if len(filas) == 2 and all(f["in"] is not None for f in filas):
        a, b = filas
        d_chars = (b["chars"] or 0) - (a["chars"] or 0)
        d_in, d_out, d_total = b["in"] - a["in"], b["out"] - a["out"], b["total"] - a["total"]
        d_usd = (b["usd"] - a["usd"]) if a["usd"] is not None and b["usd"] is not None else None
        d_horas = (b["horas"] - a["horas"]) if a["horas"] is not None and b["horas"] is not None else None
        print(
            f"{'Diferencia (2ª - 1ª)':<{ancho}}{fmt_int(d_chars, True):>9}{fmt_int(d_in, True):>14}"
            f"{fmt_int(d_out, True):>13}{fmt_int(d_total, True):>12}{fmt_usd(d_usd, True):>12}   "
            f"{fmt_int(d_horas, True) if d_horas is not None else '?'} h"
        )

        # Contexto fijo (CAG): estimado a partir de las dos llamadas. Los tokens de entrada son
        # contexto fijo + transcripción; con dos transcripciones de distinto tamaño se puede
        # despejar la relación tokens/carácter y, con ella, el tamaño del contexto fijo.
        print()
        if d_chars:
            tokens_por_char = d_in / d_chars
            fijo = round(((a["in"] - a["chars"] * tokens_por_char) + (b["in"] - b["chars"] * tokens_por_char)) / 2)
            print(
                f"Contexto fijo (system prompt con los ejemplos CAG): {fmt_int(a['system_chars'])} caracteres ≈ "
                f"{fmt_int(fijo)} tokens en TODAS las llamadas "
                f"({fijo / a['in']:.0%} de la entrada del 1º caso, {fijo / b['in']:.0%} del 2º)."
            )
        if d_usd is not None and b["usd"]:
            barato, caro = (a, b) if a["usd"] <= b["usd"] else (b, a)
            ahorro = caro["usd"] - barato["usd"]
            print(
                f"Ahorro en tokens de «{barato['caso']}» frente a «{caro['caso']}»: "
                f"{fmt_usd(ahorro)} USD por llamada ({ahorro / caro['usd']:.1%}), "
                f"es decir, {ahorro * 1000:.2f} USD por cada 1.000 estimaciones."
            )
        if d_horas is not None:
            print(
                f"Diferencia entre las estimaciones obtenidas: {fmt_int(abs(d_horas))} horas "
                f"({fmt_int(abs(d_horas) * 50)} € a 50 €/hora). Ese es el orden de magnitud de lo que "
                f"está en juego frente al ahorro en tokens."
            )


if __name__ == "__main__":
    main(sys.argv[1:])
