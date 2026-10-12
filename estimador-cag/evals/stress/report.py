"""Genera `REPORT.md` a partir de `results.csv` (sesión 06, paso 6).

    uv run python -m evals.stress.report                                  # evals/stress/results.csv -> REPORT.md
    uv run python -m evals.stress.report --input otro.csv --output otro.md

Todo son tablas en Markdown (sin gráficos): la tabla resumen por escenario × tamaño de adjunto, las tres
curvas canónicas (latencia vs tokens, coste acumulado vs turno, recall vs N), la curva de adjuntos y dos
párrafos de lectura con afirmaciones cuantitativas calculadas de los datos. Los párrafos los escribe este
módulo a partir de los números; conviene releerlos y matizarlos antes de entregar, pero las cifras son las del CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from evals.stress.scenarios import CHECKPOINTS, SCENARIOS

DEFAULT_INPUT = Path(__file__).resolve().parent / "results.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "REPORT.md"
RECALL_FLOOR = 0.6  # umbral del enunciado: "el recall cae bajo el 60 %"


# --- Lectura y utilidades -------------------------------------------------------------------------

_INT = ("attachment_kb", "repeat", "turn_index", "enriched_transcript_chars", "attachments_total_chars", "messages_in_window",
        "anchors_count", "summary_chars", "tokens_in", "tokens_out", "latency_ms", "system_prompt_chars", "context_chars",
        "dropped_turns", "summarizer_tokens_in", "summarizer_tokens_out", "summarizer_latency_ms", "llm_latency_ms",
        "client_latency_ms", "estimation_chars", "attachment_markers_expected", "attachment_markers_recalled",
        "facts_expected", "facts_recalled")
_FLOAT = ("cost_usd", "summarizer_cost_usd", "cumulative_cost_usd", "attachment_recall", "memory_drift_score")
_BOOL = ("summarizer_called", "title_matches_project", "latency_budget_pass", "cost_budget_pass")


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in _INT:
            row[key] = int(float(row[key])) if row.get(key) not in (None, "") else None
        for key in _FLOAT:
            row[key] = float(row[key]) if row.get(key) not in (None, "") else None
        for key in _BOOL:
            row[key] = str(row.get(key, "")).lower() == "true"
    return rows


def percentile(values: Sequence[float], p: float) -> float | None:
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    rank = max(1, math.ceil(p / 100 * len(data)))
    return data[rank - 1]


def mean(values: Iterable[float | None]) -> float | None:
    data = [v for v in values if v is not None]
    return sum(data) / len(data) if data else None


def fmt(value: float | None, digits: int = 0, suffix: str = "") -> str:
    if value is None:
        return "–"
    if digits == 0:
        return f"{int(round(value)):,}".replace(",", ".") + suffix
    return f"{value:.{digits}f}{suffix}"


def pct(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.0f} %"


def table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def misses(rows: Iterable[dict]) -> list[dict]:
    return [r for r in rows if r.get("cache_hit_kind") == "none"]


def by(rows: Iterable[dict], *keys: str) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    return dict(sorted(groups.items(), key=lambda kv: tuple((x is None, x) for x in kv[0])))


# --- Secciones -------------------------------------------------------------------------------------

def summary_table(rows: list[dict]) -> str:
    out = []
    for (scenario, kb), group in by(rows, "scenario", "attachment_kb").items():
        lat = [r["latency_ms"] for r in group]
        lat_miss = [r["latency_ms"] for r in misses(group)]
        exact = sum(r["cache_hit_kind"] == "exact" for r in group) / len(group)
        semantic = sum(r["cache_hit_kind"] == "semantic" for r in group) / len(group)
        out.append([
            scenario, f"{kb} KB", len(group),
            fmt(percentile(lat, 50)), fmt(percentile(lat, 95)),
            fmt(percentile(lat_miss, 50)), fmt(percentile(lat_miss, 95)),
            fmt(sum(r["cost_usd"] or 0 for r in group), 4),
            pct(exact), pct(semantic),
            pct(mean(r["memory_drift_score"] for r in group)),
            pct(sum(r["latency_budget_pass"] for r in group) / len(group)),
            pct(sum(r["cost_budget_pass"] for r in group) / len(group)),
        ])
    return table(
        ["Escenario", "Adjunto", "Turnos", "P50 lat (ms)", "P95 lat (ms)", "P50 sin caché", "P95 sin caché",
         "Coste total (USD)", "Hit exacto", "Hit semántico", "Recall medio", "Latency budget OK", "Cost budget OK"],
        out,
    )


def curve_latency_vs_tokens(rows: list[dict], bin_size: int = 2000) -> str:
    data = [r for r in misses(rows) if r["tokens_in"]]
    if not data:
        return "_Sin llamadas reales al LLM (todo caché o LLM simulado)._"
    bins: dict[int, list[dict]] = defaultdict(list)
    for r in data:
        bins[(r["tokens_in"] // bin_size) * bin_size].append(r)
    out = []
    for start in sorted(bins):
        group = bins[start]
        out.append([
            f"{start:,}–{start + bin_size - 1:,}".replace(",", "."), len(group),
            fmt(percentile([r["latency_ms"] for r in group], 50)), fmt(percentile([r["latency_ms"] for r in group], 95)),
            fmt(mean(r["tokens_out"] for r in group)), fmt(mean(r["cost_usd"] for r in group), 6),
            pct(sum(r["summarizer_called"] for r in group) / len(group)),
        ])
    return table(["tokens_in", "Turnos", "P50 latencia (ms)", "P95 latencia (ms)", "tokens_out medio", "Coste medio (USD)", "Con summarizer"], out)


def curve_cost_vs_turn(rows: list[dict]) -> str:
    base = [r for r in rows if r["attachment_kb"] == 0 and r["repeat"] == 1]
    if not base:
        base = [r for r in rows if r["repeat"] == 1]
    scenarios = sorted({r["scenario"] for r in base})
    turns = sorted({r["turn_index"] for r in base})
    index = {(r["scenario"], r["turn_index"]): r for r in base}
    out = []
    for t in turns:
        line = [t]
        for s in scenarios:
            r = index.get((s, t))
            line.append("–" if r is None else f"{r['cumulative_cost_usd']:.5f} ({r['cost_usd']:.5f} · {fmt(r['tokens_in'])} tok)")
        out.append(line)
    return table(["Turno", *[f"{s}: acumulado (turno · tokens_in)" for s in scenarios]], out)


def curve_recall_vs_n(rows: list[dict]) -> str:
    scenarios = sorted({r["scenario"] for r in rows})
    checkpoints = [n for n in CHECKPOINTS if any(r["turn_index"] == n for r in rows)]
    out = []
    for n in checkpoints:
        line = [n]
        for s in scenarios:
            group = [r for r in rows if r["scenario"] == s and r["turn_index"] == n]
            recall = mean(r["memory_drift_score"] for r in group)
            title = mean(float(r["title_matches_project"]) for r in group) if group else None
            facts = group[0]["facts_expected"] if group else 0
            line.append(f"{pct(recall)} ({facts} hechos · título OK {pct(title)})" if group else "–")
        out.append(line)
    return table(["N (turnos)", *[f"{s}: recall (hechos esperados · nombre en título)" for s in scenarios]], out)


def curve_attachments(rows: list[dict]) -> str:
    first = [r for r in rows if r["turn_index"] == 1]
    out = []
    for (kb,), group in by(first, "attachment_kb").items():
        real = misses(group) or group
        out.append([
            f"{kb} KB", fmt(mean(r["attachments_total_chars"] for r in group)),
            fmt(percentile([r["tokens_in"] for r in real], 50)), fmt(percentile([r["latency_ms"] for r in real], 50)),
            fmt(percentile([r["latency_ms"] for r in real], 95)), fmt(mean(r["cost_usd"] for r in real), 6),
            fmt(mean(r["last_resolved_tier"] == "extended" for r in group) if group else None, 2),
            pct(mean(r["attachment_recall"] for r in group if r["attachment_recall"] is not None)) if kb else "–",
        ])
    return table(["Adjunto", "Chars extraídos", "P50 tokens_in", "P50 latencia (ms)", "P95 latencia (ms)", "Coste medio turno 1 (USD)", "Tier extended", "Recall marcadores"], out)


def drift_detail(rows: list[dict]) -> str:
    """Qué pasa con los hechos retirados (pivot, contradiction) y dónde se encuentran los hechos al final."""
    out = []
    for scenario_key in ("pivot", "contradiction"):
        group = [r for r in rows if r["scenario"] == scenario_key]
        if not group:
            continue
        last_n = max(r["turn_index"] for r in group)
        finals = [r for r in group if r["turn_index"] == last_n]
        stale = sum(bool(r["stale_retained"]) for r in finals) / len(finals)
        examples = sorted({r["stale_retained"].replace("|", ", ") for r in finals if r["stale_retained"]})  # "|" rompería la tabla
        where: dict[str, int] = defaultdict(int)
        for r in finals:
            for item in (r["facts_found_in"] or "").split("|"):
                if "@" in item:
                    where[item.split("@", 1)[1]] += 1
        out.append([scenario_key, last_n, pct(stale), "; ".join(examples) or "–", ", ".join(f"{k}: {v}" for k, v in sorted(where.items(), key=lambda kv: -kv[1])) or "–"])
    return table(["Escenario", "N final", "Conversaciones con hechos retirados aún presentes", "Hechos retirados presentes", "Dónde se encuentran los hechos recordados (conteo)"], out) if out else "_Sin escenarios pivot/contradiction en el CSV._"


# --- Lectura -----------------------------------------------------------------------------------------

def _summarizer_description(rows: list[dict]) -> str:
    activated = [r for r in rows if r["summarizer_mode"] not in (None, "", "off")]
    if not activated:
        return "no se activó (ningún turno salió de la ventana)"
    modes = sorted({r["summarizer_mode"] for r in activated})
    onset = min(r["turn_index"] for r in activated)
    return f"modo `{'`, `'.join(modes)}`, activo desde el turno {onset}"


def _signed_pct(value: float) -> str:
    return f"{'+' if value >= 0 else '−'}{abs(value) * 100:.0f} %"


def _stale_sentence(rows: list[dict]) -> str:
    pivot = [r for r in rows if r["scenario"] == "pivot"]
    if not pivot:
        return ""
    last_n = max(r["turn_index"] for r in pivot)
    finals = [r for r in pivot if r["turn_index"] == last_n]
    share = sum(bool(r["stale_retained"]) for r in finals) / len(finals)
    return (
        f" En `pivot`, el {pct(share)} de las conversaciones llega al turno {last_n} con el stack retirado (React Native) todavía presente: "
        f"`project_metadata` acumula tecnologías pero no sabe retirar ninguna."
    )


def _first_turn_below(rows: list[dict], scenario: str, floor: float = RECALL_FLOOR) -> int | None:
    per_turn = by([r for r in rows if r["scenario"] == scenario and r["memory_drift_score"] is not None], "turn_index")
    for (t,), group in per_turn.items():
        if (mean(r["memory_drift_score"] for r in group) or 0) < floor:
            return t
    return None


def _ratio(a: float | None, b: float | None) -> float | None:
    return None if not a or not b else a / b


def reading(rows: list[dict], meta: dict | None) -> str:
    base = [r for r in rows if r["attachment_kb"] == 0] or rows
    real = misses(base) or base
    scenarios = sorted({r["scenario"] for r in rows})
    last_n = max(r["turn_index"] for r in rows)
    t1 = [r for r in real if r["turn_index"] == 1]
    tn = [r for r in real if r["turn_index"] == last_n]
    cost_ratio = _ratio(mean(r["cost_usd"] for r in tn), mean(r["cost_usd"] for r in t1))
    tokens_ratio = _ratio(mean(r["tokens_in"] for r in tn), mean(r["tokens_in"] for r in t1))
    latency_ratio = _ratio(mean(r["latency_ms"] for r in tn), mean(r["latency_ms"] for r in t1))
    summarizer_onset = min((r["turn_index"] for r in rows if r["summarizer_called"]), default=None)
    over_latency = 1 - (sum(r["latency_budget_pass"] for r in real) / len(real)) if real else 0
    over_cost = 1 - (sum(r["cost_budget_pass"] for r in real) / len(real)) if real else 0
    breaks = {s: _first_turn_below(rows, s) for s in scenarios}
    recall_last = {s: mean(r["memory_drift_score"] for r in rows if r["scenario"] == s and r["turn_index"] == last_n) for s in scenarios}
    title_last = mean(float(r["title_matches_project"]) for r in rows if r["turn_index"] == last_n)
    first = [r for r in rows if r["turn_index"] == 1]
    att = {kb: group for (kb,), group in by(first, "attachment_kb").items()}
    att_recall = {kb: mean(r["attachment_recall"] for r in g) for kb, g in att.items() if kb}
    att_latency = {kb: percentile([r["latency_ms"] for r in (misses(g) or g)], 50) for kb, g in att.items()}
    fake = bool(meta and meta.get("fake_llm"))

    degradations = {
        "coste": (cost_ratio or 1) - 1,
        "latencia": (latency_ratio or 1) - 1,
        "pérdida de memoria": 1 - (mean(recall_last.values()) if any(v is not None for v in recall_last.values()) else 1),
    }
    dominant = max(degradations, key=degradations.get)

    p1 = (
        f"**Dónde empieza a romperse mi CAG.** Con la ventana de {meta['service'].get('session_max_turns') if meta and meta.get('service') else 6} turnos, "
        f"el primer síntoma no es un error sino un coste: {'el summarizer entra en el turno ' + str(summarizer_onset) + ' (el primero que expulsa un turno de la ventana) y a partir de ahí cada turno paga dos llamadas' if summarizer_onset else 'el summarizer no llegó a activarse en esta tirada'}. "
        f"El coste del turno {last_n} multiplica por {fmt(cost_ratio, 2)} el del turno 1 (tokens de entrada ×{fmt(tokens_ratio, 2)}) y la latencia por {fmt(latency_ratio, 2)}; "
        f"el {pct(over_latency)} de los turnos sin caché supera el presupuesto de latencia y el {pct(over_cost)} el de coste. "
        f"La memoria se degrada en silencio: "
        + "; ".join(
            f"en `{s}` el recall del fact-tracker {'cae bajo el 60 % a partir del turno ' + str(breaks[s]) if breaks[s] else 'no baja del 60 % en ' + str(last_n) + ' turnos'} y termina en {pct(recall_last[s])}"
            for s in scenarios
        )
        + f". El nombre del proyecto sigue en el título de la estimación en el {pct(title_last)} de las conversaciones en el turno {last_n}."
    )
    att_sentence = ""
    if att_recall:
        worst = min(att_recall, key=lambda kb: att_recall[kb] or 0)
        att_sentence = (
            f" Con adjuntos, la latencia P50 del turno 1 pasa de {fmt(att_latency.get(0))} ms sin adjunto a {fmt(att_latency.get(max(att_latency)))} ms con {max(att_latency)} KB, "
            f"y el recall de los marcadores del PDF baja hasta {pct(att_recall[worst])} con {worst} KB"
            + ("; el tope `ATTACHMENT_MAX_CHARS=60.000` trunca el de 100 KB, así que los marcadores del 75 % y del final del documento nunca llegan al modelo." if max(att_latency) >= 100 else ".")
        )
    p2 = (
        f"**Qué dimensión domina y cuándo saltar a RAG.** Entre coste ({_signed_pct(degradations['coste'])}), latencia ({_signed_pct(degradations['latencia'])}) y pérdida de memoria "
        f"({_signed_pct(-degradations['pérdida de memoria'])} de recall), la dimensión que más se degrada entre el turno 1 y el {last_n} es **{dominant}**. "
        f"Coste y latencia crecen de forma acotada porque la ventana y el resumen ponen un techo al contexto; lo que no está acotado es lo que se pierde para mantener ese techo: "
        f"el resumen está limitado a 1.500 caracteres y las anclas a 24, así que cada turno nuevo desplaza hechos antiguos. "
        f"Los hechos que sobreviven lo hacen por la memoria estructurada (`project_metadata`) o por una ancla literal; los que se reformulan en el resumen cuentan como perdidos."
        + _stale_sentence(rows)
        + att_sentence
        + f" El caso límite que justifica RAG es el que combina las dos curvas: un proyecto que supera la ventana (N > {(meta['service'].get('session_max_turns') if meta and meta.get('service') else 6)}) y además arrastra documentación que no cabe (> 60.000 caracteres). "
        f"Ahí el CAG solo puede elegir entre pagar el contexto completo en cada turno o resumirlo y perder recall; recuperar por relevancia solo lo que cada turno necesita es la única opción que no sacrifica ninguna de las tres curvas."
        + (" **Esta tirada usó un LLM simulado: las cifras validan el pipeline, no son un baseline.**" if fake else "")
    )
    return p1 + "\n\n" + p2


# --- Documento ---------------------------------------------------------------------------------------

def render(rows: list[dict], meta: dict | None = None) -> str:
    if not rows:
        return "# Stress del CAG\n\n_Sin filas en el CSV._\n"
    service = (meta or {}).get("service") or {}
    args = (meta or {}).get("args") or {}
    scenarios = sorted({r["scenario"] for r in rows})
    sizes = sorted({r["attachment_kb"] for r in rows})
    repeats = max(r["repeat"] for r in rows)
    real = misses(rows)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fake = bool((meta or {}).get("fake_llm"))
    parts = [
        "# Stress del CAG — baseline cuantitativo (sesión 06)",
        "",
        f"_Generado el {generated} por `evals/stress/report.py` a partir de `results.csv` ({len(rows)} filas, {len(real)} turnos con llamada real al LLM)._",
        "",
        ("> **Aviso:** esta tirada usó el LLM simulado (`--fake`). Las tablas demuestran que el pipeline funciona de punta a punta; "
         "los números no son un baseline y hay que regenerarlos con `--http` contra el servicio con API key.\n" if fake else ""),
        "## Configuración de la tirada",
        "",
        table(["Parámetro", "Valor"], [
            ["Proveedor / modelo", f"{service.get('provider', '?')} / {service.get('model', '?')}"],
            ["Plantilla", service.get("prompt_template", rows[0].get("prompt_version", "?"))],
            ["Ventana (SESSION_MAX_TURNS)", service.get("session_max_turns", "?")],
            ["Tope de adjunto (ATTACHMENT_MAX_CHARS)", "60.000 caracteres"],
            ["Summarizer", _summarizer_description(rows)],
            ["Escenarios", ", ".join(f"`{s}` ({SCENARIOS[s].name})" if s in SCENARIOS else s for s in scenarios)],
            ["Tamaños de adjunto", ", ".join(f"{kb} KB" for kb in sizes)],
            ["Repeticiones", repeats],
            ["Presupuestos", f"latencia ≤ {args.get('latency_budget_ms', '?')} ms · coste ≤ {args.get('cost_budget_usd', '?')} USD por turno"],
            ["Inicio / fin", f"{(meta or {}).get('started_at', '?')} / {(meta or {}).get('finished_at', '?')}"],
        ]),
        "",
        "Las repeticiones miden las **ráfagas**: la segunda y tercera pasada de cada conversación envían exactamente los mismos prompts, "
        "así que son el test de la caché exacta (y, si cambia alguna palabra, de la semántica). Por eso la tabla resumen separa la latencia "
        "de todos los turnos de la de los turnos **sin caché**, que es la curva real del LLM.",
        "",
        "## 1. Tabla resumen (escenario × tamaño de adjunto)",
        "",
        summary_table(rows),
        "",
        "## 2. Curva 1 — latencia vs tokens de entrada (solo turnos sin caché)",
        "",
        curve_latency_vs_tokens(rows),
        "",
        "## 3. Curva 2 — coste acumulado vs turno (sin adjunto, primera repetición)",
        "",
        curve_cost_vs_turn(rows),
        "",
        "## 4. Curva 3 — MemoryDriftMetric vs N (media sobre tamaños y repeticiones)",
        "",
        "Recall = fracción de los hechos declarados en turnos k < N que aparecen literalmente (case-insensitive) en el resumen, "
        "las anclas o `project_metadata` del turno N. No mira el historial: mide lo que sobrevive a la ventana.",
        "",
        curve_recall_vs_n(rows),
        "",
        "### Hechos retirados y dónde viven los recordados",
        "",
        drift_detail(rows),
        "",
        "## 5. Adjuntos grandes (turno 1 de cada conversación)",
        "",
        curve_attachments(rows),
        "",
        "## 6. Lectura",
        "",
        reading(rows, meta),
        "",
        "## 7. Notas de método",
        "",
        "- `turn_observed` se lee del campo `turn_observed` de la respuesta de `POST /sessions/{id}/estimate` y se contrasta con "
        "`GET /sessions/{id}.last_turn_observed`: es el mismo diccionario que se emite al log con structlog. No hay que parsear stdout "
        "ni reconciliar timestamps, y el runner funciona igual contra Docker, uvicorn o in-process.",
        "- `LatencyBudgetMetric`, `CostBudgetMetric` y `MemoryDriftMetric` viven en `evals/stress/metrics.py` porque evalúan el shape de "
        "`turn_observed` y del snapshot de sesión, no una estimación suelta como las tres de `evals/metrics.py`; comparten `MetricResult` y `run_all_metrics`.",
        "- `latency_ms` y `cost_usd` del turno incluyen la llamada del summarizer cuando la hay (`summarizer_*` las desglosa). Un acierto de caché "
        "registra 0 tokens y 0 USD.",
        "- Los PDFs son sintéticos y deterministas (`evals/stress/fixtures/build_pdfs.py`); llevan cinco marcadores repartidos por el documento "
        "para medir el recall del adjunto en la estimación. No se versionan: el runner los regenera.",
        "- No se ha tocado ninguna constante entre medidas (ventana de 6, resumen de 1.500 caracteres, 24 anclas, tope de adjunto de 60.000).",
        "",
    ]
    return "\n".join(parts)


def write_report(rows: list[dict], path: Path, meta: dict | None = None) -> Path:
    typed = _retype(rows)  # idempotente: vale para filas recién medidas y para filas leídas del CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(typed, meta), encoding="utf-8")
    return path


def _retype(rows: list[dict]) -> list[dict]:
    """Las filas recién escritas por el runner ya son tipadas salvo los strings vacíos; las del CSV se tipan al leer."""
    typed = []
    for row in rows:
        copy = dict(row)
        for key in _INT:
            copy[key] = int(float(copy[key])) if copy.get(key) not in (None, "") else None
        for key in _FLOAT:
            copy[key] = float(copy[key]) if copy.get(key) not in (None, "") else None
        for key in _BOOL:
            copy[key] = copy.get(key) is True or str(copy.get(key, "")).lower() == "true"
        typed.append(copy)
    return typed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="REPORT.md a partir del CSV del stress")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    meta_path = args.input.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    write_report(load_rows(args.input), args.output, meta)
    print(f"Reporte en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
