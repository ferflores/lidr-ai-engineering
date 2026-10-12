"""Framework mínimo de evals: una observación, varias métricas binarias, un resultado por métrica.

El patrón es deliberadamente pequeño para que añadir una métrica sean ~15 líneas:

    class MiMetrica:
        name = "mi_metrica"
        def evaluate(self, observation) -> MetricResult: ...

`MetricResult` lleva siempre `name`, `score` (0.0-1.0), `passed` (el veredicto binario) y `details` (lo
que haga falta para entender por qué). `run_all_metrics` evalúa una lista de métricas sobre una
observación y nunca lanza: una métrica que explota produce un resultado fallido con la excepción en
`details`, porque un eval que se cae a mitad no mide nada.

Las tres métricas de la sesión 05 evalúan una estimación suelta (`EvalObservation`, lo que devuelve
`POST /api/v1/estimate` más lo que el caso golden espera):

- `SchemaAdherenceMetric`: la salida respeta el formato que exige la plantilla (títulos y líneas obligatorias
  según `detail_level`). Fallar aquí es fallar "duro": el parseo de `project_metadata` depende de ese formato.
- `CostBoundsMetric`: el coste de la llamada no supera el tope del caso (o el global).
- `ContentRecallMetric`: los términos que el caso declara como imprescindibles aparecen en la estimación.
  Es la que detecta la degradación silenciosa: la salida puede ser válida, barata y a la vez vacía.

Determinismo > sofisticación: todo es match de texto, sin LLM-as-judge ni embeddings.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class MetricResult:
    name: str
    score: float
    passed: bool
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"name": self.name, "score": self.score, "passed": self.passed, "details": self.details}


class Metric(Protocol):
    name: str

    def evaluate(self, observation: object) -> MetricResult: ...


@dataclass(frozen=True)
class EvalObservation:
    """Lo que se observa al ejecutar un caso golden contra el servicio (una estimación suelta)."""

    case_id: str
    estimation: str
    detail_level: str = "completa"
    cost_usd: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: int | None = None
    expected_terms: tuple[str, ...] = ()
    max_cost_usd: float | None = None


def run_all_metrics(observation: object, metrics: Sequence[Metric]) -> list[MetricResult]:
    """Evalúa todas las métricas; una que falle con excepción cuenta como no superada, no interrumpe el eval."""
    results: list[MetricResult] = []
    for metric in metrics:
        try:
            results.append(metric.evaluate(observation))
        except Exception as exc:  # noqa: BLE001 - un eval nunca se cae a medias
            results.append(MetricResult(name=metric.name, score=0.0, passed=False, details={"error": f"{exc.__class__.__name__}: {exc}"}))
    return results


def contains(haystack: str, needle: str) -> bool:
    """Match exacto sin distinguir mayúsculas ni espacios repetidos."""
    return " ".join(needle.casefold().split()) in " ".join(haystack.casefold().split())


# --- Métricas de la sesión 05 -----------------------------------------------------------------------

REQUIRED_SECTIONS: dict[str, tuple[str, ...]] = {
    "completa": (
        "## Estimación:",
        "### Resumen del alcance",
        "### Desglose de tareas",
        "**Subtotal de tareas:",
        "**Contingencia (",
        "**Total estimado:",
        "**Coste estimado:",
        "**Equipo recomendado:",
        "**Duración estimada:",
        "### Supuestos y riesgos",
        "### Preguntas pendientes para el cliente",
    ),
    "resumen": (
        "## Estimación:",
        "### Resumen del alcance",
        "### Desglose por bloques",
        "**Subtotal de bloques:",
        "**Contingencia (",
        "**Total estimado:",
        "**Coste estimado:",
        "**Equipo recomendado:",
        "**Duración estimada:",
        "### Supuestos y riesgos",
    ),
}
FORBIDDEN_SECTIONS: dict[str, tuple[str, ...]] = {
    "completa": (),
    "resumen": ("### Preguntas pendientes",),
}
_TOTAL = re.compile(r"\*\*Total estimado:\s*([\d.,]+)\s*h", re.IGNORECASE)


class SchemaAdherenceMetric:
    """1.0 si la estimación trae todas las secciones y líneas que exige la plantilla para su `detail_level`
    (y ninguna de las prohibidas) y el total estimado es un número; score = fracción de requisitos cumplidos."""

    name = "schema_adherence"

    def evaluate(self, observation: EvalObservation) -> MetricResult:
        level = observation.detail_level if observation.detail_level in REQUIRED_SECTIONS else "completa"
        text = observation.estimation
        required = REQUIRED_SECTIONS[level]
        missing = [section for section in required if not contains(text, section)]
        forbidden = [section for section in FORBIDDEN_SECTIONS[level] if contains(text, section)]
        total_ok = _TOTAL.search(text) is not None
        checks = len(required) + len(FORBIDDEN_SECTIONS[level]) + 1
        failures = len(missing) + len(forbidden) + (0 if total_ok else 1)
        score = round(1 - failures / checks, 4)
        return MetricResult(
            name=self.name,
            score=score,
            passed=failures == 0,
            details={"detail_level": level, "missing": missing, "forbidden_present": forbidden, "total_parsed": total_ok},
        )


class CostBoundsMetric:
    """1.0 si el coste de la llamada ≤ tope (el del caso, si lo declara; si no, `max_usd`); 0.0 si no o si no hay coste."""

    name = "cost_bounds"

    def __init__(self, max_usd: float = 0.01):
        self.max_usd = max_usd

    def evaluate(self, observation: EvalObservation) -> MetricResult:
        bound = observation.max_cost_usd if observation.max_cost_usd is not None else self.max_usd
        cost = observation.cost_usd
        passed = cost is not None and cost <= bound
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details={"cost_usd": cost, "max_usd": bound, "ratio": round(cost / bound, 3) if cost is not None and bound else None},
        )


class ContentRecallMetric:
    """score = fracción de `expected_terms` presentes en la estimación (case-insensitive); 1.0/pasa si ≥ `threshold`."""

    name = "content_recall"

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def evaluate(self, observation: EvalObservation) -> MetricResult:
        terms = list(observation.expected_terms)
        if not terms:
            return MetricResult(name=self.name, score=1.0, passed=True, details={"expected": [], "missing": [], "note": "sin términos esperados"})
        missing = [term for term in terms if not contains(observation.estimation, term)]
        score = round((len(terms) - len(missing)) / len(terms), 4)
        return MetricResult(
            name=self.name,
            score=score,
            passed=score >= self.threshold,
            details={"expected": terms, "missing": missing, "threshold": self.threshold},
        )


def default_metrics(max_cost_usd: float = 0.01, recall_threshold: float = 1.0) -> list[Metric]:
    return [SchemaAdherenceMetric(), CostBoundsMetric(max_usd=max_cost_usd), ContentRecallMetric(threshold=recall_threshold)]


def summarize(results: Iterable[Sequence[MetricResult]]) -> dict[str, dict[str, float]]:
    """Tasa de aprobados y score medio por métrica sobre una colección de casos."""
    totals: dict[str, list[MetricResult]] = {}
    for case_results in results:
        for result in case_results:
            totals.setdefault(result.name, []).append(result)
    return {
        name: {
            "cases": len(items),
            "pass_rate": round(sum(r.passed for r in items) / len(items), 4),
            "mean_score": round(sum(r.score for r in items) / len(items), 4),
        }
        for name, items in totals.items()
    }
