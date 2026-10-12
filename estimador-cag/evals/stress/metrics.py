"""Métricas del stress (sesión 06, bloque 4): presupuestos como contratos y deriva de memoria.

Viven en `evals/stress/metrics.py` y no en `evals/metrics.py` por una razón de dependencia, no de gusto:
las tres métricas de la sesión 05 evalúan **una estimación suelta** (`EvalObservation`: el texto y su
coste); estas tres evalúan **un turno dentro de una sesión**: el evento `turn_observed` (latencia y coste
del turno, summarizer incluido) y el snapshot de la sesión (`GET /api/v1/sessions/{id}`: resumen, anclas,
`project_metadata`). Mezclar los dos shapes en un módulo obligaría a cada métrica a saber de los dos. Lo
que sí se comparte es el patrón: `MetricResult` y `run_all_metrics` se importan de `evals.metrics` tal cual.

- `LatencyBudgetMetric(budget_ms)`: 1.0 si `latency_ms ≤ budget_ms`. Convierte el SLA en un test.
- `CostBudgetMetric(budget_usd)`: 1.0 si `cost_usd ≤ budget_usd`. El presupuesto por turno como contrato.
- `MemoryDriftMetric(fact, where)`: 1.0 si el hecho declarado en el turno k aparece (match exacto,
  case-insensitive) en el resumen, las anclas o `project_metadata` del turno N > k. Determinismo >
  sofisticación: ni embeddings ni LLM-as-judge; un hecho reformulado cuenta como perdido, y eso se discute en
  el reporte, no se maquilla en la métrica.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from evals.metrics import MetricResult, contains, run_all_metrics  # noqa: F401  (run_all_metrics se reexporta)

DEFAULT_WHERE: tuple[str, ...] = ("summary", "anchors", "metadata")


@dataclass(frozen=True)
class SessionSnapshot:
    """Lo que una métrica de memoria puede mirar de una sesión en el turno N."""

    turn_index: int
    summary: str = ""
    anchors: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_api(cls, state: dict) -> SessionSnapshot:
        """Construye el snapshot desde la respuesta de `GET /api/v1/sessions/{id}`."""
        compression = state.get("compression") or {}
        return cls(
            turn_index=int(state.get("turn_count", 0)),
            summary=compression.get("summary") or "",
            anchors=tuple(a["text"] for a in compression.get("anchors") or []),
            metadata=state.get("project_metadata") or {},
        )

    def text_of(self, where: str) -> str:
        if where == "summary":
            return self.summary
        if where == "anchors":
            return "\n".join(self.anchors)
        if where == "metadata":
            return json.dumps(self.metadata, ensure_ascii=False)
        raise ValueError(f"Fuente desconocida para MemoryDriftMetric: {where!r} (usa summary, anchors o metadata)")


class LatencyBudgetMetric:
    """1.0 si latency_ms ≤ budget_ms; 0.0 si no."""

    name = "latency_budget"

    def __init__(self, budget_ms: int):
        self.budget_ms = budget_ms

    def evaluate(self, observation: dict) -> MetricResult:
        latency = observation.get("latency_ms")
        passed = latency is not None and latency <= self.budget_ms
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details={"latency_ms": latency, "budget_ms": self.budget_ms, "ratio": round(latency / self.budget_ms, 3) if latency is not None else None},
        )


class CostBudgetMetric:
    """1.0 si cost_usd ≤ budget_usd; 0.0 si no."""

    name = "cost_budget"

    def __init__(self, budget_usd: float):
        self.budget_usd = budget_usd

    def evaluate(self, observation: dict) -> MetricResult:
        cost = observation.get("cost_usd")
        passed = cost is not None and cost <= self.budget_usd
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details={"cost_usd": cost, "budget_usd": self.budget_usd, "ratio": round(cost / self.budget_usd, 3) if cost is not None and self.budget_usd else None},
        )


class MemoryDriftMetric:
    """1.0 si el fact declarado del turno k aparece en summary, anchors o ProjectMetadata del turno N (N > k); 0.0 si no."""

    name = "memory_drift"

    def __init__(self, fact: str, where: Sequence[str] = DEFAULT_WHERE, declared_at: int | None = None):
        if not fact or not fact.strip():
            raise ValueError("MemoryDriftMetric necesita un hecho no vacío")
        self.fact = fact
        self.where = tuple(where)
        self.declared_at = declared_at

    def evaluate(self, session_snapshot: SessionSnapshot) -> MetricResult:
        found_in = [where for where in self.where if contains(session_snapshot.text_of(where), self.fact)]
        passed = bool(found_in)
        details = {"fact": self.fact, "found_in": found_in, "searched": list(self.where), "turn_index": session_snapshot.turn_index}
        if self.declared_at is not None:
            details["declared_at"] = self.declared_at
            if self.declared_at >= session_snapshot.turn_index:
                details["note"] = "el hecho es del turno actual o posterior: la métrica solo tiene sentido con N > k"
        return MetricResult(name=self.name, score=1.0 if passed else 0.0, passed=passed, details=details)


def recall_of_facts(snapshot: SessionSnapshot, facts: Sequence[tuple[int, str]], where: Sequence[str] = DEFAULT_WHERE) -> tuple[float | None, list[str], list[str]]:
    """Recall medio del fact-tracker en un turno: (score medio o None si no hay hechos, recordados, perdidos)."""
    if not facts:
        return None, [], []
    results = [(fact, MemoryDriftMetric(fact, where, declared_at=turn).evaluate(snapshot)) for turn, fact in facts]
    recalled = [fact for fact, result in results if result.passed]
    missing = [fact for fact, result in results if not result.passed]
    return round(len(recalled) / len(results), 4), recalled, missing
