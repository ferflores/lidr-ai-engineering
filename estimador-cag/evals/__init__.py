"""Evals del estimador: casos golden (`golden_dataset.json`), métricas binarias (`metrics.py`), runner (`run.py`)
y, en `stress/`, el ejercicio de carga de la sesión 06 (escenarios multi-turno, adjuntos grandes, métricas de
presupuesto y deriva de memoria, runner y reporte)."""

from evals.metrics import (
    ContentRecallMetric,
    CostBoundsMetric,
    EvalObservation,
    Metric,
    MetricResult,
    SchemaAdherenceMetric,
    contains,
    default_metrics,
    run_all_metrics,
    summarize,
)

__all__ = [
    "ContentRecallMetric",
    "CostBoundsMetric",
    "EvalObservation",
    "Metric",
    "MetricResult",
    "SchemaAdherenceMetric",
    "contains",
    "default_metrics",
    "run_all_metrics",
    "summarize",
]
