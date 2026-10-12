"""Framework de evals de la sesión 05: MetricResult, las tres métricas de una estimación suelta y el dataset golden."""

import json
from pathlib import Path

from evals import ContentRecallMetric, CostBoundsMetric, EvalObservation, SchemaAdherenceMetric, default_metrics, run_all_metrics, summarize
from evals.run import GOLDEN, load_cases
from tests.fakes import estimacion_falsa


def _obs(**overrides) -> EvalObservation:
    values = dict(case_id="caso", estimation=estimacion_falsa("Panadería online", con_pagos=True), detail_level="completa", cost_usd=0.0008, expected_terms=("Panadería online", "Stripe"), max_cost_usd=0.01)
    values.update(overrides)
    return EvalObservation(**values)


def test_schema_adherence_pasa_con_el_formato_completo():
    result = SchemaAdherenceMetric().evaluate(_obs())

    assert result.passed and result.score == 1.0
    assert result.details["missing"] == [] and result.details["total_parsed"] is True


def test_schema_adherence_detecta_secciones_que_faltan_y_prohibidas():
    sin_total = _obs(estimation=estimacion_falsa("X", False).replace("**Total estimado: 84 horas**", ""))
    result = SchemaAdherenceMetric().evaluate(sin_total)
    assert not result.passed and "**Total estimado:" in result.details["missing"] and result.details["total_parsed"] is False
    assert 0 < result.score < 1

    # En `resumen` la sección de preguntas pendientes está prohibida y el desglose es por bloques
    resumen = SchemaAdherenceMetric().evaluate(_obs(detail_level="resumen"))
    assert not resumen.passed
    assert "### Preguntas pendientes" in resumen.details["forbidden_present"]
    assert "### Desglose por bloques" in resumen.details["missing"]


def test_cost_bounds_usa_el_tope_del_caso_o_el_global():
    assert CostBoundsMetric(max_usd=0.0001).evaluate(_obs()).passed is True  # el caso declara 0.01
    assert CostBoundsMetric(max_usd=0.0001).evaluate(_obs(max_cost_usd=None)).passed is False
    assert CostBoundsMetric().evaluate(_obs(cost_usd=None)).passed is False
    assert CostBoundsMetric().evaluate(_obs(cost_usd=0.01)).details["ratio"] == 1.0


def test_content_recall_fraccion_de_terminos_y_umbral():
    completo = ContentRecallMetric().evaluate(_obs())
    assert completo.passed and completo.score == 1.0

    parcial = ContentRecallMetric(threshold=0.5).evaluate(_obs(expected_terms=("Panadería online", "Kubernetes")))
    assert parcial.passed and parcial.score == 0.5 and parcial.details["missing"] == ["Kubernetes"]
    assert ContentRecallMetric(threshold=1.0).evaluate(_obs(expected_terms=("Kubernetes",))).passed is False
    assert ContentRecallMetric().evaluate(_obs(expected_terms=())).passed is True


def test_run_all_metrics_y_summarize():
    results = run_all_metrics(_obs(), default_metrics())
    assert [r.name for r in results] == ["schema_adherence", "cost_bounds", "content_recall"]
    assert all(r.passed for r in results)

    totals = summarize([results, run_all_metrics(_obs(cost_usd=0.5), default_metrics())])
    assert totals["cost_bounds"] == {"cases": 2, "pass_rate": 0.5, "mean_score": 0.5}
    assert totals["schema_adherence"]["pass_rate"] == 1.0


def test_el_dataset_golden_tiene_16_casos_bien_formados():
    data = json.loads(Path(GOLDEN).read_text(encoding="utf-8"))
    cases = load_cases()

    assert len(cases) == 16 and len({c.id for c in cases}) == 16
    for case in cases:
        assert len(case.transcript) >= 20
        assert case.expected_terms, f"{case.id} no declara términos esperados"
        assert case.max_cost_usd == data["default_max_cost_usd"]
        assert case.detail_level in ("completa", "resumen")
    assert [c.id for c in load_cases(only={"landing-hubspot"})] == ["landing-hubspot"]
