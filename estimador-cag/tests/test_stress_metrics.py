"""Las tres métricas del stress (sesión 06, bloque 4): una que pasa, una que falla y una con datos límite, por métrica."""

import pytest

from evals.metrics import MetricResult, run_all_metrics
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, MemoryDriftMetric, SessionSnapshot, recall_of_facts


# --- LatencyBudgetMetric ------------------------------------------------------------------------------


def test_latency_dentro_del_presupuesto_pasa():
    result = LatencyBudgetMetric(budget_ms=4000).evaluate({"latency_ms": 2500})

    assert isinstance(result, MetricResult)
    assert result.name == "latency_budget"
    assert result.score == 1.0 and result.passed is True
    assert result.details == {"latency_ms": 2500, "budget_ms": 4000, "ratio": 0.625}


def test_latency_por_encima_del_presupuesto_falla():
    result = LatencyBudgetMetric(budget_ms=4000).evaluate({"latency_ms": 4001})

    assert result.score == 0.0 and result.passed is False


def test_latency_en_el_limite_exacto_pasa_y_sin_dato_falla():
    assert LatencyBudgetMetric(budget_ms=4000).evaluate({"latency_ms": 4000}).passed is True
    assert LatencyBudgetMetric(budget_ms=4000).evaluate({"latency_ms": 0}).passed is True
    sin_dato = LatencyBudgetMetric(budget_ms=4000).evaluate({})
    assert sin_dato.passed is False and sin_dato.details["ratio"] is None


# --- CostBudgetMetric ---------------------------------------------------------------------------------


def test_coste_dentro_del_presupuesto_pasa():
    result = CostBudgetMetric(budget_usd=0.01).evaluate({"cost_usd": 0.0042})

    assert result.name == "cost_budget"
    assert result.passed is True and result.score == 1.0
    assert result.details["ratio"] == 0.42


def test_coste_por_encima_del_presupuesto_falla():
    result = CostBudgetMetric(budget_usd=0.01).evaluate({"cost_usd": 0.0101})

    assert result.passed is False and result.score == 0.0


def test_coste_cero_de_un_acierto_de_cache_y_coste_exacto_pasan():
    assert CostBudgetMetric(budget_usd=0.01).evaluate({"cost_usd": 0.0}).passed is True
    assert CostBudgetMetric(budget_usd=0.01).evaluate({"cost_usd": 0.01}).passed is True
    assert CostBudgetMetric(budget_usd=0.01).evaluate({"cost_usd": None}).passed is False


# --- MemoryDriftMetric --------------------------------------------------------------------------------


SNAPSHOT = SessionSnapshot(
    turn_index=10,
    summary="Turno 3: el cliente fijó un presupuesto cerrado de 30.000 € para la primera fase.",
    anchors=("[turno 8] Presupuesto: 80.000 €", "[turno 5] Stack: Flutter"),
    metadata={"project_name": "Nimbus", "mentioned_technologies": ["React Native", "Flutter"]},
)


def test_el_hecho_se_encuentra_en_el_resumen_las_anclas_o_la_memoria():
    en_resumen = MemoryDriftMetric("30.000", declared_at=3).evaluate(SNAPSHOT)
    en_anclas = MemoryDriftMetric("80.000").evaluate(SNAPSHOT)
    en_memoria = MemoryDriftMetric("nimbus").evaluate(SNAPSHOT)  # case-insensitive

    assert en_resumen.name == "memory_drift"
    assert en_resumen.passed and en_resumen.score == 1.0 and en_resumen.details["found_in"] == ["summary"]
    assert en_resumen.details["declared_at"] == 3 and "note" not in en_resumen.details
    assert en_anclas.details["found_in"] == ["anchors"]
    assert en_memoria.details["found_in"] == ["metadata"]


def test_el_hecho_ausente_falla_con_cero():
    result = MemoryDriftMetric("Kubernetes").evaluate(SNAPSHOT)

    assert result.passed is False and result.score == 0.0
    assert result.details["found_in"] == [] and result.details["searched"] == ["summary", "anchors", "metadata"]


def test_where_restringe_donde_buscar_y_los_limites():
    solo_memoria = MemoryDriftMetric("30.000", where=["metadata"]).evaluate(SNAPSHOT)
    assert solo_memoria.passed is False

    mismo_turno = MemoryDriftMetric("Nimbus", declared_at=10).evaluate(SNAPSHOT)
    assert mismo_turno.passed is True and "N > k" in mismo_turno.details["note"]

    with pytest.raises(ValueError):
        MemoryDriftMetric("   ")
    with pytest.raises(ValueError):
        MemoryDriftMetric("x", where=["historial"]).evaluate(SNAPSHOT)


def test_snapshot_desde_la_respuesta_del_endpoint_y_recall_medio():
    state = {
        "turn_count": 4,
        "project_metadata": {"project_name": "Faro", "mentioned_technologies": ["Django"]},
        "compression": {"summary": "Turno 1: plataforma de cursos.", "anchors": [{"kind": "budget", "text": "Presupuesto: 30.000 €", "turn": 3}]},
    }
    snapshot = SessionSnapshot.from_api(state)

    assert snapshot.turn_index == 4
    assert snapshot.anchors == ("Presupuesto: 30.000 €",)
    recall, recalled, missing = recall_of_facts(snapshot, [(1, "Faro"), (1, "Django"), (3, "30.000"), (2, "12 semanas")])
    assert recall == 0.75 and recalled == ["Faro", "Django", "30.000"] and missing == ["12 semanas"]
    assert recall_of_facts(snapshot, []) == (None, [], [])


def test_run_all_metrics_combina_las_tres_sobre_un_turno():
    observation = {"latency_ms": 3200, "cost_usd": 0.002}
    results = run_all_metrics(observation, [LatencyBudgetMetric(4000), CostBudgetMetric(0.01)])

    assert [r.name for r in results] == ["latency_budget", "cost_budget"]
    assert all(r.passed for r in results)
    # Una métrica que explota no interrumpe el eval: cuenta como fallida con el error en details
    roto = run_all_metrics(observation, [MemoryDriftMetric("x")])  # espera un snapshot, recibe un dict
    assert roto[0].passed is False and "error" in roto[0].details
