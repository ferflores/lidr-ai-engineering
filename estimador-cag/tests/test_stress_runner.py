"""El runner del stress de punta a punta (in-process, LLM simulado): CSV con una fila por turno y REPORT.md con sus secciones."""

import csv

import pytest

from app.main import app
from app.services import llm_service
from app.services.response_cache import get_response_cache
from evals.stress import report
from evals.stress.run import COLUMNS, main

pytestmark = pytest.mark.anyio


@pytest.fixture
def runner_env():
    original = llm_service.openai.AsyncOpenAI
    get_response_cache().clear()
    yield
    llm_service.openai.AsyncOpenAI = original
    app.dependency_overrides.clear()
    get_response_cache().clear()


async def test_el_runner_produce_el_csv_y_el_reporte(tmp_path, runner_env, capsys):
    output = tmp_path / "results.csv"
    report_path = tmp_path / "REPORT.md"

    code = await main([
        "--fake", "--scenarios", "contradiction", "--attachment-sizes", "0,5", "--repeats", "2", "--turns", "4",
        "--latency-budget-ms", "4000", "--cost-budget-usd", "0.01", "--output", str(output), "--report", str(report_path),
    ])

    assert code == 0
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(COLUMNS)
    assert len(rows) == 1 * 2 * 2 * 4  # escenarios × tamaños × repeticiones × turnos
    primera = rows[0]
    assert primera["scenario"] == "contradiction" and primera["attachment_kb"] == "0" and primera["turn_index"] == "1"
    assert primera["cache_hit_kind"] == "none" and int(primera["tokens_in"]) > 0
    assert primera["latency_budget_pass"] == "True" and primera["cost_budget_pass"] == "True"
    assert primera["memory_drift_score"] == "" and primera["facts_expected"] == "0"
    # En el turno 4 hay hechos declarados antes y el fact-tracker los busca
    cuarto = rows[3]
    assert int(cuarto["facts_expected"]) >= 4 and cuarto["memory_drift_score"] != ""
    assert "30.000@anchors" in cuarto["facts_found_in"]
    # La segunda repetición es idéntica: acierta en la caché exacta y cuesta 0
    segunda = rows[4]
    assert segunda["repeat"] == "2" and segunda["cache_hit_kind"] == "exact" and float(segunda["cost_usd"]) == 0
    # Con adjunto: caracteres de adjunto > 0 en el turno 1 y recall de marcadores medido
    con_adjunto = [r for r in rows if r["attachment_kb"] == "5" and r["turn_index"] == "1"][0]
    assert int(con_adjunto["attachments_total_chars"]) > 4000 and con_adjunto["attachment_recall"] != ""
    assert (output.with_suffix(".meta.json")).exists()

    texto = report_path.read_text(encoding="utf-8")
    for section in ("## 1. Tabla resumen", "## 2. Curva 1", "## 3. Curva 2", "## 4. Curva 3", "## 5. Adjuntos grandes", "## 6. Lectura", "LLM simulado"):
        assert section in texto
    assert "| contradiction | 0 KB | 8 |" in texto

    # El reporte se regenera desde el CSV con el mismo resultado
    regenerated = tmp_path / "REPORT2.md"
    report.main(["--input", str(output), "--output", str(regenerated)])
    assert regenerated.read_text(encoding="utf-8").split("_Generado")[1].split("._", 1)[1] == texto.split("_Generado")[1].split("._", 1)[1]
    assert "filas" in capsys.readouterr().out
