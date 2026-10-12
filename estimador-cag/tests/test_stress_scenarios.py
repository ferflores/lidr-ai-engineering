"""Escenarios del stress, su fact-tracker y el corpus de PDFs sintéticos."""

import pytest

from app.services.attachments import extract_text
from evals.stress.fixtures.build_pdfs import MARKER_TERMS, MARKERS, build_pdf, build_text_lines
from evals.stress.scenarios import CHECKPOINTS, SCENARIOS, get_scenarios


def test_los_tres_escenarios_tienen_20_turnos_bien_formados():
    assert list(SCENARIOS) == ["growing", "pivot", "contradiction"]
    assert CHECKPOINTS == (1, 3, 6, 10, 20)
    for scenario in SCENARIOS.values():
        assert len(scenario) == 20
        assert [t.turn_index for t in scenario.turns] == list(range(1, 21))
        assert scenario.turns[0].params["project_name"] == scenario.project_name
        for turn in scenario.turns:
            assert len(turn.transcript) >= 20
            for fact in turn.facts:  # los hechos se buscan literalmente: tienen que estar en la transcripción
                assert fact.casefold() in turn.transcript.casefold(), f"{scenario.key} t{turn.turn_index}: {fact!r}"


def test_el_fact_tracker_acumula_y_retira_hechos():
    contradiction = SCENARIOS["contradiction"]
    en_turno_5 = contradiction.facts_expected_at(5)
    assert (3, "30.000") in en_turno_5 and (2, "12 semanas") in en_turno_5
    assert all(turn < 5 for turn, _ in en_turno_5)

    en_turno_9 = dict((f, t) for t, f in contradiction.facts_expected_at(9))
    assert "80.000" in en_turno_9 and "30.000" not in en_turno_9  # retirado por el turno 8
    assert contradiction.retired_before(9) == ["30.000"] and contradiction.retired_before(13) == ["30.000", "12 semanas"]
    assert contradiction.facts_expected_at(1) == []

    pivot = SCENARIOS["pivot"]
    assert "React Native" not in {f for _, f in pivot.facts_expected_at(6)}
    assert "Flutter" in {f for _, f in pivot.facts_expected_at(6)}


def test_get_scenarios_valida_las_claves():
    assert [s.key for s in get_scenarios(["pivot", "growing"])] == ["pivot", "growing"]
    with pytest.raises(KeyError):
        get_scenarios(["inexistente"])


def test_los_pdfs_sinteticos_estan_calibrados_y_llevan_los_marcadores_en_orden():
    lines = build_text_lines(5000)
    text = "\n".join(lines)
    assert 5000 <= len(text) <= 5800
    assert all(len(line) <= 85 for line in lines)
    positions = [text.index(sentence) for _, sentence, _ in MARKERS]
    assert positions == sorted(positions)
    assert MARKER_TERMS == ("Redsys", "Power BI", "Azure", "SAP", "catalán")

    pdf = build_pdf(5)
    extracted = extract_text("attach_5kb.pdf", pdf, max_chars=60_000)
    assert extracted.pages == 2 and 4500 <= extracted.chars <= 6000 and not extracted.truncated
    assert "Redsys" in extracted.text and "catalan" in extracted.text  # el PDF mínimo es WinAnsi sin tildes

    truncado = extract_text("attach_100kb.pdf", build_pdf(100), max_chars=60_000)
    assert truncado.truncated and truncado.chars <= 60_100
    assert "Azure" in truncado.text and "SAP" not in truncado.text  # el 75 % del documento ya no entra
