"""Anclas heurísticas: qué se extrae de un mensaje del usuario y cómo se fusionan entre turnos."""

from app.schemas.estimation import EstimationRequest
from app.services.anchors import anchors_text, extract_anchors, merge_anchors
from app.services.sessions import Anchor

TRANSCRIPCION = (
    "Reunión inicial. El proyecto se llama Nimbus: una plataforma web para gestionar una flota. "
    "El presupuesto es cerrado, 30.000 € como máximo, y quieren lanzar en 12 semanas. "
    "Lo harán en React con PostgreSQL. Necesitamos autenticación con Google mediante OAuth. "
    "Cambio importante: en vez de React Native usaremos Flutter para la app."
)


def _kinds(anchors):
    return [(a.kind, a.text) for a in anchors]


def test_extrae_nombre_presupuesto_plazo_cambio_stack_y_requisitos_por_prioridad():
    anchors = extract_anchors(TRANSCRIPCION, None, turn=1, per_turn=20)

    textos = _kinds(anchors)
    assert textos[0] == ("name", "Proyecto: Nimbus")
    assert ("budget", "Presupuesto: 30.000 €") in textos
    assert ("deadline", "Plazo: 12 semanas") in textos
    assert ("change", "Cambio: Cambio importante: en vez de React Native usaremos Flutter para la app.") in textos
    assert ("stack", "Stack: React Native") in textos and ("stack", "Stack: Flutter") in textos and ("stack", "Stack: PostgreSQL") in textos
    assert ("requirement", "Requisito: Necesitamos autenticación con Google mediante OAuth.") in textos
    # La frase de cambio no se duplica como requisito, y el orden sigue la prioridad por tipo
    assert not any(k == "requirement" and "en vez de" in t for k, t in textos)
    kinds = [k for k, _ in textos]
    assert kinds == sorted(kinds, key=["name", "budget", "deadline", "change", "stack", "requirement"].index)
    assert all(a.turn == 1 for a in anchors)


def test_el_formulario_manda_sobre_la_transcripcion_y_el_plazo_del_formulario_es_el_respaldo():
    request = EstimationRequest(transcription="Reunión sin datos concretos sobre nada.", project_name="Atlas", deadline_weeks=8)
    anchors = extract_anchors("Reunión sin datos concretos sobre nada.", request, turn=2)

    assert _kinds(anchors) == [("name", "Proyecto: Atlas"), ("deadline", "Plazo objetivo: 8 semanas")]


def test_tope_por_turno_y_recorte_de_frases_largas():
    larga = "Necesitamos " + "una cosa más, " * 30 + "y ya."
    anchors = extract_anchors(larga + " Hay que añadir otra. Quieren una tercera.", None, turn=3, per_turn=2)

    assert len(anchors) == 2
    assert all(len(a.text) <= 160 for a in anchors)
    assert anchors[0].text.endswith("…")


def test_merge_sustituye_los_de_un_solo_valor_y_acumula_el_resto_sin_duplicados():
    current = [Anchor("budget", "Presupuesto: 30.000 €", 3), Anchor("requirement", "Requisito: foros por curso.", 5)]
    nuevos = [Anchor("budget", "Presupuesto: 80.000 €", 8), Anchor("requirement", "requisito: FOROS por curso.", 8), Anchor("stack", "Stack: Zoom", 8)]

    merged = merge_anchors(current, nuevos, max_anchors=24)

    assert _kinds(merged) == [("requirement", "Requisito: foros por curso."), ("budget", "Presupuesto: 80.000 €"), ("stack", "Stack: Zoom")]
    assert merge_anchors(current, nuevos, max_anchors=0) == []


def test_al_superar_el_tope_caen_las_acumulables_mas_antiguas_nunca_las_de_un_solo_valor():
    current = [Anchor("name", "Proyecto: Faro", 1)] + [Anchor("requirement", f"Requisito: cosa {i}", i) for i in range(1, 6)]

    merged = merge_anchors(current, [Anchor("requirement", "Requisito: cosa nueva", 9)], max_anchors=4)

    assert [a.text for a in merged] == ["Proyecto: Faro", "Requisito: cosa 4", "Requisito: cosa 5", "Requisito: cosa nueva"]
    assert anchors_text(merged).splitlines()[0] == "[turno 1] Proyecto: Faro"
