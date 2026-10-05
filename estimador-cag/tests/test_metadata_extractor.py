"""Tests del extractor heurístico de project_metadata."""

from app.schemas.estimation import EstimationRequest
from app.services.attachments import ExtractedAttachment
from app.services.metadata_extractor import extract_project_metadata, find_technologies, parse_hours, team_size
from app.services.sessions import ProjectMetadata
from tests.fakes import estimacion_falsa

TRANSCRIPCION = "El cliente quiere una tienda online con pagos por Stripe y el catálogo en Excel. Hosting en AWS."


def test_extrae_los_hechos_de_la_estimacion_y_de_los_parametros():
    request = EstimationRequest(transcription=TRANSCRIPCION, project_type="web", hourly_rate_eur=60, deadline_weeks=8)
    adjunto = ExtractedAttachment("spec.pdf", "pdf", "Integración con HubSpot y WordPress.", 10, 1, False, None)

    metadata = extract_project_metadata(request, estimacion_falsa("Panadería online", con_pagos=True), [adjunto])

    assert metadata.project_name == "Panadería online"  # del título, porque la memoria aún no tiene nombre
    assert metadata.project_type == "web"
    assert metadata.assumed_team == "1 desarrollador full-stack, 1 diseñador UX/UI"
    assert metadata.assumed_team_size == 2
    assert metadata.mentioned_technologies == ["Stripe", "Excel", "AWS", "HubSpot", "WordPress"]
    assert metadata.agreed_scope == "Tienda online para una panadería artesanal con pasarela de pago Stripe."
    assert metadata.hourly_rate_eur == 60
    assert metadata.deadline_weeks == 8
    assert metadata.last_total_hours == 108
    assert metadata.attached_documents == ["spec.pdf"]


def test_el_nombre_lo_fija_el_formulario_o_el_primer_turno_y_no_deriva():
    sin_nombre = EstimationRequest(transcription=TRANSCRIPCION)
    memoria = ProjectMetadata(project_name="Tienda online La Espiga")

    # El modelo reformula el título en el turno siguiente: la memoria no lo sigue
    update = extract_project_metadata(sin_nombre, estimacion_falsa("Forn de la Plaça", con_pagos=False), current=memoria)
    assert update.project_name is None
    assert memoria.merged(update).project_name == "Tienda online La Espiga"

    # El formulario sí lo cambia
    con_nombre = EstimationRequest(transcription=TRANSCRIPCION, project_name="Nuevo nombre")
    update = extract_project_metadata(con_nombre, estimacion_falsa("Otro título", con_pagos=False), current=memoria)
    assert memoria.merged(update).project_name == "Nuevo nombre"


def test_una_estimacion_sin_formato_no_rompe_la_extraccion():
    metadata = extract_project_metadata(EstimationRequest(transcription=TRANSCRIPCION), "Lo siento, no puedo estimar esto.")

    assert metadata.project_name is None
    assert metadata.assumed_team is None and metadata.assumed_team_size is None
    assert metadata.agreed_scope is None
    assert metadata.last_total_hours is None
    assert metadata.mentioned_technologies == ["Stripe", "Excel", "AWS"]  # de la transcripción


def test_find_technologies_limites_de_palabra_y_nombres_largos_primero():
    assert find_technologies(["Usarán React Native para la app y React para la web."]) == ["React Native", "React"]
    assert find_technologies(["Solo React Native."]) == ["React Native"]
    assert find_technologies(["Programado en Java, no en JavaScript."]) == ["Java", "JavaScript"]
    assert find_technologies(["javascript y PYTHON"]) == ["JavaScript", "Python"]  # nombre canónico
    assert find_technologies(["El resto del equipo descansa."]) == []  # 'rest' no es REST
    assert find_technologies(["Node.js y Next.js en Vercel"]) == ["Node.js", "Next.js", "Vercel"]


def test_team_size():
    assert team_size("1 desarrollador full-stack, 1 diseñador UX/UI") == 2
    assert team_size("2 desarrolladores backend + 1 frontend + 1 QA (part-time)") == 4
    assert team_size("un desarrollador senior y una diseñadora") == 2
    assert team_size("Desarrollador full-stack, diseñador UX y QA") == 3
    assert team_size("Desarrollador full-stack senior") == 1


def test_parse_hours():
    assert parse_hours("80") == 80
    assert parse_hours("1.250") == 1250
    assert parse_hours("47,5") == 47.5
    assert parse_hours("264.") == 264
    assert parse_hours(None) is None
    assert parse_hours("muchas") is None
