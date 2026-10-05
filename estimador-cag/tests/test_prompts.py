"""Tests del registro de plantillas de prompt: `prompts/estimacion/<versión>/` cargado por app/services/prompts.py."""

import pytest

from app.context.examples import ESTIMATION_EXAMPLES, render_examples
from app.schemas.estimation import EstimationRequest
from app.services import prompts
from app.services.prompts import (
    PromptRenderError,
    PromptTemplateNotFound,
    get_template,
    latest_version,
    list_versions,
    load_template,
)
from tests.conftest import make_settings

TRANSCRIPCION = "El cliente quiere una landing page con formulario de contacto e integración con HubSpot."


def _variables(**overrides) -> dict:
    request = EstimationRequest(transcription=TRANSCRIPCION, **overrides)
    return {**request.prompt_variables(), "examples": render_examples()}


# --- Registro ----------------------------------------------------------------------


def test_hay_versiones_y_la_ultima_es_la_mayor():
    versiones = list_versions()

    assert versiones[:3] == ["v1", "v2", "v3"]
    assert latest_version() == versiones[-1]


def test_la_version_configurada_por_defecto_existe():
    settings = make_settings()

    assert get_template(settings).version == settings.prompt_version == "v3"
    assert get_template(settings, "v1").version == "v1"


def test_metadatos_de_la_plantilla():
    template = load_template("estimacion", "v2")

    assert template.id == "estimacion/v2"
    assert template.description
    assert template.created is not None
    assert len(template.changelog) >= 1
    assert "examples" in template.system_variables
    assert "transcription" in template.user_variables
    assert "Eres un estimador" in template.system_source
    assert "{{ transcription }}" in template.user_source


def test_version_inexistente():
    with pytest.raises(PromptTemplateNotFound, match="v99"):
        load_template("estimacion", "v99")
    with pytest.raises(PromptTemplateNotFound):
        list_versions("no-existe")


def test_filtro_num():
    assert prompts._num(50.0) == "50"
    assert prompts._num(47.5) == "47.5"
    assert prompts._num(1000.0) == "1000"


# --- v1: el prompt original ----------------------------------------------------------


def test_v1_es_el_prompt_original():
    template = load_template("estimacion", "v1")

    system = template.render_system(**_variables())
    assert system.startswith("Eres un estimador de software senior")
    assert "Tarifa de referencia para el coste: 50 €/hora." in system
    assert ESTIMATION_EXAMPLES[0]["meeting_summary"] in system

    user = template.render_user(**_variables())
    assert user == (
        "Esta es la transcripción de la reunión con el cliente. Genera la estimación siguiendo el formato indicado."
        f"\n\n<transcripcion>\n{TRANSCRIPCION}\n</transcripcion>"
    )


# --- v2: parámetros tipados en el mensaje de usuario ----------------------------------


def test_v2_pone_los_parametros_en_el_mensaje_de_usuario():
    template = load_template("estimacion", "v2")
    variables = _variables(
        project_name="Landing HubSpot",
        project_type="web",
        hourly_rate_eur=47.5,
        deadline_weeks=4,
        estimation_mode="optimista",
        detail_level="resumen",
    )

    system = template.render_system(**variables)
    user = template.render_user(**variables)

    # El system prompt es el contexto fijo: instrucciones + ejemplos, sin datos de la petición
    assert ESTIMATION_EXAMPLES[0]["meeting_summary"] in system
    assert "Landing HubSpot" not in system
    assert "47.5" not in system
    for esperado in (
        "Nombre del proyecto: Landing HubSpot",
        "Tipo de proyecto: aplicación web",
        "Tarifa: 47.5 €/hora",
        "Plazo objetivo: 4 semanas",
        "Enfoque: optimista",
        "Nivel de detalle: resumen",
        f"<transcripcion>\n{TRANSCRIPCION}\n</transcripcion>",
    ):
        assert esperado in user


def test_v2_parametros_opcionales_ausentes():
    user = load_template("estimacion", "v2").render_user(**_variables())

    assert "Nombre del proyecto: no indicado" in user
    assert "Tipo de proyecto: no indicado" in user
    assert "Tarifa: 50 €/hora" in user
    assert "Plazo objetivo: no indicado" in user
    assert "Enfoque: conservadora" in user
    assert "Nivel de detalle: completa" in user


def test_v2_el_system_prompt_no_cambia_entre_peticiones():
    template = load_template("estimacion", "v2")

    a = template.render_system(**_variables())
    b = template.render_system(**_variables(project_name="Otro", hourly_rate_eur=90, deadline_weeks=2, detail_level="resumen"))

    assert a == b


# --- Contrato de variables --------------------------------------------------------------


def test_faltan_variables_declaradas():
    template = load_template("estimacion", "v2")

    with pytest.raises(PromptRenderError, match="transcription"):
        template.render_user(examples="")
    with pytest.raises(PromptRenderError, match="project_metadata"):
        load_template("estimacion", "v3").render_system(examples="")


# --- v3: conversación con memoria ------------------------------------------------------------


def test_v3_declara_project_metadata_y_el_bloque_va_vacio_sin_memoria():
    template = load_template("estimacion", "v3")
    assert "project_metadata" in template.system_variables
    assert template.user_variables == load_template("estimacion", "v2").user_variables

    system = template.render_system(**_variables(), project_metadata=None)

    assert "## Conversación con memoria" in system
    assert system.rstrip().endswith("<project_metadata>\n</project_metadata>")
    assert "--- attachment: <fichero> ---" in system


def test_v3_inyecta_la_memoria_al_final_sin_tocar_el_prefijo():
    from app.services.sessions import ProjectMetadata

    template = load_template("estimacion", "v3")
    memoria = ProjectMetadata(
        project_name="Panadería online",
        project_type="web",
        agreed_scope="Tienda online con recogida y reparto.",
        mentioned_technologies=["Stripe", "WordPress"],
        assumed_team="1 desarrollador full-stack, 1 diseñador",
        assumed_team_size=2,
        last_total_hours=250,
        hourly_rate_eur=50,
        deadline_weeks=8,
        attached_documents=["especificacion.pdf"],
    )

    vacio = template.render_system(**_variables(), project_metadata=None)
    lleno = template.render_system(**_variables(), project_metadata=memoria.prompt_variables())

    bloque = lleno[lleno.index("<project_metadata>"):]
    for esperado in (
        "- Nombre del proyecto: Panadería online",
        "- Tipo de proyecto: web",
        "- Alcance acordado hasta ahora: Tienda online con recogida y reparto.",
        "- Tecnologías mencionadas: Stripe, WordPress",
        "- Equipo asumido: 1 desarrollador full-stack, 1 diseñador (2 personas)",
        "- Última estimación: 250 horas",
        "- Última tarifa: 50 €/hora",
        "- Último plazo objetivo: 8 semanas",
        "- Documentos adjuntados en la sesión: especificacion.pdf",
    ):
        assert esperado in bloque
    # El prefijo (rol, reglas, formato, ejemplos CAG) es idéntico: solo cambia la cola -> cacheable
    assert vacio.split("## Proyecto en curso")[0] == lleno.split("## Proyecto en curso")[0]
    assert "Panadería online" not in vacio


def test_variable_usada_pero_no_declarada_falla(tmp_path, monkeypatch):
    carpeta = tmp_path / "prueba" / "v1"
    carpeta.mkdir(parents=True)
    (carpeta / "template.toml").write_text('[prompt]\nname = "prueba"\nversion = "v1"\n', encoding="utf-8")
    (carpeta / "system.md.j2").write_text("Hola {{ nombre }}", encoding="utf-8")
    (carpeta / "user.md.j2").write_text("{{ transcription }}", encoding="utf-8")
    monkeypatch.setattr(prompts, "PROMPTS_DIR", tmp_path)
    prompts._environment.cache_clear()
    prompts.load_template.cache_clear()
    try:
        template = prompts.load_template("prueba", "v1")
        with pytest.raises(PromptRenderError, match="nombre"):
            template.render_system()
        assert template.render_system(nombre="Fer") == "Hola Fer"
    finally:
        prompts._environment.cache_clear()
        prompts.load_template.cache_clear()
