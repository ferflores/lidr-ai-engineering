# lidr-ai-engineering

Proyectos y ejercicios de Fer para el curso **AI Engineering** de [Lidr](https://lidr.co).

## Proyectos

| Sesión | Proyecto | Qué es | Rama |
|---|---|---|---|
| 02 | [`estimador-cag/`](estimador-cag/) | Servicio FastAPI que recibe la transcripción de una reunión y devuelve una estimación de software generada por un LLM, con arquitectura CAG (contexto estático inyectado en el prompt). Se ejecuta con `uv` o con `docker compose`. | `session-02-estimador-cag` |
| 03 | [`estimador-cag/streamlit_app.py`](estimador-cag/streamlit_app.py) | Interfaz de chat con Streamlit sobre el mismo proyecto: estimación en streaming, historial de conversación y panel con el contexto CAG y las métricas de tokens y coste. La rama incluye todo lo de la sesión 02. | `session-03-streamlit-chat` |
| 04 | [`estimador-cag/prompts/`](estimador-cag/prompts/) | Formulario con parámetros tipados y prompts como plantillas versionadas (`v1`, `v2`). | `session-04-formulario-prompts` |
| 05 | [`estimador-cag/app/services/sessions.py`](estimador-cag/app/services/sessions.py) | Sesiones con memoria conversacional (ventana deslizante + `project_metadata`), adjuntos PDF/Word y, del directo, compresión del historial (anclas + resumen acumulativo), tier dinámico, caché de respuestas y el evento `turn_observed`. | `pre-session-05`, `main` |
| 06 | [`estimador-cag/evals/stress/REPORT.md`](estimador-cag/evals/stress/REPORT.md) | Pre-work: framework de evals (`evals/`) y stress del CAG (escenarios multi-turno, adjuntos grandes, ráfagas) con `results.csv` y el reporte con las tres curvas. | `session-06-stress-cag` |

## Integración continua

[![estimador-cag](https://github.com/ferflores/lidr-ai-engineering/actions/workflows/estimador-cag.yml/badge.svg?branch=session-03-streamlit-chat)](https://github.com/ferflores/lidr-ai-engineering/actions/workflows/estimador-cag.yml)

Cada push que toque `estimador-cag/` ejecuta `.github/workflows/estimador-cag.yml`, que valida la
estructura del proyecto, pasa los tests, arranca la API y la interfaz de chat para comprobar que
responden (ver `estimador-cag/scripts/verificar.sh`), y después construye la imagen Docker y levanta
los dos servicios con `docker compose` para comprobar `/health` y `/_stcore/health`.
