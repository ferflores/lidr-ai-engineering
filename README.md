# lidr-ai-engineering

Proyectos y ejercicios de Fer para el curso **AI Engineering** de [Lidr](https://lidr.co).

## Proyectos

| Sesión | Proyecto | Qué es | Rama |
|---|---|---|---|
| 02 | [`estimador-cag/`](estimador-cag/) | Servicio FastAPI que recibe la transcripción de una reunión y devuelve una estimación de software generada por un LLM, con arquitectura CAG (contexto estático inyectado en el prompt). Se ejecuta con `uv` o con `docker compose`. | `session-02-estimador-cag` |
| 03 | [`estimador-cag/streamlit_app.py`](estimador-cag/streamlit_app.py) | Interfaz de chat con Streamlit sobre el mismo proyecto: estimación en streaming, historial de conversación y panel con el contexto CAG y las métricas de tokens y coste. La rama incluye todo lo de la sesión 02. | `session-03-streamlit-chat` |

## Integración continua

[![estimador-cag](https://github.com/ferflores/lidr-ai-engineering/actions/workflows/estimador-cag.yml/badge.svg?branch=session-03-streamlit-chat)](https://github.com/ferflores/lidr-ai-engineering/actions/workflows/estimador-cag.yml)

Cada push que toque `estimador-cag/` ejecuta `.github/workflows/estimador-cag.yml`, que valida la
estructura del proyecto, pasa los tests, arranca la API y la interfaz de chat para comprobar que
responden (ver `estimador-cag/scripts/verificar.sh`), y después construye la imagen Docker y levanta
los dos servicios con `docker compose` para comprobar `/health` y `/_stcore/health`.
