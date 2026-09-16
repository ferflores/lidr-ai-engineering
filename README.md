# lidr-ai-engineering

Proyectos y ejercicios de Fer para el curso **AI Engineering** de [Lidr](https://lidr.co).

## Proyectos

| Sesión | Proyecto | Qué es | Rama |
|---|---|---|---|
| 02 | [`estimador-cag/`](estimador-cag/) | Servicio FastAPI que recibe la transcripción de una reunión y devuelve una estimación de software generada por un LLM, con arquitectura CAG (contexto estático inyectado en el prompt). | `session-02-estimador-cag` |

## Integración continua

[![estimador-cag](https://github.com/ferflores/lidr-ai-engineering/actions/workflows/estimador-cag.yml/badge.svg?branch=session-02-estimador-cag)](https://github.com/ferflores/lidr-ai-engineering/actions/workflows/estimador-cag.yml)

Cada push que toque `estimador-cag/` ejecuta `.github/workflows/estimador-cag.yml`, que valida la
estructura del proyecto, pasa los tests y arranca el servicio para comprobar `/health`, `/docs` y el
endpoint de estimación. Ver `estimador-cag/scripts/verificar.sh`.
