"""Estimación como objeto tipado, no como texto: *structured outputs* en los dos proveedores.

Hueco de la sesión 04. El material propone la librería **Instructor**, que abstrae ambos proveedores y
reintenta cuando la validación falla. No se puede usar aquí: la última versión de Instructor fija
`jiter>=0.6.1,<0.15` y el SDK de OpenAI que usa el proyecto exige `jiter>=0.16.0`, así que el resolutor
de dependencias no encuentra solución. En vez de bajar el SDK de OpenAI se implementa a mano lo que
Instructor hace por dentro, que son tres cosas:

1. Convertir el modelo Pydantic en JSON Schema y pasárselo al proveedor en su formato nativo.
2. Validar la respuesta contra el modelo.
3. Si la validación falla, **mostrarle el error al modelo y reintentar** (hasta `MAX_ATTEMPTS`).

Cada proveedor lo expone a su manera, y las dos son nativas:

- **OpenAI**: `client.chat.completions.parse(response_format=Modelo)`.
- **Anthropic**: `client.messages.create(output_config={"format": {"type": "json_schema", "schema": ...}})`.
  El parámetro antiguo `output_format` está obsoleto; el esquema va dentro de `output_config.format`.

El tercer punto es el que de verdad cambia el resultado: el esquema garantiza la *forma*, pero no que la
aritmética cuadre. Los validadores de `StructuredEstimation` comprueban que los bloques sumen el subtotal,
que la contingencia cuadre y que el coste sea horas × tarifa. Cuando algo de eso falla, el error vuelve al
modelo como un mensaje más y se le pide que lo corrija.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import anthropic
import openai
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.schemas.estimation import EstimationRequest
from app.schemas.structured import StructuredEstimation
from app.services.llm_service import (
    LLMProviderError,
    RenderedPrompt,
    TokenUsage,
    _anthropic_extra_body,
    _translate_anthropic_error,
    _translate_openai_error,
    build_prompt,
    require_api_key,
)

logger = logging.getLogger(__name__)

# Intentos totales por petición: el primero más dos correcciones. Instructor usa tres por defecto.
MAX_ATTEMPTS = 3

# Se añade al prompt solo en el modo estructurado: el system prompt de las plantillas pide Markdown.
STRUCTURED_INSTRUCTION = (
    "Devuelve la estimación como un objeto JSON que cumpla el esquema indicado. "
    "No escribas Markdown ni texto fuera del objeto. "
    "La aritmética tiene que cuadrar: la suma de las horas de `items` es `subtotal_hours`, "
    "`total_hours` es el subtotal más la contingencia, y `total_cost_eur` es `total_hours` por `hourly_rate_eur`."
)


@dataclass(frozen=True)
class StructuredResult:
    estimation: StructuredEstimation
    model: str
    provider: str
    usage: TokenUsage | None
    attempts: int                 # 1 si validó a la primera
    repair_errors: tuple[str, ...]  # los errores de validación que hubo que corregir, en orden
    elapsed_seconds: float


class StructuredOutputError(LLMProviderError):
    """El modelo no produjo un objeto válido ni después de los reintentos."""


def _messages_for(prompt: RenderedPrompt) -> list[dict[str, str]]:
    mensajes = [dict(m) for m in prompt.messages]
    mensajes[-1] = {**mensajes[-1], "content": f"{mensajes[-1]['content']}\n\n{STRUCTURED_INSTRUCTION}"}
    return mensajes


def _error_detail(error: Exception) -> str:
    """El detalle accionable de un fallo de validación.

    `str(ValidationError)` empieza por "N validation errors for ..." y mete el detalle en las líneas
    siguientes; quedarse con la primera línea deja un diagnóstico inútil tanto para el modelo como para
    quien lea los logs. Aquí se aplanan los errores a "campo: motivo".
    """
    if isinstance(error, ValidationError):
        return "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'raíz'}: {e['msg']}" for e in error.errors()[:6]
        )
    return str(error)


def _repair_message(error: Exception) -> dict[str, str]:
    """El mensaje con el que se le pide al modelo que corrija. Es lo que hace Instructor al reintentar."""
    detalle = _error_detail(error)
    return {
        "role": "user",
        "content": (
            f"El objeto anterior no es válido: {detalle}. "
            "Devuelve el objeto corregido completo, respetando el esquema y cuadrando la aritmética."
        ),
    }


async def generate_structured_estimation(
    request: EstimationRequest,
    settings: Settings | None = None,
) -> StructuredResult:
    """Genera la estimación como objeto validado, reintentando si la validación falla."""
    settings = settings or get_settings()
    prompt = build_prompt(request, settings)
    api_key = require_api_key(settings)
    mensajes = _messages_for(prompt)

    logger.info(
        "Estimación estructurada con %s/%s, plantilla %s",
        settings.llm_provider, settings.model, prompt.template.id,
    )

    started = time.perf_counter()
    errores: list[str] = []
    for intento in range(1, MAX_ATTEMPTS + 1):
        if settings.llm_provider == "openai":
            crudo, usage = await _parse_openai(prompt.system, mensajes, settings, api_key)
        else:
            crudo, usage = await _parse_anthropic(prompt.system, mensajes, settings, api_key)

        try:
            estimacion = StructuredEstimation.model_validate(crudo)
        except (ValidationError, ValueError) as exc:
            errores.append(_error_detail(exc)[:300])
            if intento == MAX_ATTEMPTS:
                logger.error("La salida estructurada no validó tras %d intentos", MAX_ATTEMPTS)
                raise StructuredOutputError(
                    f"El modelo no ha producido una estimación válida tras {MAX_ATTEMPTS} intentos: {errores[-1]}"
                ) from exc
            logger.warning("Intento %d inválido, se pide corrección: %s", intento, errores[-1])
            mensajes = [*mensajes, {"role": "assistant", "content": json.dumps(crudo, ensure_ascii=False)}, _repair_message(exc)]
            continue

        return StructuredResult(
            estimation=estimacion,
            model=settings.model,
            provider=settings.llm_provider,
            usage=usage,
            attempts=intento,
            repair_errors=tuple(errores),
            elapsed_seconds=time.perf_counter() - started,
        )
    raise AssertionError("inalcanzable")  # pragma: no cover


async def _parse_openai(
    system: str, mensajes: list[dict[str, str]], settings: Settings, api_key: str
) -> tuple[dict, TokenUsage | None]:
    """Structured outputs nativos: el modelo Pydantic viaja como `response_format`."""
    async with openai.AsyncOpenAI(api_key=api_key, timeout=settings.llm_timeout_seconds) as client:
        try:
            response = await client.chat.completions.parse(
                model=settings.openai_model,
                messages=[{"role": "system", "content": system}, *mensajes],
                response_format=StructuredEstimation,
                temperature=settings.llm_temperature,
                max_completion_tokens=settings.llm_max_tokens,
            )
        except openai.APIError as exc:
            raise _translate_openai_error(exc) from exc

    mensaje = response.choices[0].message
    if getattr(mensaje, "refusal", None):
        raise LLMProviderError(f"OpenAI ha rechazado la petición: {mensaje.refusal}")
    usage = None
    if response.usage is not None:
        usage = TokenUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)
    # `parsed` ya viene validado por el SDK; si vuelve vacío se cae al texto crudo para poder reintentar.
    if mensaje.parsed is not None:
        return mensaje.parsed.model_dump(), usage
    return json.loads(mensaje.content or "{}"), usage


async def _parse_anthropic(
    system: str, mensajes: list[dict[str, str]], settings: Settings, api_key: str
) -> tuple[dict, TokenUsage | None]:
    """Structured outputs nativos de Anthropic: el esquema va en `output_config.format`."""
    esquema = StructuredEstimation.model_json_schema()
    async with anthropic.AsyncAnthropic(api_key=api_key, timeout=settings.llm_timeout_seconds) as client:
        try:
            response = await client.messages.create(
                model=settings.anthropic_model,
                max_tokens=settings.llm_max_tokens,
                system=system,
                messages=mensajes,
                output_config={"format": {"type": "json_schema", "schema": esquema}},
                extra_body=_anthropic_extra_body(settings),
            )
        except anthropic.APIError as exc:
            raise _translate_anthropic_error(exc) from exc

    if response.stop_reason == "refusal":
        raise LLMProviderError("El modelo ha rechazado generar la estimación (stop_reason=refusal)")
    texto = "".join(b.text for b in response.content if b.type == "text").strip()
    if not texto:
        raise LLMProviderError(f"Anthropic ha devuelto una respuesta vacía (stop_reason={response.stop_reason})")
    usage = TokenUsage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
    try:
        return json.loads(texto), usage
    except json.JSONDecodeError as exc:
        raise LLMProviderError(f"Anthropic no ha devuelto JSON válido: {texto[:200]}") from exc
