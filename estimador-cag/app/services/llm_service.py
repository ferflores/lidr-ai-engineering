"""Servicio de llamada al LLM: el corazón de la arquitectura CAG.

Todo el contexto que necesita el modelo (las estimaciones de ejemplo de
`app/context/examples.py`) viaja dentro del system prompt en cada llamada.
No hay base de datos, ni retrieval, ni persistencia.

Estructura de mensajes:

    [system]    -> instrucciones + ejemplos de estimaciones previas
    [user]      -> transcripción de la reunión a estimar
    [assistant] -> (respuesta del modelo: la estimación)

Se soportan dos proveedores, seleccionables con `LLM_PROVIDER`:
- openai    -> gpt-4o-mini (Chat Completions API)
- anthropic -> claude-haiku-4-5 (Messages API)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import anthropic
import openai

from app.config import Settings, get_settings
from app.context.examples import render_examples

logger = logging.getLogger(__name__)

HOURLY_RATE_EUR = 50


# --- Errores del servicio -----------------------------------------------------


class LLMServiceError(Exception):
    """Error base del servicio LLM."""


class LLMConfigurationError(LLMServiceError):
    """Falta configuración necesaria (proveedor no soportado o API key ausente)."""


class LLMProviderError(LLMServiceError):
    """El proveedor devolvió un error (autenticación, cuota, red, respuesta vacía...)."""


# --- Resultado -----------------------------------------------------------------


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class EstimationResult:
    estimation: str
    model: str
    provider: str
    usage: TokenUsage | None = None


# --- Construcción del prompt ---------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
Eres un estimador de software senior en una consultora de desarrollo a medida. \
Tu trabajo es leer la transcripción de una reunión con un cliente y producir una \
estimación de esfuerzo realista, siguiendo el mismo formato y los mismos criterios \
que las estimaciones previas de la empresa que se incluyen más abajo.

## Cómo estimar
- Identifica los requisitos funcionales y técnicos mencionados en la reunión: \
funcionalidades, integraciones, plataformas, plazos, si existe diseño previo, etc.
- Desglosa el trabajo en tareas concretas y asigna horas a cada una. Usa las \
estimaciones previas como referencia de granularidad y de magnitud: trabajos \
similares deben tener horas similares.
- Incluye siempre tareas transversales: análisis, testing y QA, despliegue.
- Si el cliente menciona un plazo, valora si es viable con el equipo recomendado y dilo.
- Si la reunión no aclara algo relevante, no lo inventes: recógelo en "Supuestos y \
riesgos" o en "Preguntas pendientes para el cliente".
- Sé conservador: es mejor una estimación algo alta con supuestos explícitos que \
una optimista.
- Tarifa de referencia para el coste: {rate} €/hora.

## Formato de salida
Responde únicamente en Markdown y en español, con exactamente esta estructura:

## Estimación: <nombre del proyecto>

### Resumen del alcance
<2-4 líneas>

### Desglose de tareas:
1. <Tarea>: <N> horas
2. ...

**Total estimado: <N> horas**
**Coste estimado: <N> €** (a {rate} €/hora)
**Equipo recomendado: <perfiles>**
**Duración estimada: <X-Y> semanas**

### Supuestos y riesgos
- ...

### Preguntas pendientes para el cliente
- ...

## Ejemplos de estimaciones previas
Estas son estimaciones reales entregadas anteriormente. Úsalas como referencia de \
formato, nivel de detalle y magnitud de las horas.

{examples}
"""


def build_system_prompt(examples: list[dict[str, str]] | None = None) -> str:
    """Construye el system prompt: rol del modelo + instrucciones + ejemplos inyectados."""
    return SYSTEM_PROMPT_TEMPLATE.format(rate=HOURLY_RATE_EUR, examples=render_examples(examples))


def build_user_prompt(transcription: str) -> str:
    """Construye el mensaje de usuario con la transcripción a estimar."""
    return (
        "Esta es la transcripción de la reunión con el cliente. "
        "Genera la estimación siguiendo el formato indicado.\n\n"
        f"<transcripcion>\n{transcription.strip()}\n</transcripcion>"
    )


# --- Punto de entrada ------------------------------------------------------------


async def generate_estimation(transcription: str, settings: Settings | None = None) -> EstimationResult:
    """Genera una estimación a partir de la transcripción usando el proveedor configurado.

    Lanza `LLMConfigurationError` si falta la API key y `LLMProviderError` si el
    proveedor falla o devuelve una respuesta inutilizable.
    """
    settings = settings or get_settings()
    api_key = settings.api_key
    if not api_key:
        raise LLMConfigurationError(
            f"No hay API key para el proveedor '{settings.llm_provider}': "
            f"define {settings.api_key_env_var} en el archivo .env"
        )

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(transcription)
    logger.info(
        "Generando estimación con %s/%s (%d caracteres de transcripción)",
        settings.llm_provider, settings.model, len(transcription),
    )

    if settings.llm_provider == "openai":
        return await _call_openai(system_prompt, user_prompt, settings, api_key)
    if settings.llm_provider == "anthropic":
        return await _call_anthropic(system_prompt, user_prompt, settings, api_key)
    raise LLMConfigurationError(f"Proveedor LLM no soportado: '{settings.llm_provider}'")


# --- Proveedores ------------------------------------------------------------------


async def _call_openai(system_prompt: str, user_prompt: str, settings: Settings, api_key: str) -> EstimationResult:
    async with openai.AsyncOpenAI(api_key=api_key, timeout=settings.llm_timeout_seconds) as client:
        try:
            response = await client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=settings.llm_temperature,
                max_completion_tokens=settings.llm_max_tokens,
            )
        except openai.AuthenticationError as exc:
            raise LLMProviderError("OpenAI ha rechazado la API key (revisa OPENAI_API_KEY)") from exc
        except openai.RateLimitError as exc:
            raise LLMProviderError("OpenAI ha devuelto un error de cuota o de límite de peticiones") from exc
        except openai.APIStatusError as exc:
            raise LLMProviderError(f"OpenAI ha devuelto un error HTTP {exc.status_code}: {exc.message}") from exc
        except openai.APIConnectionError as exc:
            raise LLMProviderError("No se ha podido conectar con OpenAI (red o timeout)") from exc

    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        raise LLMProviderError(f"OpenAI ha devuelto una respuesta vacía (finish_reason={choice.finish_reason})")
    if choice.finish_reason == "length":
        logger.warning("La respuesta de OpenAI se ha cortado por LLM_MAX_TOKENS=%d", settings.llm_max_tokens)

    usage = None
    if response.usage is not None:
        usage = TokenUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)
    return EstimationResult(estimation=text, model=settings.openai_model, provider="openai", usage=usage)


async def _call_anthropic(system_prompt: str, user_prompt: str, settings: Settings, api_key: str) -> EstimationResult:
    async with anthropic.AsyncAnthropic(api_key=api_key, timeout=settings.llm_timeout_seconds) as client:
        try:
            response = await client.messages.create(
                model=settings.anthropic_model,
                max_tokens=settings.llm_max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                # El SDK 1.x ya no expone `temperature` como argumento. Los modelos de
                # la familia 4.5/4.6 (claude-haiku-4-5 incluido) siguen aceptándolo, así
                # que lo enviamos en el cuerpo de la petición. Si cambias a Sonnet 5 u
                # Opus 4.7+, elimina esta línea: esos modelos lo rechazan con un 400.
                extra_body={"temperature": settings.llm_temperature},
            )
        except anthropic.AuthenticationError as exc:
            raise LLMProviderError("Anthropic ha rechazado la API key (revisa ANTHROPIC_API_KEY)") from exc
        except anthropic.RateLimitError as exc:
            raise LLMProviderError("Anthropic ha devuelto un error de cuota o de límite de peticiones") from exc
        except anthropic.APIStatusError as exc:
            raise LLMProviderError(f"Anthropic ha devuelto un error HTTP {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMProviderError("No se ha podido conectar con Anthropic (red o timeout)") from exc

    if response.stop_reason == "refusal":
        raise LLMProviderError("El modelo ha rechazado generar la estimación (stop_reason=refusal)")

    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise LLMProviderError(f"Anthropic ha devuelto una respuesta vacía (stop_reason={response.stop_reason})")
    if response.stop_reason == "max_tokens":
        logger.warning("La respuesta de Anthropic se ha cortado por LLM_MAX_TOKENS=%d", settings.llm_max_tokens)

    usage = TokenUsage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
    return EstimationResult(estimation=text, model=settings.anthropic_model, provider="anthropic", usage=usage)
