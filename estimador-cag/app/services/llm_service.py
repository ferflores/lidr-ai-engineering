"""Servicio de llamada al LLM: el corazón de la arquitectura CAG.

Todo el contexto que necesita el modelo (las estimaciones de ejemplo de
`app/context/examples.py`) viaja dentro del system prompt en cada llamada.
No hay base de datos, ni retrieval, ni persistencia.

Estructura de mensajes:

    [system]    -> instrucciones + ejemplos de estimaciones previas
    [user]      -> transcripción de la reunión a estimar
    [assistant] -> (respuesta del modelo: la estimación)

Dos formas de usarlo:
- `generate_estimation(...)`: asíncrona, respuesta completa. La usa el endpoint REST.
- `EstimationStream(...)`: síncrona, en streaming (fragmento a fragmento). La usa la
  interfaz de chat de Streamlit. Admite historial de conversación para preguntas de
  seguimiento sobre una estimación.

Se soportan dos proveedores, seleccionables con `LLM_PROVIDER`:
- openai    -> gpt-4o-mini (Chat Completions API)
- anthropic -> claude-haiku-4-5 (Messages API)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import anthropic
import openai

from app.config import Settings, get_settings
from app.context.examples import render_examples
from app.services.pricing import CostEstimate, estimate_cost, find_pricing

logger = logging.getLogger(__name__)

HOURLY_RATE_EUR = 50


# --- Errores del servicio -----------------------------------------------------


class LLMServiceError(Exception):
    """Error base del servicio LLM."""


class LLMConfigurationError(LLMServiceError):
    """Falta configuración necesaria (proveedor no soportado o API key ausente)."""


class LLMProviderError(LLMServiceError):
    """El proveedor devolvió un error (autenticación, cuota, red, respuesta vacía...)."""


# --- Datos -----------------------------------------------------------------------


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class PromptInfo:
    """Tamaño de lo que viaja en el prompt: el contexto fijo (CAG) y la transcripción."""

    system_prompt_chars: int
    transcription_chars: int


@dataclass(frozen=True)
class EstimationResult:
    estimation: str
    model: str
    provider: str
    usage: TokenUsage | None = None
    cost: CostEstimate | None = None
    prompt: PromptInfo | None = None
    elapsed_seconds: float | None = None


@dataclass(frozen=True)
class ChatTurn:
    """Un turno previo de la conversación (`role` es "user" o "assistant")."""

    role: str
    content: str


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


def build_messages(message: str, history: Iterable[ChatTurn] = ()) -> list[dict[str, str]]:
    """Mensajes de la conversación (sin el system prompt), en el formato común a ambos proveedores.

    El primer mensaje de usuario de la conversación es la transcripción y se envuelve con
    `build_user_prompt`; los siguientes (preguntas o ajustes sobre la estimación) se envían tal cual.
    """
    turns = [*history, ChatTurn(role="user", content=message)]
    messages: list[dict[str, str]] = []
    first_user_seen = False
    for turn in turns:
        content = turn.content.strip()
        if turn.role == "user" and not first_user_seen:
            content = build_user_prompt(content)
            first_user_seen = True
        messages.append({"role": turn.role, "content": content})
    return messages


# --- Utilidades comunes ----------------------------------------------------------


def _require_api_key(settings: Settings) -> str:
    api_key = settings.api_key
    if not api_key:
        raise LLMConfigurationError(
            f"No hay API key para el proveedor '{settings.llm_provider}': "
            f"define {settings.api_key_env_var} en el archivo .env"
        )
    if settings.llm_provider not in ("openai", "anthropic"):
        raise LLMConfigurationError(f"Proveedor LLM no soportado: '{settings.llm_provider}'")
    return api_key


def _build_result(
    text: str,
    usage: TokenUsage | None,
    settings: Settings,
    system_prompt: str,
    transcription: str,
    elapsed_seconds: float | None = None,
) -> EstimationResult:
    cost = None
    if usage is not None:
        pricing = find_pricing(settings.model, settings.llm_input_price_per_mtok, settings.llm_output_price_per_mtok)
        if pricing is not None:
            cost = estimate_cost(usage.input_tokens, usage.output_tokens, pricing)
        logger.info(
            "Tokens: %d entrada + %d salida = %d; coste: %s; tiempo: %s",
            usage.input_tokens, usage.output_tokens, usage.total_tokens,
            f"{cost.total_usd:.6f} USD" if cost else "desconocido (modelo sin precio en la tabla)",
            f"{elapsed_seconds:.1f} s" if elapsed_seconds is not None else "?",
        )
    return EstimationResult(
        estimation=text,
        model=settings.model,
        provider=settings.llm_provider,
        usage=usage,
        cost=cost,
        prompt=PromptInfo(system_prompt_chars=len(system_prompt), transcription_chars=len(transcription.strip())),
        elapsed_seconds=elapsed_seconds,
    )


def _translate_openai_error(exc: openai.APIError) -> LLMProviderError:
    if isinstance(exc, openai.AuthenticationError):
        return LLMProviderError("OpenAI ha rechazado la API key (revisa OPENAI_API_KEY)")
    if isinstance(exc, openai.RateLimitError):
        return LLMProviderError("OpenAI ha devuelto un error de cuota o de límite de peticiones")
    if isinstance(exc, openai.APIStatusError):
        return LLMProviderError(f"OpenAI ha devuelto un error HTTP {exc.status_code}: {exc.message}")
    if isinstance(exc, openai.APIConnectionError):
        return LLMProviderError("No se ha podido conectar con OpenAI (red o timeout)")
    return LLMProviderError(f"Error de OpenAI: {exc}")


def _translate_anthropic_error(exc: anthropic.APIError) -> LLMProviderError:
    if isinstance(exc, anthropic.AuthenticationError):
        return LLMProviderError("Anthropic ha rechazado la API key (revisa ANTHROPIC_API_KEY)")
    if isinstance(exc, anthropic.RateLimitError):
        return LLMProviderError("Anthropic ha devuelto un error de cuota o de límite de peticiones")
    if isinstance(exc, anthropic.APIStatusError):
        return LLMProviderError(f"Anthropic ha devuelto un error HTTP {exc.status_code}: {exc.message}")
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMProviderError("No se ha podido conectar con Anthropic (red o timeout)")
    return LLMProviderError(f"Error de Anthropic: {exc}")


# El SDK de Anthropic 1.x ya no expone `temperature` como argumento. Los modelos de la
# familia 4.5/4.6 (claude-haiku-4-5 incluido) siguen aceptándolo, así que se envía en el
# cuerpo de la petición. Con Sonnet 5 u Opus 4.7+ hay que quitarlo: lo rechazan con un 400.
def _anthropic_extra_body(settings: Settings) -> dict[str, float]:
    return {"temperature": settings.llm_temperature}


# --- Respuesta completa (endpoint REST) --------------------------------------------


async def generate_estimation(transcription: str, settings: Settings | None = None) -> EstimationResult:
    """Genera una estimación a partir de la transcripción usando el proveedor configurado.

    Lanza `LLMConfigurationError` si falta la API key y `LLMProviderError` si el
    proveedor falla o devuelve una respuesta inutilizable.
    """
    settings = settings or get_settings()
    api_key = _require_api_key(settings)
    system_prompt = build_system_prompt()
    messages = build_messages(transcription)
    logger.info(
        "Generando estimación con %s/%s (%d caracteres de transcripción)",
        settings.llm_provider, settings.model, len(transcription),
    )

    started = time.perf_counter()
    if settings.llm_provider == "openai":
        text, usage = await _call_openai(system_prompt, messages, settings, api_key)
    else:
        text, usage = await _call_anthropic(system_prompt, messages, settings, api_key)
    return _build_result(text, usage, settings, system_prompt, transcription, time.perf_counter() - started)


async def _call_openai(
    system_prompt: str, messages: list[dict[str, str]], settings: Settings, api_key: str
) -> tuple[str, TokenUsage | None]:
    async with openai.AsyncOpenAI(api_key=api_key, timeout=settings.llm_timeout_seconds) as client:
        try:
            response = await client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "system", "content": system_prompt}, *messages],
                temperature=settings.llm_temperature,
                max_completion_tokens=settings.llm_max_tokens,
            )
        except openai.APIError as exc:
            raise _translate_openai_error(exc) from exc

    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        raise LLMProviderError(f"OpenAI ha devuelto una respuesta vacía (finish_reason={choice.finish_reason})")
    if choice.finish_reason == "length":
        logger.warning("La respuesta de OpenAI se ha cortado por LLM_MAX_TOKENS=%d", settings.llm_max_tokens)

    usage = None
    if response.usage is not None:
        usage = TokenUsage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)
    return text, usage


async def _call_anthropic(
    system_prompt: str, messages: list[dict[str, str]], settings: Settings, api_key: str
) -> tuple[str, TokenUsage | None]:
    async with anthropic.AsyncAnthropic(api_key=api_key, timeout=settings.llm_timeout_seconds) as client:
        try:
            response = await client.messages.create(
                model=settings.anthropic_model,
                max_tokens=settings.llm_max_tokens,
                system=system_prompt,
                messages=messages,
                extra_body=_anthropic_extra_body(settings),
            )
        except anthropic.APIError as exc:
            raise _translate_anthropic_error(exc) from exc

    if response.stop_reason == "refusal":
        raise LLMProviderError("El modelo ha rechazado generar la estimación (stop_reason=refusal)")

    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise LLMProviderError(f"Anthropic ha devuelto una respuesta vacía (stop_reason={response.stop_reason})")
    if response.stop_reason == "max_tokens":
        logger.warning("La respuesta de Anthropic se ha cortado por LLM_MAX_TOKENS=%d", settings.llm_max_tokens)

    usage = TokenUsage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
    return text, usage


# --- Streaming (interfaz de chat) --------------------------------------------------


class EstimationStream:
    """Estimación en streaming: un iterable de fragmentos de texto, listo para `st.write_stream`.

        stream = EstimationStream(transcripcion, settings, history=turnos_previos)
        for fragmento in stream:
            ...
        stream.result   # EstimationResult con tokens, coste y tiempo (disponible al terminar)

    Los errores (`LLMConfigurationError`, `LLMProviderError`) se lanzan al iterar.
    """

    def __init__(self, message: str, settings: Settings | None = None, history: Iterable[ChatTurn] = ()):
        self.message = message
        self.settings = settings or get_settings()
        self.history = tuple(history)
        self.result: EstimationResult | None = None

    def __iter__(self) -> Iterator[str]:
        settings = self.settings
        api_key = _require_api_key(settings)
        system_prompt = build_system_prompt()
        messages = build_messages(self.message, self.history)
        logger.info(
            "Estimación en streaming con %s/%s (%d turnos previos, %d caracteres)",
            settings.llm_provider, settings.model, len(self.history), len(self.message),
        )

        started = time.perf_counter()
        if settings.llm_provider == "openai":
            generator = _stream_openai(system_prompt, messages, settings, api_key)
        else:
            generator = _stream_anthropic(system_prompt, messages, settings, api_key)

        chunks: list[str] = []
        # `yield from` reenvía los fragmentos y recoge el valor de retorno del generador (el usage).
        usage = yield from _collecting(generator, chunks)
        text = "".join(chunks).strip()
        if not text:
            raise LLMProviderError(f"{settings.llm_provider} ha devuelto una respuesta vacía")
        self.result = _build_result(text, usage, settings, system_prompt, self.message, time.perf_counter() - started)


def _collecting(generator: Iterator[str], chunks: list[str]) -> Iterator[str]:
    """Reenvía cada fragmento y lo acumula en `chunks`; devuelve el valor de retorno del generador."""
    usage = None
    while True:
        try:
            chunk = next(generator)
        except StopIteration as stop:
            usage = stop.value
            break
        chunks.append(chunk)
        yield chunk
    return usage


def _stream_openai(
    system_prompt: str, messages: list[dict[str, str]], settings: Settings, api_key: str
) -> Iterator[str]:
    usage = None
    with openai.OpenAI(api_key=api_key, timeout=settings.llm_timeout_seconds) as client:
        try:
            stream = client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "system", "content": system_prompt}, *messages],
                temperature=settings.llm_temperature,
                max_completion_tokens=settings.llm_max_tokens,
                stream=True,
                stream_options={"include_usage": True},  # el último chunk trae los tokens consumidos
            )
            for chunk in stream:
                if chunk.usage is not None:
                    usage = TokenUsage(input_tokens=chunk.usage.prompt_tokens, output_tokens=chunk.usage.completion_tokens)
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason == "length":
                    logger.warning("La respuesta de OpenAI se ha cortado por LLM_MAX_TOKENS=%d", settings.llm_max_tokens)
                if choice.delta and choice.delta.content:
                    yield choice.delta.content
        except openai.APIError as exc:
            raise _translate_openai_error(exc) from exc
    return usage


def _stream_anthropic(
    system_prompt: str, messages: list[dict[str, str]], settings: Settings, api_key: str
) -> Iterator[str]:
    with anthropic.Anthropic(api_key=api_key, timeout=settings.llm_timeout_seconds) as client:
        try:
            with client.messages.stream(
                model=settings.anthropic_model,
                max_tokens=settings.llm_max_tokens,
                system=system_prompt,
                messages=messages,
                extra_body=_anthropic_extra_body(settings),
            ) as stream:
                for text in stream.text_stream:
                    yield text
                final = stream.get_final_message()
        except anthropic.APIError as exc:
            raise _translate_anthropic_error(exc) from exc

    if final.stop_reason == "refusal":
        raise LLMProviderError("El modelo ha rechazado generar la estimación (stop_reason=refusal)")
    if final.stop_reason == "max_tokens":
        logger.warning("La respuesta de Anthropic se ha cortado por LLM_MAX_TOKENS=%d", settings.llm_max_tokens)
    return TokenUsage(input_tokens=final.usage.input_tokens, output_tokens=final.usage.output_tokens)
