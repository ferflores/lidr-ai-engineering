"""Servicio de llamada al LLM: el corazón de la arquitectura CAG.

Todo el contexto que necesita el modelo (las estimaciones de ejemplo de
`app/context/examples.py`) viaja dentro del system prompt en cada llamada.
No hay base de datos, ni retrieval, ni persistencia.

El texto de los prompts NO está aquí: vive en `prompts/estimacion/<versión>/` como plantillas
versionadas (ver `prompts/README.md`) y lo carga `app/services/prompts.py`. Este módulo solo
decide qué variables recibe la plantilla y cómo se llama al proveedor.

Estructura de mensajes:

    [system]    -> plantilla system.md.j2: rol, reglas, formato + ejemplos de estimaciones previas
    [user]      -> plantilla user.md.j2: parámetros tipados de la petición + transcripción
    [assistant] -> (respuesta del modelo: la estimación)

Dos formas de usarlo:
- `generate_estimation(request, settings)`: asíncrona, respuesta completa. La usa el endpoint REST.
- `EstimationStream(request, settings)`: síncrona, en streaming (fragmento a fragmento). La usa el
  formulario de Streamlit.

Dos modos de uso:
- Transaccional (`POST /api/v1/estimate`, sesiones 02-04): cada llamada es independiente y reproducible;
  `RenderedPrompt.history` va vacío.
- Conversacional (`POST /api/v1/sessions/{id}/estimate`, sesión 05): `app/services/conversation.py` añade
  los turnos anteriores de la ventana deslizante en `history` y la memoria del proyecto (`ProjectMetadata`)
  como variable de la plantilla del system prompt. Este módulo no sabe de sesiones: solo renderiza y llama.

Se soportan dos proveedores, seleccionables con `LLM_PROVIDER`:
- openai    -> gpt-4o-mini (Chat Completions API)
- anthropic -> claude-haiku-4-5 (Messages API)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import anthropic
import openai

from app.config import Settings, get_settings
from app.context.examples import render_examples
from app.schemas.estimation import EstimationRequest
from app.services.pricing import CostEstimate, estimate_cost, find_pricing
from app.services.prompts import PromptTemplate, get_template
from app.services.sessions import Anchor, ProjectMetadata

logger = logging.getLogger(__name__)


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
    """Qué plantilla se usó y cuánto ocupa cada parte del prompt."""

    template: str            # nombre de la plantilla ("estimacion")
    version: str             # versión usada ("v2")
    system_prompt_chars: int  # contexto fijo: instrucciones + ejemplos CAG (igual en todas las llamadas)
    user_prompt_chars: int    # parte variable: parámetros + transcripción
    transcription_chars: int


@dataclass(frozen=True)
class RenderedPrompt:
    """Lo que se envía al modelo: system prompt, turnos anteriores (si hay) y el mensaje de usuario actual."""

    template: PromptTemplate
    system: str
    user: str
    history: tuple[dict[str, str], ...] = ()  # turnos user/assistant anteriores dentro de la ventana; vacío en modo transaccional

    @property
    def messages(self) -> list[dict[str, str]]:
        """Mensajes de la conversación sin el system prompt, en el formato común a ambos proveedores."""
        return [*self.history, {"role": "user", "content": self.user}]


@dataclass(frozen=True)
class EstimationResult:
    estimation: str
    model: str
    provider: str
    usage: TokenUsage | None = None
    cost: CostEstimate | None = None
    prompt: PromptInfo | None = None
    elapsed_seconds: float | None = None


# --- Construcción del prompt ---------------------------------------------------


def build_prompt(
    request: EstimationRequest,
    settings: Settings | None = None,
    examples: list[dict[str, str]] | None = None,
    project_metadata: ProjectMetadata | None = None,
    anchors: Sequence[Anchor] = (),
    summary: str = "",
) -> RenderedPrompt:
    """Renderiza la plantilla (la versión pedida en `request.prompt_version` o la configurada).

    Las plantillas reciben siempre el mismo conjunto de variables: los parámetros tipados de la
    petición, el bloque de ejemplos CAG, la memoria del proyecto (`project_metadata`, None si no hay
    sesión o la sesión acaba de empezar), las anclas y el resumen acumulativo. Cada versión usa las que
    necesita: `v3` renderiza el bloque `<project_metadata>`; `v4` añade `<anchors>` y `<summary>`;
    `v1` y `v2` ignoran todo eso.
    """
    template = get_template(settings, request.prompt_version)
    variables = {
        **request.prompt_variables(),
        "examples": render_examples(examples),
        "project_metadata": project_metadata.prompt_variables() if project_metadata else None,
        "anchors": [{"kind": a.kind, "text": a.text, "turn": a.turn} for a in anchors] or None,
        "summary": summary.strip() or None,
    }
    return RenderedPrompt(
        template=template,
        system=template.render_system(**variables),
        user=template.render_user(**variables),
    )


def build_system_prompt(request: EstimationRequest | None = None, settings: Settings | None = None) -> str:
    """System prompt renderizado. Sin petición, usa los parámetros por defecto (útil para inspeccionarlo)."""
    return build_prompt(request or EstimationRequest(transcription="(vista previa del system prompt)"), settings).system


def build_user_prompt(request: EstimationRequest, settings: Settings | None = None) -> str:
    """Mensaje de usuario renderizado: parámetros + transcripción."""
    return build_prompt(request, settings).user


# --- Utilidades comunes ----------------------------------------------------------


def require_api_key(settings: Settings) -> str:
    api_key = settings.api_key
    if not api_key:
        raise LLMConfigurationError(
            f"No hay API key para el proveedor '{settings.llm_provider}': "
            f"define {settings.api_key_env_var} en el archivo .env"
        )
    if settings.llm_provider not in ("openai", "anthropic"):
        raise LLMConfigurationError(f"Proveedor LLM no soportado: '{settings.llm_provider}'")
    return api_key


def cost_for(usage: TokenUsage | None, settings: Settings, model: str | None = None) -> CostEstimate | None:
    """Coste de una llamada según la tabla de precios (o los precios sobrescritos por configuración)."""
    if usage is None:
        return None
    pricing = find_pricing(model or settings.model, settings.llm_input_price_per_mtok, settings.llm_output_price_per_mtok)
    return estimate_cost(usage.input_tokens, usage.output_tokens, pricing) if pricing is not None else None


def build_result(
    text: str,
    usage: TokenUsage | None,
    settings: Settings,
    prompt: RenderedPrompt,
    request: EstimationRequest,
    elapsed_seconds: float | None = None,
    model: str | None = None,
) -> EstimationResult:
    cost = None
    if usage is not None:
        cost = cost_for(usage, settings, model)
        logger.info(
            "Tokens: %d entrada + %d salida = %d; coste: %s; tiempo: %s; plantilla: %s",
            usage.input_tokens, usage.output_tokens, usage.total_tokens,
            f"{cost.total_usd:.6f} USD" if cost else "desconocido (modelo sin precio en la tabla)",
            f"{elapsed_seconds:.1f} s" if elapsed_seconds is not None else "?",
            prompt.template.id,
        )
    return EstimationResult(
        estimation=text,
        model=model or settings.model,
        provider=settings.llm_provider,
        usage=usage,
        cost=cost,
        prompt=PromptInfo(
            template=prompt.template.name,
            version=prompt.template.version,
            system_prompt_chars=len(prompt.system),
            user_prompt_chars=len(prompt.user),
            transcription_chars=len(request.transcription),
        ),
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


async def generate_estimation(request: EstimationRequest, settings: Settings | None = None) -> EstimationResult:
    """Genera una estimación para la petición usando el proveedor configurado.

    Lanza `PromptTemplateNotFound` si la versión de plantilla no existe, `LLMConfigurationError`
    si falta la API key y `LLMProviderError` si el proveedor falla o devuelve una respuesta inutilizable.
    """
    settings = settings or get_settings()
    prompt = build_prompt(request, settings)  # antes que la API key: una versión inexistente es un error de la petición
    api_key = require_api_key(settings)
    logger.info(
        "Generando estimación con %s/%s, plantilla %s (%d caracteres de transcripción)",
        settings.llm_provider, settings.model, prompt.template.id, len(request.transcription),
    )

    started = time.perf_counter()
    text, usage = await call_llm(prompt, settings, api_key)
    return build_result(text, usage, settings, prompt, request, time.perf_counter() - started)


async def call_llm(
    prompt: RenderedPrompt,
    settings: Settings,
    api_key: str,
    model: str | None = None,
    max_tokens: int | None = None,
) -> tuple[str, TokenUsage | None]:
    """Envía el prompt (system + historial + mensaje actual) al proveedor configurado y devuelve texto y tokens.

    `model` y `max_tokens` permiten que el tier dinámico (app/services/tiering.py) elija otro modelo o
    otro presupuesto de salida para un turno concreto; por defecto, los de la configuración.
    """
    if settings.llm_provider == "openai":
        return await _call_openai(prompt, settings, api_key, model or settings.openai_model, max_tokens or settings.llm_max_tokens)
    return await _call_anthropic(prompt, settings, api_key, model or settings.anthropic_model, max_tokens or settings.llm_max_tokens)


async def _call_openai(
    prompt: RenderedPrompt, settings: Settings, api_key: str, model: str, max_tokens: int
) -> tuple[str, TokenUsage | None]:
    async with openai.AsyncOpenAI(api_key=api_key, timeout=settings.llm_timeout_seconds) as client:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt.system}, *prompt.messages],
                temperature=settings.llm_temperature,
                max_completion_tokens=max_tokens,
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
    prompt: RenderedPrompt, settings: Settings, api_key: str, model: str, max_tokens: int
) -> tuple[str, TokenUsage | None]:
    async with anthropic.AsyncAnthropic(api_key=api_key, timeout=settings.llm_timeout_seconds) as client:
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=prompt.system,
                messages=prompt.messages,
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


# --- Streaming (formulario de Streamlit) ------------------------------------------


class EstimationStream:
    """Estimación en streaming: un iterable de fragmentos de texto, listo para `st.write_stream`.

        stream = EstimationStream(request, settings)
        for fragmento in stream:
            ...
        stream.result   # EstimationResult con tokens, coste y tiempo (disponible al terminar)
        stream.prompt   # RenderedPrompt: lo que se envió al modelo (disponible al empezar a iterar)

    Los errores (`PromptTemplateNotFound`, `LLMConfigurationError`, `LLMProviderError`) se lanzan al iterar.
    """

    def __init__(self, request: EstimationRequest, settings: Settings | None = None):
        self.request = request
        self.settings = settings or get_settings()
        self.prompt: RenderedPrompt | None = None
        self.result: EstimationResult | None = None

    def __iter__(self) -> Iterator[str]:
        settings = self.settings
        prompt = self.prompt = build_prompt(self.request, settings)
        api_key = require_api_key(settings)
        logger.info(
            "Estimación en streaming con %s/%s, plantilla %s (%d caracteres de transcripción)",
            settings.llm_provider, settings.model, prompt.template.id, len(self.request.transcription),
        )

        started = time.perf_counter()
        if settings.llm_provider == "openai":
            generator = _stream_openai(prompt, settings, api_key)
        else:
            generator = _stream_anthropic(prompt, settings, api_key)

        chunks: list[str] = []
        # `yield from` reenvía los fragmentos y recoge el valor de retorno del generador (el usage).
        usage = yield from _collecting(generator, chunks)
        text = "".join(chunks).strip()
        if not text:
            raise LLMProviderError(f"{settings.llm_provider} ha devuelto una respuesta vacía")
        self.result = build_result(text, usage, settings, prompt, self.request, time.perf_counter() - started)


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


def _stream_openai(prompt: RenderedPrompt, settings: Settings, api_key: str) -> Iterator[str]:
    usage = None
    with openai.OpenAI(api_key=api_key, timeout=settings.llm_timeout_seconds) as client:
        try:
            stream = client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "system", "content": prompt.system}, *prompt.messages],
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


def _stream_anthropic(prompt: RenderedPrompt, settings: Settings, api_key: str) -> Iterator[str]:
    with anthropic.Anthropic(api_key=api_key, timeout=settings.llm_timeout_seconds) as client:
        try:
            with client.messages.stream(
                model=settings.anthropic_model,
                max_tokens=settings.llm_max_tokens,
                system=prompt.system,
                messages=prompt.messages,
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
