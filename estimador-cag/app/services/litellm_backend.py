"""Wrapper de proveedores con LiteLLM, como alternativa a los SDK directos.

Hueco de la sesión 03. La analogía del material es exacta: **lo que un ORM hace con las bases de datos,
LiteLLM lo hace con los LLMs**. Cambiar de proveedor pasa a ser configuración en vez de código.

El proyecto ya tenía abstracción de proveedores, pero escrita a mano: dos funciones hermanas en
`llm_service.py`, cada una con su SDK, su forma de pasar el system prompt y su forma de leer la respuesta.
Funciona y es explícita, pero cada proveedor nuevo son otras cincuenta líneas. LiteLLM unifica las tres
cosas bajo la convención de OpenAI.

Se añade como **backend opcional** (`LLM_BACKEND=litellm`), no como sustitución:

- Los SDK directos siguen siendo el camino por defecto, y son los que usaron las entregas de las sesiones
  02 a 07. Cambiarlos ahora invalidaría la comparación con el baseline del stress.
- El streaming de la interfaz sigue en los SDK: la ganancia sería menor y el riesgo de romper el
  formulario, alto.
- Así se puede comparar: misma petición, mismo prompt, los dos caminos, y ver qué cambia.

**Aviso de seguridad del material**, que aquí se respeta: las versiones 1.82.7 y 1.82.8 de LiteLLM en PyPI
estuvieron comprometidas en marzo de 2026. La dependencia se fija con una versión mínima posterior a esas
dos, y `uv.lock` deja constancia de la versión exacta y de su hash.
"""

from __future__ import annotations

import logging

from app.config import Settings
from app.services.llm_service import (
    LLMProviderError,
    RenderedPrompt,
    TokenUsage,
)

logger = logging.getLogger(__name__)


def model_id(settings: Settings, model: str | None = None) -> str:
    """El identificador que entiende LiteLLM: `proveedor/modelo`, salvo OpenAI, que va sin prefijo."""
    nombre = model or settings.model
    if settings.llm_provider == "anthropic":
        return nombre if "/" in nombre else f"anthropic/{nombre}"
    return nombre


async def call_litellm(
    prompt: RenderedPrompt,
    settings: Settings,
    api_key: str,
    model: str | None = None,
    max_tokens: int | None = None,
) -> tuple[str, TokenUsage | None]:
    """La misma firma que `call_llm`, para que el desvío sea un `if` y no una refactorización.

    Toda la diferencia entre proveedores desaparece aquí: el system prompt es un mensaje más, la respuesta
    se lee siempre como `choices[0].message.content` y el uso como `usage.prompt_tokens`, venga de donde venga.
    """
    import litellm  # import perezoso: son ~100 MB de dependencias que no se cargan si el backend es `sdk`

    mensajes = [{"role": "system", "content": prompt.system}, *prompt.messages]
    try:
        response = await litellm.acompletion(
            model=model_id(settings, model),
            messages=mensajes,
            api_key=api_key,
            temperature=settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            num_retries=settings.litellm_retries,
        )
    except Exception as exc:  # noqa: BLE001 - LiteLLM normaliza los errores de todos los proveedores
        # Las excepciones de LiteLLM heredan de las de OpenAI, así que el mensaje ya viene normalizado.
        raise LLMProviderError(f"{type(exc).__name__}: {exc}") from exc

    choice = response.choices[0]
    texto = (choice.message.content or "").strip()
    if not texto:
        raise LLMProviderError(f"El proveedor ha devuelto una respuesta vacía (finish_reason={choice.finish_reason})")
    if choice.finish_reason == "length":
        logger.warning("La respuesta se ha cortado por el límite de tokens de salida")

    usage = None
    if getattr(response, "usage", None) is not None:
        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
    return texto, usage
