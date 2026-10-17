"""Guardrails de entrada y de salida: lo que el esquema no puede garantizar.

Hueco de la sesión 04 (bloque "Guardrails y validación de outputs"). La idea de partida del material es que
**la forma no garantiza el contenido**: una estimación puede validar contra el esquema y aun así venir de
una inyección de prompt, hablar de reformar un baño o inventarse una fase de negociación con la NASA.

La matriz del curso tiene dos ejes, entrada/salida × sintáctico/semántico. Lo sintáctico ya lo cubre Pydantic
gratis (`EstimationRequest` y `StructuredEstimation`); lo que falta y vive aquí es lo semántico:

| | Entrada | Salida |
|---|---|---|
| **Sintáctico** | `EstimationRequest` (ya existía) | `StructuredEstimation` (ya existía) |
| **Semántico** | este módulo: moderación + inyección + PII | este módulo: alcance, confianza y cifras disparatadas |

Y las **tres políticas de fallo** del material, que la regla obliga a declarar explícitamente para cada
guardrail en vez de dejarlo implícito:

- `BLOCK`: se aborta y el cliente recibe 400. Para lo grave: contenido marcado por moderación, inyección clara.
- `REPAIR`: se le devuelve el problema al modelo para que lo corrija. Ya lo hace el bucle de
  `structured_output.py`, así que aquí no se duplica.
- `DEGRADE`: no se aborta; se marca el resultado y se deja pasar. Para lo dudoso.

**Modo "solo registrar"**: `GUARDRAILS_MODE=log` (el valor por defecto) evalúa todo, lo registra y **no
bloquea nada**. Es la recomendación explícita del material: desplegar una o dos semanas observando, mirar
los falsos positivos, calibrar, y solo entonces poner `GUARDRAILS_MODE=enforce`. Cada falso positivo es un
usuario insatisfecho.

**Orden**: los guardrails de entrada van ANTES de la caché. Servir un acierto de caché saltándose la
validación convierte la caché en un vector de inyección, y cachear antes de validar propaga errores.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

import openai
import structlog

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)
events = structlog.get_logger("estimador.guardrails")

GUARDRAIL_TRIGGERED = "guardrail_triggered"


class Policy(str, Enum):
    """Qué hacer cuando un guardrail salta. Cada regla declara la suya; no hay comportamiento implícito."""

    BLOCK = "block"
    DEGRADE = "degrade"


@dataclass(frozen=True)
class Finding:
    """Un guardrail que ha saltado."""

    rule: str
    policy: Policy
    detail: str
    excerpt: str = ""


@dataclass
class GuardrailReport:
    """Resultado de pasar un texto por los guardrails."""

    findings: list[Finding] = field(default_factory=list)
    enforced: bool = True       # False en modo "solo registrar": se evalúa pero no se bloquea
    checked_moderation: bool = False

    @property
    def blocked(self) -> bool:
        return self.enforced and any(f.policy is Policy.BLOCK for f in self.findings)

    @property
    def degraded(self) -> bool:
        return any(f.policy is Policy.DEGRADE for f in self.findings)

    @property
    def rules(self) -> list[str]:
        return [f.rule for f in self.findings]

    def reason(self) -> str:
        """Mensaje para el cliente: dice qué pasó y qué hacer, sin filtrar el patrón que lo detectó."""
        bloqueos = [f for f in self.findings if f.policy is Policy.BLOCK]
        if not bloqueos:
            return ""
        return "; ".join(f.detail for f in bloqueos)


# --- Entrada: inyección de prompt -----------------------------------------------------------------
#
# Heurísticas deliberadamente simples. El material es explícito: por sí solas son frágiles, pero
# combinadas con un prompt robusto y validación de salida cubren la mayoría de los casos reales.
# Se cubren los dos idiomas del producto: el prompt está en español y los intentos suelen venir en inglés.

INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"ignor[ae][a-z]*\s+(todas?\s+)?(las\s+)?(instrucciones|indicaciones|reglas)", "ignorar instrucciones (es)"),
    (r"ignore\s+(all\s+|any\s+)?(previous|prior|above)\s+instructions?", "ignorar instrucciones (en)"),
    (r"olvida\s+(todo\s+)?(lo\s+anterior|las\s+instrucciones)", "olvidar el contexto (es)"),
    (r"disregard\s+(the\s+)?(above|previous|system)", "descartar el contexto (en)"),
    (r"(ahora\s+eres|a\s+partir\s+de\s+ahora\s+eres)\b", "reasignación de rol (es)"),
    (r"you\s+are\s+now\s+(a|an|the)\b", "reasignación de rol (en)"),
    (r"(system\s*prompt|prompt\s+del\s+sistema)", "referencia al system prompt"),
    (r"</?(system|instructions|project_metadata|anchors|summary)>", "etiqueta de bloque del prompt"),
    (r"(revela|muestra|imprime)\s+(tus\s+)?(instrucciones|prompt)", "exfiltración del prompt (es)"),
    (r"(reveal|show|print|repeat)\s+(your\s+)?(instructions|system prompt|prompt)", "exfiltración del prompt (en)"),
)

# PII que no tiene por qué viajar a un proveedor externo dentro de una transcripción de requisitos.
PII_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b[A-Z]{2}\d{2}[ ]?(?:\d{4}[ ]?){4}\d{0,4}\b", "IBAN"),
    (r"\b(?:\d[ -]?){13,19}\b", "posible número de tarjeta"),
    (r"\b\d{8}[A-HJ-NP-TV-Z]\b", "DNI"),
)

_INJECTION = tuple((re.compile(p, re.IGNORECASE), n) for p, n in INJECTION_PATTERNS)
_PII = tuple((re.compile(p), n) for p, n in PII_PATTERNS)


def _excerpt(text: str, match: re.Match) -> str:
    inicio, fin = max(0, match.start() - 25), min(len(text), match.end() + 25)
    return ("…" if inicio else "") + text[inicio:fin].replace("\n", " ") + ("…" if fin < len(text) else "")


def scan_injection(text: str) -> list[Finding]:
    """Patrones de inyección. Política BLOCK: quien pide esto no está pidiendo una estimación."""
    hallazgos = []
    for patron, nombre in _INJECTION:
        match = patron.search(text)
        if match:
            hallazgos.append(
                Finding(
                    rule=f"injection:{nombre}",
                    policy=Policy.BLOCK,
                    detail="La transcripción contiene instrucciones dirigidas al modelo, no requisitos del cliente.",
                    excerpt=_excerpt(text, match),
                )
            )
    return hallazgos


def scan_pii(text: str) -> list[Finding]:
    """PII evidente. Política DEGRADE: un IBAN en una transcripción suele ser un descuido, no un ataque;
    se avisa y se registra, pero no se le tira la petición al usuario a la cara."""
    hallazgos = []
    for patron, nombre in _PII:
        match = patron.search(text)
        if match:
            hallazgos.append(
                Finding(
                    rule=f"pii:{nombre}",
                    policy=Policy.DEGRADE,
                    detail=f"La transcripción parece contener {nombre}; conviene retirarlo antes de enviarla.",
                    excerpt="(no se registra el valor)",
                )
            )
    return hallazgos


async def moderate(text: str, settings: Settings) -> list[Finding]:
    """Moderación de OpenAI. Gratis y de 50 a 100 ms, así que debería estar en cualquier servicio que
    acepte texto de usuarios. Si la llamada falla, no se bloquea la estimación por ello."""
    api_key = (settings.openai_api_key or "").strip()
    if not api_key:
        return []
    try:
        async with openai.AsyncOpenAI(api_key=api_key, timeout=10.0) as client:
            response = await client.moderations.create(model="omni-moderation-latest", input=text[:4000])
    except Exception as exc:  # noqa: BLE001
        # Cualquier fallo, no solo los de la API: en los tests y en los evals el cliente está sustituido por
        # un doble que no implementa moderación. Un guardrail que no puede ejecutarse no debe impedir
        # generar la estimación; se registra y se sigue sin él.
        logger.warning("La moderación no ha podido ejecutarse (%s); se continúa sin ella", type(exc).__name__)
        return []

    resultado = response.results[0]
    if not resultado.flagged:
        return []
    categorias = sorted(c for c, activa in resultado.categories.model_dump().items() if activa)
    return [
        Finding(
            rule="moderation",
            policy=Policy.BLOCK,
            detail="El texto ha sido marcado por el filtro de contenido.",
            excerpt=", ".join(categorias[:4]),
        )
    ]


async def check_input(text: str, settings: Settings | None = None) -> GuardrailReport:
    """Guardrails de entrada. Se llama ANTES de la caché y antes del LLM."""
    settings = settings or get_settings()
    if settings.guardrails_mode == "off":
        return GuardrailReport(enforced=False)

    hallazgos = scan_injection(text) + scan_pii(text)
    checked = False
    if settings.guardrails_moderation:
        hallazgos += await moderate(text, settings)
        checked = True

    report = GuardrailReport(
        findings=hallazgos,
        enforced=settings.guardrails_mode == "enforce",
        checked_moderation=checked,
    )
    _emit("input", report, len(text))
    return report


# --- Salida: ¿tiene sentido lo que ha vuelto? ------------------------------------------------------

# Señales de que el modelo se ha ido del dominio. Si la estimación habla de esto, no es software.
OUT_OF_SCOPE_HINTS: tuple[str, ...] = (
    "reforma", "fontaner", "albañil", "pintar la", "mudanza", "catering", "cuarto de baño", "cocina nueva",
)
# Una estimación de desarrollo que no menciona nada de esto es sospechosa.
DOMAIN_HINTS: tuple[str, ...] = (
    "desarrollo", "software", "api", "web", "frontend", "backend", "testing", "qa", "despliegue",
    "integración", "aplicación", "base de datos", "diseño", "móvil", "análisis",
)

MAX_PLAUSIBLE_HOURS = 20_000      # ~10 años-persona: por encima de esto no es una estimación, es un error
MIN_CONFIDENCE_ITEMS = 2          # un desglose de una sola línea no es un desglose


def check_structured_output(estimation, *, settings: Settings | None = None) -> GuardrailReport:
    """Validación semántica de la salida, sobre el objeto ya validado sintácticamente.

    Todo DEGRADE a propósito: a estas alturas el usuario ya ha pagado la llamada, y una estimación
    sospechosa con un aviso es más útil que un error. Lo que no se hace es dejarla pasar en silencio.
    """
    settings = settings or get_settings()
    if settings.guardrails_mode == "off":
        return GuardrailReport(enforced=False)

    hallazgos: list[Finding] = []
    texto = f"{estimation.title} {estimation.scope_summary} {' '.join(i.name for i in estimation.items)}".lower()

    fuera = [p for p in OUT_OF_SCOPE_HINTS if p in texto]
    if fuera:
        hallazgos.append(
            Finding(rule="scope:fuera-de-dominio", policy=Policy.DEGRADE,
                    detail="La estimación no parece de un proyecto de software.", excerpt=", ".join(fuera[:3]))
        )
    elif not any(p in texto for p in DOMAIN_HINTS):
        hallazgos.append(
            Finding(rule="scope:sin-señales-de-dominio", policy=Policy.DEGRADE,
                    detail="La estimación no menciona ninguna actividad reconocible de desarrollo de software.")
        )

    if estimation.total_hours > MAX_PLAUSIBLE_HOURS:
        hallazgos.append(
            Finding(rule="rango:horas-implausibles", policy=Policy.DEGRADE,
                    detail=f"El total de {estimation.total_hours:g} horas está fuera de lo plausible.")
        )
    if len(estimation.items) < MIN_CONFIDENCE_ITEMS:
        hallazgos.append(
            Finding(rule="detalle:desglose-pobre", policy=Policy.DEGRADE,
                    detail="El desglose tiene una sola línea; no permite revisar la estimación.")
        )
    if estimation.confidence == "baja" and not estimation.open_questions:
        hallazgos.append(
            Finding(rule="confianza:baja-sin-preguntas", policy=Policy.DEGRADE,
                    detail="La confianza es baja pero no se plantea ninguna pregunta al cliente.")
        )

    report = GuardrailReport(findings=hallazgos, enforced=settings.guardrails_mode == "enforce")
    _emit("output", report, 0)
    return report


def _emit(stage: str, report: GuardrailReport, text_chars: int) -> None:
    """Un evento structlog por evaluación con hallazgos. Es lo que se mira durante las semanas de
    'solo registrar' para calibrar los umbrales antes de activar el bloqueo."""
    if not report.findings:
        return
    events.info(
        GUARDRAIL_TRIGGERED,
        stage=stage,
        mode="enforce" if report.enforced else "log",
        blocked=report.blocked,
        rules=report.rules,
        policies=[f.policy.value for f in report.findings],
        details=[f.detail for f in report.findings],
        excerpts=[f.excerpt for f in report.findings if f.excerpt],
        text_chars=text_chars,
    )
