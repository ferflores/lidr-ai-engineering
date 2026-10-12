"""Anclas heurísticas: hechos cortos del cliente que deben sobrevivir a la ventana deslizante.

La ventana descarta turnos enteros; `project_metadata` guarda lo que el *modelo* concluyó (alcance, equipo,
horas) y lo que vino por el formulario. Lo que decía el *cliente* en un turno (un presupuesto, un plazo, un
requisito concreto, un cambio de stack) se perdía al salir ese turno de la ventana. Las anclas cubren ese hueco:
se extraen del mensaje del usuario de cada turno con reglas deterministas (sin LLM) y viajan en el bloque
`<anchors>` del system prompt (plantilla `v4`) mientras la sesión viva.

Reglas:

| kind         | Qué captura                                                     | Fusión                      |
|--------------|-----------------------------------------------------------------|-----------------------------|
| name         | nombre del proyecto (formulario o "el proyecto se llama X")     | el más reciente sustituye   |
| budget       | "presupuesto de 30.000 €", "budget 80k€"                        | el más reciente sustituye   |
| deadline     | "plazo de 12 semanas", `deadline_weeks` del formulario          | el más reciente sustituye   |
| change       | frases con "en vez de", "ya no", "cambiamos a", "descartamos"...| se acumulan                 |
| stack        | tecnologías del vocabulario de metadata_extractor               | se acumulan                 |
| requirement  | frases con "necesita", "quiere", "debe", "hay que", "añadir"... | se acumulan                 |

Un turno aporta como mucho `anchors_per_turn` anclas (en ese orden de prioridad) y la sesión conserva como
mucho `anchors_max`; al superar el tope se descartan las acumulables más antiguas. Los hechos de un solo
valor (name, budget, deadline) siguen la misma regla que la memoria: el dato más reciente manda. Eso es una
decisión medible: en el escenario "contradicción" del stress el presupuesto antiguo solo sobrevive si el
resumen acumulativo lo menciona.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from app.schemas.estimation import EstimationRequest
from app.services.metadata_extractor import find_technologies
from app.services.sessions import Anchor

SINGLE_VALUED_KINDS: frozenset[str] = frozenset({"name", "budget", "deadline"})
KIND_PRIORITY: tuple[str, ...] = ("name", "budget", "deadline", "change", "stack", "requirement")
MAX_ANCHOR_CHARS = 160

_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
_NAME_IN_TEXT = re.compile(
    r"(?:el proyecto|la app|la aplicaci[oó]n|la plataforma|el producto|el sistema|la herramienta)\s+"
    r"(?:se llama(?:r[aá])?|se llamar[ií]a|llamad[oa]|bautizad[oa] como|con nombre)\s+[«\"']?((?-i:[A-ZÁÉÍÓÚÑ])[\w.\-]{1,40})",
    re.IGNORECASE,
)
_NAME_LABEL = re.compile(r"nombre del proyecto\s*[:=]\s*[«\"']?([A-ZÁÉÍÓÚÑ][\w.\- ]{1,40}?)[»\"']?(?=[.,;\n]|$)", re.IGNORECASE)
_BUDGET = re.compile(
    r"(?:presupuesto|budget|inversi[oó]n|tope de gasto)[^.\n;]{0,60}?(\d[\d.,]*\s*(?:k|mil)?\s*(?:€|eur(?:os)?|euros))",
    re.IGNORECASE,
)
_DEADLINE = re.compile(
    r"(?:plazo|entrega|lanzamiento|lanzar|listo|en producci[oó]n|salir|deadline)[^.\n;]{0,60}?(\d+\s*(?:semanas?|meses|d[ií]as))",
    re.IGNORECASE,
)
_CHANGE = re.compile(
    r"\b(?:en vez de|en lugar de|ya no|cambiamos|cambiar(?:emos)? a|sustitui(?:mos|r)|descarta(?:mos|r)|"
    r"pasamos de|dejamos de|en su lugar|finalmente|olvidad|se cae|se sustituye|rectific)\w*",
    re.IGNORECASE,
)
_REQUIREMENT = re.compile(
    r"\b(?:necesita(?:mos|n)?|quiere(?:n)?|requisito|imprescindible|obligatori[oa]|debe(?:n|r[ií]a)?|"
    r"tiene(?:n)? que|hay que|a[ñn]ad(?:ir|imos|en|e)|incluir|incorporar|hace falta|exigen?|piden)\b",
    re.IGNORECASE,
)


def extract_anchors(transcript: str, request: EstimationRequest | None, turn: int, per_turn: int = 6) -> list[Anchor]:
    """Anclas que aporta el mensaje del usuario de un turno (transcripción sin adjuntos + parámetros)."""
    found: list[Anchor] = []
    seen: set[str] = set()

    def add(kind: str, text: str) -> None:
        text = _clip(" ".join(text.split()))
        if not text:
            return
        anchor = Anchor(kind=kind, text=text, turn=turn)
        if anchor.key not in seen:
            seen.add(anchor.key)
            found.append(anchor)

    # name: el formulario manda; si no, la transcripción
    if request is not None and request.project_name:
        add("name", f"Proyecto: {request.project_name}")
    else:
        match = _NAME_IN_TEXT.search(transcript) or _NAME_LABEL.search(transcript)
        if match:
            add("name", f"Proyecto: {match.group(1).strip(' .,;')}")

    for match in _BUDGET.finditer(transcript):
        add("budget", f"Presupuesto: {match.group(1)}")
    for match in _DEADLINE.finditer(transcript):
        add("deadline", f"Plazo: {match.group(1)}")
    if request is not None and request.deadline_weeks and not any(a.kind == "deadline" for a in found):
        add("deadline", f"Plazo objetivo: {request.deadline_weeks} semanas")

    sentences = [s.strip() for s in _SENTENCES.split(transcript) if s and s.strip()]
    for sentence in sentences:
        if _CHANGE.search(sentence):
            add("change", f"Cambio: {sentence}")
    for technology in find_technologies([transcript]):
        add("stack", f"Stack: {technology}")
    for sentence in sentences:
        if _REQUIREMENT.search(sentence) and not _CHANGE.search(sentence):
            add("requirement", f"Requisito: {sentence}")

    # Prioridad por tipo y tope por turno: un turno largo no puede inundar la sesión de requisitos
    found.sort(key=lambda a: KIND_PRIORITY.index(a.kind))
    return found[:per_turn]


def merge_anchors(current: Sequence[Anchor], new: Iterable[Anchor], max_anchors: int) -> list[Anchor]:
    """Anclas de la sesión tras incorporar las de un turno nuevo.

    - name / budget / deadline: el valor nuevo sustituye al anterior (el dato más reciente manda).
    - El resto se acumula sin duplicados (misma clase y mismo texto, sin distinguir mayúsculas).
    - Si se supera `max_anchors`, se descartan las acumulables más antiguas; las de un solo valor no se descartan.
    """
    if max_anchors <= 0:
        return []
    result = list(current)
    for anchor in new:
        if anchor.kind in SINGLE_VALUED_KINDS:
            result = [a for a in result if a.kind != anchor.kind]
            result.append(anchor)
        elif anchor.key not in {a.key for a in result}:
            result.append(anchor)
    while len(result) > max_anchors:
        oldest = min((a for a in result if a.kind not in SINGLE_VALUED_KINDS), key=lambda a: a.turn, default=None)
        if oldest is None:
            break
        result.remove(oldest)
    return result


def anchors_text(anchors: Sequence[Anchor]) -> str:
    """Las anclas como texto plano (para buscar hechos en ellas y para mostrarlas)."""
    return "\n".join(f"[turno {a.turn}] {a.text}" for a in anchors)


def _clip(text: str) -> str:
    return text if len(text) <= MAX_ANCHOR_CHARS else text[: MAX_ANCHOR_CHARS - 1].rstrip() + "…"
