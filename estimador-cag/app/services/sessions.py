"""Estado conversacional del estimador: historial con ventana deslizante y memoria del proyecto.

Son dos cosas distintas y conviene no mezclarlas:

- **Historial** (`ConversationHistory`): los mensajes `user`/`assistant` que viajan al LLM en el array
  `messages` de cada llamada. Es texto bruto, crece con cada turno y por eso se recorta con una ventana
  deslizante: se conservan los últimos `max_turns` turnos (un turno = un par user+assistant) y los más
  antiguos se descartan.
- **Memoria** (`ProjectMetadata`): los hechos del proyecto en curso (nombre, equipo asumido, tecnologías,
  alcance acordado...) que se extraen de cada turno y se inyectan en el system prompt. Es conocimiento
  estructurado y pequeño, y sobrevive a la ventana: un turno puede haber salido del historial y el dato
  que aportó sigue en la memoria.

Ambas viven dentro de una `Session`, y las sesiones en un `SessionStore`: un diccionario en memoria del
proceso indexado por `session_id`.

Por qué se acepta esa volatilidad en esta fase (sin base de datos, sin Redis):

- El objetivo del ejercicio es separar historial de memoria y gobernar lo que viaja al LLM; persistir no
  aporta nada a eso y añade infraestructura que hay que levantar hasta para pasar los tests.
- Una sesión de estimación dura minutos. Perderla al reiniciar el servicio es aceptable mientras haya un
  solo proceso; con varios workers o réplicas (cada uno con su diccionario) o si se quiere retomar una
  sesión al día siguiente, deja de serlo.
- La interfaz del `SessionStore` (`create`, `get`, `delete`) es la misma que tendría uno respaldado en
  Redis o Postgres: cambiar la implementación no toca el router ni el servicio.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.estimation import ProjectType

DEFAULT_MAX_TURNS = 6
DEFAULT_MAX_SESSIONS = 500


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --- Historial: ventana deslizante -----------------------------------------------------


@dataclass
class Turn:
    """Un turno de la conversación: lo que envió el usuario y lo que respondió el modelo."""

    user: str
    assistant: str | None = None
    created_at: datetime = field(default_factory=_now)

    @property
    def is_complete(self) -> bool:
        return self.assistant is not None


class ConversationHistory:
    """Lista limitada de turnos con ventana deslizante.

    - El system prompt NO se guarda aquí: es un invariante que se regenera en cada turno a partir del
      `ProjectMetadata` actual y se antepone en `to_messages_list()`. Así nunca sale de la ventana y
      siempre refleja la memoria más reciente.
    - Un turno empieza con `start_turn(user)` (queda incompleto mientras el LLM responde) y se cierra con
      `complete_turn(assistant)`. Si la llamada falla, `discard_incomplete_turn()` lo retira para que un
      error no deje un mensaje de usuario huérfano en el historial.
    - La ventana cuenta turnos, no mensajes: con `max_turns=6` viajan como mucho 6 mensajes de usuario
      (los 5 anteriores con su respuesta más el actual) además del system prompt.
    """

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS):
        if max_turns < 1:
            raise ValueError("max_turns debe ser al menos 1")
        self.max_turns = max_turns
        self.turns: list[Turn] = []
        self.dropped_turns = 0  # cuántos turnos han salido ya por la ventana (para mostrarlo y medirlo)

    def __len__(self) -> int:
        return len(self.turns)

    @property
    def completed_turns(self) -> list[Turn]:
        return [turn for turn in self.turns if turn.is_complete]

    def start_turn(self, user_content: str) -> Turn:
        """Añade el mensaje del usuario del turno actual y recorta la ventana para que quepa."""
        if self.turns and not self.turns[-1].is_complete:
            raise RuntimeError("Ya hay un turno en curso: complétalo o descártalo antes de empezar otro")
        turn = Turn(user=user_content)
        self.turns.append(turn)
        self._trim()
        return turn

    def complete_turn(self, assistant_content: str) -> Turn:
        """Cierra el turno en curso con la respuesta del modelo."""
        if not self.turns or self.turns[-1].is_complete:
            raise RuntimeError("No hay ningún turno en curso que completar")
        self.turns[-1].assistant = assistant_content
        return self.turns[-1]

    def discard_incomplete_turn(self) -> bool:
        """Retira el turno en curso (la llamada al LLM falló). Devuelve True si había alguno."""
        if self.turns and not self.turns[-1].is_complete:
            self.turns.pop()
            return True
        return False

    def _trim(self) -> None:
        """Ventana deslizante: descarta los turnos más antiguos hasta quedarse con `max_turns`."""
        while len(self.turns) > self.max_turns:
            self.turns.pop(0)
            self.dropped_turns += 1

    def messages(self) -> list[dict[str, str]]:
        """Los mensajes user/assistant de la ventana, sin system prompt, en el formato común a los proveedores."""
        result: list[dict[str, str]] = []
        for turn in self.turns:
            result.append({"role": "user", "content": turn.user})
            if turn.assistant is not None:
                result.append({"role": "assistant", "content": turn.assistant})
        return result

    def to_messages_list(self, system_prompt: str) -> list[dict[str, str]]:
        """El array `messages` listo para la API del LLM: system prompt (regenerado fuera, con la memoria
        actual) seguido de los turnos que caben en la ventana."""
        return [{"role": "system", "content": system_prompt}, *self.messages()]


# --- Memoria: hechos del proyecto en curso ------------------------------------------------


class ProjectMetadata(BaseModel):
    """Lo que el sistema "sabe" del proyecto en curso, independientemente de qué turnos sigan en la ventana.

    Se rellena después de cada respuesta del LLM (ver `app/services/metadata_extractor.py`) y se inyecta
    en el bloque `<project_metadata>` del system prompt en el turno siguiente. Todos los campos son
    opcionales: en la primera llamada de la sesión el bloque va vacío.
    """

    model_config = ConfigDict(extra="forbid")

    project_name: str | None = Field(default=None, description="Nombre del proyecto (del formulario o del título de la estimación).")
    project_type: ProjectType | None = Field(default=None, description="Tipo de proyecto indicado en el formulario.")
    assumed_team: str | None = Field(default=None, description="Equipo recomendado en la última estimación, tal cual.")
    assumed_team_size: int | None = Field(default=None, ge=1, description="Número de personas deducido del equipo recomendado.")
    mentioned_technologies: list[str] = Field(default_factory=list, description="Tecnologías mencionadas en transcripciones, adjuntos y estimaciones.")
    agreed_scope: str | None = Field(default=None, description="Resumen del alcance de la última estimación (lo acordado hasta ahora).")
    hourly_rate_eur: float | None = Field(default=None, description="Última tarifa usada.")
    deadline_weeks: int | None = Field(default=None, description="Último plazo objetivo indicado.")
    last_total_hours: float | None = Field(default=None, description="Total de horas de la última estimación.")
    attached_documents: list[str] = Field(default_factory=list, description="Nombres de los documentos adjuntados en la sesión.")
    updated_at: datetime | None = Field(default=None, description="Última actualización de la memoria.")

    def is_empty(self) -> bool:
        """True si todavía no se sabe nada del proyecto (primera llamada de la sesión)."""
        return not any(
            value for name, value in self.model_dump().items() if name != "updated_at"
        )

    def merged(self, update: ProjectMetadata) -> ProjectMetadata:
        """Memoria resultante de incorporar lo extraído en un turno nuevo.

        Regla: un hecho nuevo sustituye al anterior (el turno más reciente manda), un hecho ausente no
        borra nada, y las listas se acumulan sin duplicados conservando el orden de aparición.
        """
        data = self.model_dump()
        for name, value in update.model_dump().items():
            if name == "updated_at":
                continue
            if isinstance(value, list):
                data[name] = _union(data[name], value)
            elif value is not None:
                data[name] = value
        data["updated_at"] = _now()
        return ProjectMetadata(**data)

    def prompt_variables(self) -> dict[str, object] | None:
        """Lo que recibe la plantilla del system prompt: los hechos, o None si no hay ninguno."""
        if self.is_empty():
            return None
        return self.model_dump(exclude={"updated_at"})


def _union(current: list[str], new: list[str]) -> list[str]:
    seen = {item.casefold() for item in current}
    result = list(current)
    for item in new:
        if item.casefold() not in seen:
            seen.add(item.casefold())
            result.append(item)
    return result


# --- Sesión y almacén --------------------------------------------------------------------


class SessionNotFound(KeyError):
    """No existe ninguna sesión con ese identificador (o el servicio se reinició y se perdió)."""


@dataclass
class Session:
    """Una conversación de estimación: su historial (ventana) y su memoria (hechos del proyecto)."""

    session_id: str
    history: ConversationHistory
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    turn_count: int = 0  # turnos completados en toda la sesión, hayan salido o no de la ventana
    # Un turno a la vez por sesión: dos peticiones concurrentes con el mismo session_id se serializan para
    # que el historial no intercale mensajes de usuario sin su respuesta.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)

    def touch(self) -> None:
        self.updated_at = _now()


class SessionStore:
    """Diccionario de sesiones en memoria del proceso (ver la nota sobre volatilidad en el módulo).

    Con un tope de sesiones vivas: al superarlo se expulsa la que lleva más tiempo sin usarse, para que
    un cliente que cree sesiones sin parar no agote la memoria del servicio.
    """

    def __init__(self, max_sessions: int = DEFAULT_MAX_SESSIONS):
        self.max_sessions = max_sessions
        self._sessions: dict[str, Session] = {}

    def __len__(self) -> int:
        return len(self._sessions)

    def create(self, max_turns: int = DEFAULT_MAX_TURNS) -> Session:
        """Crea una sesión vacía con un identificador UUID v4."""
        if len(self._sessions) >= self.max_sessions:
            oldest = min(self._sessions.values(), key=lambda s: s.updated_at)
            del self._sessions[oldest.session_id]
        session = Session(session_id=str(uuid.uuid4()), history=ConversationHistory(max_turns=max_turns))
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise SessionNotFound(
                f"No existe la sesión '{session_id}'. Crea una con POST /api/v1/sessions "
                "(las sesiones viven en memoria del proceso: se pierden al reiniciar el servicio)."
            ) from None

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def clear(self) -> None:
        self._sessions.clear()


_store = SessionStore()


def get_session_store() -> SessionStore:
    """Almacén único de sesiones del proceso (se usa como dependencia de FastAPI; los tests lo sustituyen)."""
    return _store
