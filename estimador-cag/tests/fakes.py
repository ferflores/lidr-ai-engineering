"""Dobles del LLM para los tests de sesiones: responden según lo que reciben, sin llamar a ningún proveedor.

`FakeEstimatorLLM` imita a `openai.AsyncOpenAI` y genera una estimación con el formato de las plantillas,
de manera determinista a partir del prompt:

- El nombre del proyecto sale de los parámetros del mensaje de usuario o, si no se indican, del bloque
  `<project_metadata>` del system prompt (es la memoria lo que permite "recordarlo").
- Si en el mensaje de usuario aparece "Stripe" (por ejemplo, dentro de un adjunto), añade una tarea de
  pagos y el total cambia: así se comprueba que el contenido del documento influye en la salida.
- Si el prompt es el del summarizer (`prompts/resumen/v1`), devuelve un resumen determinista con el texto
  de los turnos que salen de la ventana (recortado), para que los tests puedan comprobar qué sobrevive.

Además guarda cada llamada (`calls`) para inspeccionar qué mensajes viajaron al modelo.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

NOMBRE_EN_USER = re.compile(r"- Nombre del proyecto: (.+)")
NOMBRE_EN_MEMORIA = re.compile(r"<project_metadata>.*?- Nombre del proyecto: (.+?)\n.*?</project_metadata>", re.DOTALL)
TURNO_RESUMEN = re.compile(r"### Turno (\S+)\nCliente:\n(.*?)\n\nEstimaci[oó]n resultante", re.DOTALL)
RESUMEN_ACTUAL = re.compile(r"## Resumen actual\n(.*?)\n\n## Turnos", re.DOTALL)
MAX_CHARS_RESUMEN = re.compile(r"m[aá]ximo (\d+) caracteres", re.IGNORECASE)


def estimacion_falsa(nombre: str, con_pagos: bool) -> str:
    tareas = ["Análisis y modelado de datos: 10 horas", "Desarrollo de la tienda: 60 horas"]
    if con_pagos:
        tareas.append("Integración de pagos con Stripe: 20 horas")
    subtotal = sum(int(re.search(r"(\d+) horas", t).group(1)) for t in tareas)
    contingencia = round(subtotal * 0.2)
    total = subtotal + contingencia
    desglose = "\n".join(f"{i}. {t}" for i, t in enumerate(tareas, start=1))
    return (
        f"## Estimación: {nombre}\n\n"
        "### Resumen del alcance\n"
        f"Tienda online para una panadería artesanal{' con pasarela de pago Stripe' if con_pagos else ''}.\n\n"
        f"### Desglose de tareas:\n{desglose}\n\n"
        f"**Subtotal de tareas: {subtotal} horas**\n"
        f"**Contingencia (20 %): {contingencia} horas**\n"
        f"**Total estimado: {total} horas**\n"
        f"**Coste estimado: {total * 50} €**\n"
        "**Equipo recomendado: 1 desarrollador full-stack, 1 diseñador UX/UI**\n"
        "**Duración estimada: 4-6 semanas**\n\n"
        "### Supuestos y riesgos\n- El catálogo lo carga el cliente.\n\n"
        "### Preguntas pendientes para el cliente\n- ¿Hay diseño previo?"
    )


def resumen_falso(user: str) -> str:
    """Resumen determinista: lo que ya había más, por cada turno que sale, los primeros 300 caracteres del cliente."""
    actual = RESUMEN_ACTUAL.search(user)
    partes = []
    if actual and not actual.group(1).strip().startswith("(vacío"):
        partes.append(actual.group(1).strip())
    for turno, texto in TURNO_RESUMEN.findall(user):
        texto = " ".join(texto.split())
        partes.append(f"Turno {turno}: {texto[:300]}")
    resumen = " || ".join(partes)
    limite = MAX_CHARS_RESUMEN.search(user)
    if limite:
        resumen = resumen[: int(limite.group(1))]
    return resumen or "Sin turnos que resumir."


def es_prompt_del_summarizer(system: str) -> bool:
    return "resumen acumulativo" in system.lower() and "estimador de software senior" not in system.lower()


class FakeEstimatorLLM:
    """Sustituto de `openai.AsyncOpenAI`: `monkeypatch.setattr(llm_service.openai, "AsyncOpenAI", fake)`."""

    def __init__(self):
        self.calls: list[dict] = []  # kwargs de cada chat.completions.create

        fake = self

        class _Client:
            def __init__(self, **kwargs):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc_info):
                return False

            async def _create(self, **kwargs):
                fake.calls.append(kwargs)
                messages = kwargs["messages"]
                system = messages[0]["content"]
                user = messages[-1]["content"]
                if es_prompt_del_summarizer(system):
                    content = resumen_falso(user)
                else:
                    nombre = "Proyecto sin nombre"
                    en_user = NOMBRE_EN_USER.search(user)
                    if en_user and not en_user.group(1).startswith("no indicado"):
                        nombre = en_user.group(1).strip()
                    else:
                        en_memoria = NOMBRE_EN_MEMORIA.search(system)
                        if en_memoria:
                            nombre = en_memoria.group(1).strip()
                    content = estimacion_falsa(nombre, con_pagos="stripe" in user.lower())
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
                    usage=SimpleNamespace(
                        prompt_tokens=sum(len(m["content"]) // 4 for m in messages),
                        completion_tokens=len(content) // 4 if es_prompt_del_summarizer(system) else 120,
                    ),
                )

        self.client_class = _Client

    # Permite usarlo directamente como clase sustituta: llm_service.openai.AsyncOpenAI = fake
    def __call__(self, **kwargs):
        return self.client_class(**kwargs)

    @property
    def messages_per_call(self) -> list[list[dict]]:
        return [call["messages"] for call in self.calls]

    @property
    def estimation_calls(self) -> list[dict]:
        """Solo las llamadas de estimación (sin las del summarizer)."""
        return [call for call in self.calls if not es_prompt_del_summarizer(call["messages"][0]["content"])]
