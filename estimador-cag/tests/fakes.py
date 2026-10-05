"""Dobles del LLM para los tests de sesiones: responden según lo que reciben, sin llamar a ningún proveedor.

`FakeEstimatorLLM` imita a `openai.AsyncOpenAI` y genera una estimación con el formato de las plantillas,
de manera determinista a partir del prompt:

- El nombre del proyecto sale de los parámetros del mensaje de usuario o, si no se indican, del bloque
  `<project_metadata>` del system prompt (es la memoria lo que permite "recordarlo").
- Si en el mensaje de usuario aparece "Stripe" (por ejemplo, dentro de un adjunto), añade una tarea de
  pagos y el total cambia: así se comprueba que el contenido del documento influye en la salida.

Además guarda cada llamada (`calls`) para inspeccionar qué mensajes viajaron al modelo.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

NOMBRE_EN_USER = re.compile(r"- Nombre del proyecto: (.+)")
NOMBRE_EN_MEMORIA = re.compile(r"<project_metadata>.*?- Nombre del proyecto: (.+?)\n.*?</project_metadata>", re.DOTALL)


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
                    usage=SimpleNamespace(prompt_tokens=sum(len(m["content"]) // 4 for m in messages), completion_tokens=120),
                )

        self.client_class = _Client

    # Permite usarlo directamente como clase sustituta: llm_service.openai.AsyncOpenAI = fake
    def __call__(self, **kwargs):
        return self.client_class(**kwargs)

    @property
    def messages_per_call(self) -> list[list[dict]]:
        return [call["messages"] for call in self.calls]
