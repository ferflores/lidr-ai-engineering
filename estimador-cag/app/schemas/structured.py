"""Contrato de salida de la estimación: el mismo objeto que hoy se devuelve como Markdown, pero tipado.

Hueco de la sesión 04 (bloque "Extracción de datos estructurados"): *un producto no consume texto, consume
datos*. Hasta ahora el endpoint devolvía la estimación en un campo de texto y quien la quisiera usar tenía
que parsearla con expresiones regulares, que es justo lo que el material desaconseja.

Este módulo define el **shape rico** y es la única fuente de verdad: de él salen la clase Python con
validación, el JSON Schema que viaja al modelo (`model_json_schema()`) y la documentación de OpenAPI.

Decisión de la sesión 04 que se respeta aquí: el servicio devuelve siempre la misma estructura, nunca la
presentación. `detail_level` sigue modulando cuánto detalle pide el prompt, pero no cambia el esquema. Si
mañana hay que sacar un PDF o un mensaje de Slack, se construye desde este objeto sin tocar el servicio.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Márgenes de coherencia interna. No son redondeos del modelo: son el contrato de que la aritmética cuadra.
HOURS_TOLERANCE = 1.0        # horas de desviación admitidas entre la suma de los bloques y el subtotal
COST_TOLERANCE_PCT = 0.02    # 2 % de desviación admitida entre horas × tarifa y el coste declarado


class EstimationItem(BaseModel):
    """Una línea del desglose: una tarea o un bloque de trabajo con sus horas."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Nombre de la tarea o bloque.")
    hours: float = Field(ge=0, le=5000, description="Horas estimadas para esta línea.")
    note: str | None = Field(default=None, description="Matiz o supuesto propio de esta línea.")


class StructuredEstimation(BaseModel):
    """La estimación como objeto. Equivale a lo que hoy devuelve el Markdown, campo a campo."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, description="Título de la estimación, normalmente el nombre del proyecto.")
    scope_summary: str = Field(min_length=1, description="Resumen del alcance en una o dos frases.")
    items: list[EstimationItem] = Field(min_length=1, description="Desglose del trabajo.")

    subtotal_hours: float = Field(ge=0, description="Suma de las horas de los bloques, sin contingencia.")
    contingency_pct: float = Field(ge=0, le=100, description="Porcentaje de contingencia aplicado.")
    total_hours: float = Field(ge=0, description="Subtotal más contingencia.")

    hourly_rate_eur: float = Field(gt=0, description="Tarifa por hora aplicada, en euros.")
    total_cost_eur: float = Field(ge=0, description="Coste total en euros: total_hours × hourly_rate_eur.")

    team: str = Field(min_length=1, description="Equipo recomendado, en texto libre.")
    duration_weeks_min: int = Field(ge=0, le=260)
    duration_weeks_max: int = Field(ge=0, le=260)

    deadline_weeks: int | None = Field(default=None, description="Plazo objetivo que pidió el cliente, si lo hay.")
    deadline_feasible: bool | None = Field(
        default=None,
        description="Si el plazo objetivo es viable con el equipo propuesto. Null cuando no hay plazo.",
    )

    assumptions: list[str] = Field(default_factory=list, description="Supuestos sobre los que se estima.")
    risks: list[str] = Field(default_factory=list, description="Riesgos identificados.")
    open_questions: list[str] = Field(default_factory=list, description="Preguntas pendientes para el cliente.")

    confidence: Literal["baja", "media", "alta"] = Field(description="Confianza en la estimación.")

    # --- Coherencia interna: lo que un schema solo no garantiza ---------------------------------

    @model_validator(mode="after")
    def _items_add_up(self) -> "StructuredEstimation":
        suma = sum(item.hours for item in self.items)
        if abs(suma - self.subtotal_hours) > HOURS_TOLERANCE:
            raise ValueError(
                f"las horas de los bloques suman {suma:g} pero subtotal_hours dice {self.subtotal_hours:g}"
            )
        return self

    @model_validator(mode="after")
    def _contingency_adds_up(self) -> "StructuredEstimation":
        esperado = self.subtotal_hours * (1 + self.contingency_pct / 100)
        if abs(esperado - self.total_hours) > HOURS_TOLERANCE:
            raise ValueError(
                f"subtotal {self.subtotal_hours:g} + {self.contingency_pct:g} % de contingencia son "
                f"{esperado:g} horas, pero total_hours dice {self.total_hours:g}"
            )
        return self

    @model_validator(mode="after")
    def _cost_matches_hours(self) -> "StructuredEstimation":
        esperado = self.total_hours * self.hourly_rate_eur
        margen = max(esperado * COST_TOLERANCE_PCT, 1.0)
        if abs(esperado - self.total_cost_eur) > margen:
            raise ValueError(
                f"{self.total_hours:g} horas × {self.hourly_rate_eur:g} €/h son {esperado:,.0f} €, "
                f"pero total_cost_eur dice {self.total_cost_eur:,.0f} €"
            )
        return self

    @model_validator(mode="after")
    def _duration_range_is_ordered(self) -> "StructuredEstimation":
        if self.duration_weeks_max < self.duration_weeks_min:
            raise ValueError(
                f"duration_weeks_max ({self.duration_weeks_max}) es menor que duration_weeks_min ({self.duration_weeks_min})"
            )
        return self

    @model_validator(mode="after")
    def _deadline_consistency(self) -> "StructuredEstimation":
        """Si hay plazo objetivo, hay que pronunciarse sobre él. Es la regla que el prompt v2 ya pedía en texto."""
        if self.deadline_weeks is not None and self.deadline_feasible is None:
            raise ValueError("hay un plazo objetivo (deadline_weeks) pero deadline_feasible viene vacío")
        return self

    # --- Presentación: se construye desde el objeto, nunca al revés ------------------------------

    def to_markdown(self) -> str:
        """Render en el mismo formato que devuelve el modo Markdown. Vive aquí para dejar claro que la
        presentación se deriva de los datos, que es la decisión de diseño del bloque 3 de la sesión 04."""
        lineas = [f"## Estimación: {self.title}", "", "### Resumen del alcance", self.scope_summary, "", "### Desglose:"]
        for i, item in enumerate(self.items, start=1):
            nota = f" — {item.note}" if item.note else ""
            lineas.append(f"{i}. {item.name}: {item.hours:g} horas{nota}")
        lineas += [
            "",
            f"**Subtotal: {self.subtotal_hours:g} horas**",
            f"**Contingencia ({self.contingency_pct:g} %): {self.total_hours - self.subtotal_hours:g} horas**",
            f"**Total estimado: {self.total_hours:g} horas**",
            f"**Coste estimado: {self.total_cost_eur:,.0f} €** (total × {self.hourly_rate_eur:g} €/hora)",
            f"**Equipo recomendado: {self.team}**",
            f"**Duración estimada: {self.duration_weeks_min}-{self.duration_weeks_max} semanas**",
        ]
        if self.deadline_weeks is not None:
            veredicto = "viable" if self.deadline_feasible else "no viable"
            lineas.append(f"**Plazo objetivo: {self.deadline_weeks} semanas — {veredicto}**")
        lineas.append(f"**Confianza: {self.confidence}**")
        for titulo, valores in (
            ("Supuestos", self.assumptions),
            ("Riesgos", self.risks),
            ("Preguntas pendientes para el cliente", self.open_questions),
        ):
            if valores:
                lineas += ["", f"### {titulo}"] + [f"- {v}" for v in valores]
        return "\n".join(lineas)
