"""Escenarios sintéticos multi-turno para el stress del CAG (sesión 06, bloque 2).

Tres perfiles sobre un mismo proyecto cada uno, con 20 turnos para poder observar N ∈ {1, 3, 6, 10, 20}
dentro de la misma conversación. Cada turno declara, además de la transcripción, un **fact-tracker**:
los hechos (strings cortos, literales) que los turnos posteriores deberían seguir "recordando" en el
resumen, las anclas o `project_metadata`. Son lo que `MemoryDriftMetric` busca en el snapshot de la sesión.

- `growing`: el proyecto crece. Cada turno añade un requisito coherente (OAuth, multi-tenant, audit log,
  exporte CSV...). Mide la curva de coste y si el `project_name` ("Nimbus") sobrevive al turno 20.
- `pivot`: el turno 5 cambia el stack (de React Native a Flutter). Mide si la memoria se actualiza
  limpiamente o si `mentioned_technologies` acumula ambos (`retired` marca lo que debería dejar de ser vigente).
- `contradiction`: el turno 3 fija "presupuesto 30.000 €" y el turno 8 dice "80.000 €" (y el plazo cambia
  en el 12). Mide cuál se preserva, cuál se promueve a ancla y cuál acaba en el resumen.

Los hechos se escriben tal como aparecen en la transcripción para que el match exacto (case-insensitive)
sea justo: si un hecho no aparece en el snapshot es porque el sistema lo perdió, no porque lo reformuló.
Las transcripciones usan verbos que las anclas reconocen ("necesitamos", "hay que", "en vez de"...) de
forma natural, igual que lo haría un cliente; no están escritas para engañar a la heurística.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScenarioTurn:
    turn_index: int                      # 1-based
    transcript: str
    facts: tuple[str, ...] = ()          # lo que los turnos posteriores deberían recordar
    retired: tuple[str, ...] = ()        # hechos anteriores que este turno deja sin vigencia
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    project_name: str
    description: str
    turns: tuple[ScenarioTurn, ...]

    def facts_expected_at(self, turn_index: int) -> list[tuple[int, str]]:
        """(turno que lo dijo, hecho) para todos los hechos declarados en turnos k < turn_index y no retirados antes de turn_index."""
        retired = {fact for turn in self.turns if turn.turn_index < turn_index for fact in turn.retired}
        return [(t.turn_index, fact) for t in self.turns if t.turn_index < turn_index for fact in t.facts if fact not in retired]

    def retired_before(self, turn_index: int) -> list[str]:
        return [fact for turn in self.turns if turn.turn_index < turn_index for fact in turn.retired]

    def __len__(self) -> int:
        return len(self.turns)


def _turns(spec: list[tuple]) -> tuple[ScenarioTurn, ...]:
    turns = []
    for index, item in enumerate(spec, start=1):
        transcript, facts, *rest = item
        retired = rest[0] if rest else ()
        params = rest[1] if len(rest) > 1 else {}
        turns.append(ScenarioTurn(turn_index=index, transcript=transcript, facts=tuple(facts), retired=tuple(retired), params=dict(params)))
    return tuple(turns)


GROWING = Scenario(
    key="growing",
    name="Proyecto que crece",
    project_name="Nimbus",
    description="Un SaaS de gestión de flotas al que cada turno añade un requisito coherente.",
    turns=_turns([
        (
            "Reunión inicial con una empresa de logística. El proyecto se llama Nimbus: una plataforma web para gestionar "
            "una flota de 300 vehículos con seguimiento de rutas, partes de incidencias y mantenimiento preventivo. "
            "Quieren hacerlo en React con PostgreSQL. Los conductores usarán el móvil para los partes.",
            ["Nimbus", "300 vehículos", "React", "PostgreSQL"],
            (),
            {"project_name": "Nimbus", "project_type": "web", "hourly_rate_eur": 60},
        ),
        ("Segunda reunión. Necesitamos autenticación con Google mediante OAuth para todos los usuarios, nada de contraseñas propias.", ["OAuth"]),
        ("Confirman que venderán la plataforma a otras empresas: hay que hacerla multi-tenant desde el principio, con datos aislados por empresa.", ["multi-tenant"]),
        ("Por auditoría interna necesitan un audit log de todas las acciones de los usuarios, con quién, qué y cuándo.", ["audit log"]),
        ("Añadir la exportación a CSV de rutas, incidencias y mantenimientos desde cualquier listado.", ["CSV"]),
        ("Piden notificaciones por WhatsApp al conductor cuando se le asigna una ruta o cambia el horario.", ["WhatsApp"]),
        ("Dirección quiere un panel de KPIs con kilómetros, consumo y coste por vehículo y por mes.", ["KPIs"]),
        ("Hay que cobrar a las empresas cliente una cuota mensual con Stripe según el número de vehículos.", ["Stripe"]),
        ("Los conductores a veces no tienen cobertura: la parte móvil debe funcionar en modo offline y sincronizar después.", ["offline"]),
        ("Definimos tres roles: administrador, gestor de flota y conductor, con permisos distintos en cada pantalla.", ["gestor de flota"]),
        ("Necesitamos ver los vehículos en un mapa con Google Maps en tiempo real y el histórico de rutas.", ["Google Maps"]),
        ("El contrato con el cliente principal exige un SLA del 99,9 % de disponibilidad con monitorización y alertas.", ["99,9"]),
        ("Legal pide cumplimiento RGPD: consentimiento, derecho al olvido y registro de accesos a datos personales.", ["RGPD"]),
        ("Seguridad quiere doble factor 2FA obligatorio para administradores y gestores.", ["2FA"]),
        ("Un cliente grande quiere integrar su ERP: hay que publicar una API pública documentada con claves por empresa.", ["API pública"]),
        ("Para migrar desde su sistema actual necesitan importación masiva de vehículos y conductores desde Excel.", ["Excel"]),
        ("Deciden que la parte de conductores será una app móvil en Flutter en vez de la web responsive.", ["Flutter"]),
        ("Los partes de entrega deben llevar firma digital del receptor en el móvil, con foto opcional.", ["firma digital"]),
        ("Necesitamos facturación mensual automática a cada empresa con PDF y envío por correo.", ["facturación mensual"]),
        ("Cierre: el presupuesto total aprobado es de 120.000 € y quieren la primera versión en 16 semanas.", ["120.000", "16 semanas"]),
    ]),
)

PIVOT = Scenario(
    key="pivot",
    name="Proyecto que pivota",
    project_name="Atlas",
    description="Una app móvil de reservas que cambia de stack en el turno 5 (React Native → Flutter).",
    turns=_turns([
        (
            "Primera reunión. El proyecto se llama Atlas: una app móvil para reservar pistas de pádel y tenis en clubes "
            "deportivos, con pago en la app y recordatorios. Quieren hacerla en React Native con backend en Node.js y Firebase.",
            ["Atlas", "React Native", "Node.js", "Firebase"],
            (),
            {"project_name": "Atlas", "project_type": "movil", "hourly_rate_eur": 55},
        ),
        ("Necesitan que cada club gestione sus pistas, horarios y precios desde un panel web sencillo.", ["panel web"]),
        ("Añadir lista de espera: si una pista se libera, avisar por notificación push al primero de la lista.", ["lista de espera"]),
        ("Pagos con Stripe y también con Bizum, que es lo que usan los socios mayores.", ["Bizum"]),
        (
            "Cambio importante: el equipo técnico del cliente ha decidido que en vez de React Native usaremos Flutter para la app, "
            "porque ya tienen gente con experiencia. El backend se queda como estaba.",
            ["Flutter"],
            ["React Native"],
        ),
        ("Quieren un sistema de puntos de fidelidad por reservas que se canjeen por descuentos.", ["puntos de fidelidad"]),
        ("Hay que integrar el calendario de torneos del club y bloquear las pistas reservadas para torneos.", ["torneos"]),
        ("Necesitamos reseñas y valoraciones de los clubes por parte de los jugadores.", ["valoraciones"]),
        ("Piden un chat entre jugadores para completar partidas cuando falta gente.", ["chat"]),
        ("Añadir un modo invitado para reservar sin registrarse, con pago anticipado.", ["modo invitado"]),
        ("Los clubes quieren informes mensuales de ocupación por pista y franja horaria en PDF.", ["informes mensuales"]),
        ("El plazo objetivo es de 12 semanas para la primera versión en las dos tiendas.", ["12 semanas"]),
        ("Quieren soporte para clases particulares con monitor, con su propia agenda.", ["clases particulares"]),
        ("Necesitamos que la app esté en castellano, catalán e inglés desde el lanzamiento.", ["catalán"]),
        ("Hay que mandar un recordatorio por SMS dos horas antes de cada reserva.", ["SMS"]),
        ("Añadir reservas recurrentes semanales para grupos fijos.", ["reservas recurrentes"]),
        ("Los clubes piden control de acceso con QR en la puerta de la pista.", ["QR"]),
        ("Necesitamos cancelación con política de devolución configurable por club.", ["devolución"]),
        ("Piden una web pública con el buscador de clubes para captar jugadores.", ["buscador de clubes"]),
        ("Cierre: presupuesto aprobado de 45.000 € para la primera fase.", ["45.000"]),
    ]),
)

CONTRADICTION = Scenario(
    key="contradiction",
    name="Proyecto que se contradice",
    project_name="Faro",
    description="Una plataforma de e-learning cuyo presupuesto cambia de 30.000 € (turno 3) a 80.000 € (turno 8) y el plazo en el turno 12.",
    turns=_turns([
        (
            "Reunión inicial con una academia. El proyecto se llama Faro: una plataforma web de cursos online con vídeo, "
            "ejercicios autocorregibles y certificados al terminar. Lo quieren en Django con PostgreSQL.",
            ["Faro", "Django", "certificados"],
            (),
            {"project_name": "Faro", "project_type": "web", "hourly_rate_eur": 50},
        ),
        ("Quieren lanzar la primera versión en 12 semanas, coincidiendo con el inicio del curso.", ["12 semanas"]),
        ("Importante: el presupuesto es cerrado, 30.000 € como máximo para toda la primera fase.", ["30.000"]),
        ("Necesitamos pagos con Stripe y suscripción mensual además de la compra de cursos sueltos.", ["suscripción mensual"]),
        ("Hay que añadir foros por curso y mensajería entre alumno y profesor.", ["foros"]),
        ("Los profesores necesitan un editor de cursos con vídeos subidos a Vimeo y cuestionarios.", ["Vimeo"]),
        ("Piden un panel de progreso del alumno con porcentaje completado y nota media.", ["progreso"]),
        (
            "Novedad: han conseguido financiación y el presupuesto sube a 80.000 €; quieren meter más alcance en la primera fase.",
            ["80.000"],
            ["30.000"],
        ),
        ("Añadir clases en directo con Zoom integradas en el calendario del curso.", ["Zoom"]),
        ("Necesitamos una app móvil para consumir los cursos, en Flutter.", ["Flutter"]),
        ("Hay que generar los certificados en PDF con código de verificación público.", ["código de verificación"]),
        (
            "Con el nuevo alcance el plazo cambia: la entrega será en 20 semanas en lugar de las 12 iniciales.",
            ["20 semanas"],
            ["12 semanas"],
        ),
        ("Quieren gamificación: insignias y ranking semanal de alumnos.", ["insignias"]),
        ("Necesitamos integración con Mailchimp para campañas a los alumnos.", ["Mailchimp"]),
        ("Añadir un marketplace donde profesores externos vendan sus cursos con comisión.", ["comisión"]),
        ("Hay que cumplir accesibilidad WCAG nivel AA en toda la plataforma.", ["WCAG"]),
        ("Piden subtítulos automáticos en los vídeos en castellano e inglés.", ["subtítulos"]),
        ("Necesitamos informes para la dirección con ingresos por curso y por profesor.", ["ingresos por curso"]),
        ("Añadir cupones de descuento y códigos promocionales por campaña.", ["cupones"]),
        ("Cierre de la reunión: confirman el presupuesto de 80.000 € y las 20 semanas, y que la app móvil entra en la primera fase.", []),
    ]),
)

SCENARIOS: dict[str, Scenario] = {s.key: s for s in (GROWING, PIVOT, CONTRADICTION)}
CHECKPOINTS: tuple[int, ...] = (1, 3, 6, 10, 20)  # los N del enunciado


def get_scenarios(keys: list[str] | None = None) -> list[Scenario]:
    keys = keys or list(SCENARIOS)
    unknown = [k for k in keys if k not in SCENARIOS]
    if unknown:
        raise KeyError(f"Escenarios desconocidos: {', '.join(unknown)} (disponibles: {', '.join(SCENARIOS)})")
    return [SCENARIOS[k] for k in keys]
