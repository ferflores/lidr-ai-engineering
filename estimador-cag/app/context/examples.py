"""Contexto estático para la arquitectura CAG.

Estas estimaciones "históricas" son el conocimiento del sistema: se inyectan
íntegras en el system prompt en cada llamada al LLM, a modo de few-shot
examples. No hay base de datos ni retrieval. Si hay que ampliar el conocimiento
del estimador, se añade aquí.

Cada ejemplo tiene:
- `meeting_summary`: resumen de la reunión original (qué pedía el cliente).
- `estimation`: la estimación que se entregó (desglose de tareas, horas, coste,
  equipo, duración, supuestos y preguntas pendientes).

Los datos son ficticios.
"""

from textwrap import dedent

ESTIMATION_EXAMPLES: list[dict[str, str]] = [
    {
        "meeting_summary": (
            "El cliente, una distribuidora de material eléctrico con tres almacenes, "
            "necesita una plataforma web de gestión de inventario. Quiere dar de alta "
            "productos y proveedores, registrar entradas y salidas de stock por almacén, "
            "recibir alertas cuando un producto baje del mínimo y ver un dashboard con "
            "métricas de rotación. Habrá tres tipos de usuario (administrador, almacenero "
            "y consulta). No tienen diseño previo y quieren empezar cuanto antes."
        ),
        "estimation": dedent("""\
            ## Estimación: Plataforma de Gestión de Inventario

            ### Resumen del alcance
            Aplicación web para gestionar productos, proveedores y movimientos de stock
            en tres almacenes, con alertas de stock mínimo, roles de usuario y un
            dashboard de métricas de rotación.

            ### Desglose de tareas:
            1. Análisis funcional y modelado de datos: 15 horas
            2. Diseño UI/UX (wireframes + diseño visual): 40 horas
            3. Backend API (CRUD de productos, proveedores y almacenes): 45 horas
            4. Movimientos de stock y alertas de mínimos: 25 horas
            5. Autenticación y roles (administrador, almacenero, consulta): 20 horas
            6. Frontend web (listados, formularios, filtros): 50 horas
            7. Dashboard con métricas de rotación: 30 horas
            8. Testing y QA: 25 horas
            9. Despliegue y documentación: 10 horas

            **Total estimado: 260 horas**
            **Coste estimado: 13.000 €** (a 50 €/hora)
            **Equipo recomendado: 2 desarrolladores full-stack + 1 diseñador UX (part-time)**
            **Duración estimada: 8-10 semanas**

            ### Supuestos y riesgos
            - No hay integración con el ERP actual; la carga inicial de productos se hace por importación CSV.
            - El dashboard se limita a métricas predefinidas; no incluye un generador de informes a medida.
            - Riesgo: los flujos de traspaso de stock entre almacenes pueden requerir varias iteraciones con el cliente.

            ### Preguntas pendientes para el cliente
            - ¿Se necesita lectura de códigos de barras desde el móvil en el almacén?
            - ¿Los proveedores deben tener acceso a la plataforma?
        """),
    },
    {
        "meeting_summary": (
            "Una clínica dental quiere una app móvil (iOS y Android) para que sus "
            "pacientes reserven, modifiquen y cancelen citas. Al reservar, el paciente "
            "paga una pequeña señal con tarjeta. Quieren recordatorios por notificación "
            "push y email, y un panel web para que recepción gestione la agenda de los "
            "cuatro dentistas. Tienen logo y colores corporativos pero no diseño de app."
        ),
        "estimation": dedent("""\
            ## Estimación: App Móvil de Reserva de Citas para Clínica Dental

            ### Resumen del alcance
            App multiplataforma para que los pacientes gestionen sus citas con pago de
            señal, recordatorios automáticos y un panel web de recepción para
            administrar la agenda de los profesionales.

            ### Desglose de tareas:
            1. Análisis funcional y definición de flujos de reserva: 20 horas
            2. Diseño UI/UX de la app y del panel web: 45 horas
            3. Backend API (pacientes, profesionales, agenda, disponibilidad, citas): 70 horas
            4. App móvil multiplataforma (Flutter): 110 horas
            5. Integración de pagos (Stripe) para el cobro de la señal: 25 horas
            6. Notificaciones push y recordatorios por email: 20 horas
            7. Panel web de recepción (gestión de agenda y citas): 40 horas
            8. Testing y QA (incluye pruebas en dispositivos reales): 40 horas
            9. Publicación en App Store / Google Play y despliegue: 15 horas

            **Total estimado: 385 horas**
            **Coste estimado: 19.250 €** (a 50 €/hora)
            **Equipo recomendado: 1 desarrollador mobile + 1 desarrollador backend + 1 diseñador UX (part-time)**
            **Duración estimada: 12-14 semanas**

            ### Supuestos y riesgos
            - Una sola clínica y una agenda por profesional; no se contempla multi-sede.
            - La clínica dispone de cuentas de desarrollador en Apple y Google.
            - Riesgo: la revisión de Apple puede retrasar la publicación entre 1 y 2 semanas.

            ### Preguntas pendientes para el cliente
            - ¿Hay que sincronizar la agenda con el software de gestión actual de la clínica?
            - ¿La app debe mostrar historial clínico o solo citas?
        """),
    },
    {
        "meeting_summary": (
            "Una empresa de cosmética vende online con Shopify y gestiona el negocio con "
            "el ERP SAP Business One. Ahora sincronizan catálogo, stock y pedidos a mano. "
            "Quieren un conector que mantenga el catálogo y el stock actualizados en la "
            "tienda desde el ERP, que envíe los pedidos y clientes nuevos al ERP, y un "
            "panel de informes de ventas por canal, producto y periodo. Ya tienen un "
            "entorno de pruebas del ERP."
        ),
        "estimation": dedent("""\
            ## Estimación: Integración de Tienda Online con ERP e Informes de Ventas

            ### Resumen del alcance
            Conector bidireccional entre Shopify y SAP Business One para sincronizar
            catálogo, stock, pedidos y clientes, más un panel de informes de ventas.

            ### Desglose de tareas:
            1. Análisis de APIs y mapeo de datos entre Shopify y el ERP: 20 horas
            2. Sincronización de catálogo y stock (ERP → tienda): 40 horas
            3. Sincronización de pedidos y clientes (tienda → ERP): 35 horas
            4. Gestión de errores, reintentos y monitorización: 20 horas
            5. Panel de informes de ventas (por canal, producto y periodo): 35 horas
            6. Testing y QA con datos reales en el entorno de pruebas: 25 horas
            7. Documentación técnica y despliegue: 10 horas

            **Total estimado: 185 horas**
            **Coste estimado: 9.250 €** (a 50 €/hora)
            **Equipo recomendado: 1 desarrollador backend senior + 1 desarrollador full-stack**
            **Duración estimada: 6-7 semanas**

            ### Supuestos y riesgos
            - La sincronización es periódica (cada 15 minutos), no en tiempo real.
            - El entorno de pruebas del ERP contiene datos representativos.
            - Riesgo: las licencias de la API del ERP pueden limitar el número de llamadas diarias.

            ### Preguntas pendientes para el cliente
            - ¿Deben sincronizarse también las devoluciones y los abonos?
            - ¿Quién mantiene el conector tras la entrega?
        """),
    },
]


def render_examples(examples: list[dict[str, str]] | None = None) -> str:
    """Convierte los ejemplos en el bloque de texto que se inyecta en el system prompt."""
    examples = ESTIMATION_EXAMPLES if examples is None else examples
    bloques = []
    for indice, ejemplo in enumerate(examples, start=1):
        bloques.append(
            f"### Ejemplo {indice}\n\n"
            f"**Resumen de la reunión:** {ejemplo['meeting_summary'].strip()}\n\n"
            f"**Estimación entregada:**\n\n{ejemplo['estimation'].strip()}\n"
        )
    return "\n".join(bloques)
