# Comparativa: transcripción pobre vs. detallada

Mismo proyecto (tienda online para una panadería artesanal), dos transcripciones de calidad muy distinta,
enviadas al endpoint `POST /api/v1/estimate` el 2026-09-16 con `gpt-4o-mini`, temperatura 0.2 y los tres
ejemplos de contexto de `app/context/examples.py`. Sirve como referencia para futuros casos y para medir
si los cambios en el prompt o en los ejemplos mejoran el resultado.

| | Pobre | Detallada |
|---|---|---|
| Caracteres de entrada | 173 | 2.672 |
| Tokens (entrada / salida) | 1929 / 341 | 2581 / 431 |
| Tareas | 7 genéricas | 10 específicas |
| Total | 230 h · 11.500 € · 6-8 semanas | 410 h · 20.500 € · 10-12 semanas |
| Preguntas pendientes | 2 básicas | 3 concretas |

## Qué se observa

- **Pobre:** el modelo inventa un alcance "estándar" de tienda online y lo estima. Las cifras parecen
  razonables, pero describen un proyecto imaginado, no el del cliente. Los supuestos rellenan lo que no se
  dijo y las preguntas son las que debería haber respondido la reunión (métodos de pago, registro).
- **Detallada:** el desglose refleja lo hablado (pedidos recurrentes, panel del obrador, notificaciones,
  hosting) y las preguntas son concretas. Aun así se deja cosas que sí estaban en la transcripción:
  - No avisa de que 410 h / 20.500 € y 10-12 semanas **superan el presupuesto de 12.000 € y el plazo de
    8 semanas** que dio la cliente, aunque el system prompt pide valorar la viabilidad del plazo.
  - Omite como tareas el reparto por radio con tarifa, el bilingüismo catalán/castellano, Bizum, la
    antelación de 48 h para encargos y la exportación mensual a Excel (esta última la convierte en pregunta).
  - El stock diario limitado por producto, que el obrador señaló como lo más importante, solo aparece de
    pasada dentro del backend.
- Conclusión: la calidad de la transcripción marca el techo de la estimación, y el prompt actual todavía
  no obliga al modelo a contrastar la estimación con las restricciones del cliente. Buen punto de partida
  para iterar en la sesión en vivo.

---

## Transcripción pobre

Entrada: `transcripciones/panaderia-descripcion-pobre.json` · modelo `gpt-4o-mini` (openai) · tokens: 1929 entrada / 341 salida

<details>
<summary>Transcripción enviada</summary>

```text
El cliente tiene una panadería y quiere vender por internet. Necesita una web donde la gente pueda hacer pedidos y pagar. Quiere que esté lista pronto y que no sea muy cara.
```

</details>

### Estimación devuelta

## Estimación: Plataforma de Venta Online para Panadería

### Resumen del alcance
Desarrollo de una página web para que los clientes realicen pedidos y pagos online para una panadería. El cliente busca una solución económica y con un plazo de entrega corto.

### Desglose de tareas:
1. Análisis funcional y definición de requisitos: 10 horas
2. Diseño UI/UX de la web: 30 horas
3. Desarrollo del backend (gestión de productos, pedidos y pagos): 40 horas
4. Implementación de la pasarela de pagos: 20 horas
5. Desarrollo del frontend (páginas de productos, carrito y checkout): 40 horas
6. Testing y QA: 20 horas
7. Despliegue y documentación: 10 horas

**Total estimado: 230 horas**
**Coste estimado: 11.500 €** (a 50 €/hora)
**Equipo recomendado: 1 desarrollador full-stack + 1 diseñador UX (part-time)**
**Duración estimada: 6-8 semanas**

### Supuestos y riesgos
- No se contempla la integración con un sistema de gestión de inventario; se asume que la gestión de productos será manual.
- El diseño será sencillo y no incluirá funcionalidades avanzadas como personalización de productos.
- Riesgo: el plazo de entrega puede verse afectado si se requieren cambios significativos en el diseño o funcionalidades.

### Preguntas pendientes para el cliente
- ¿Qué métodos de pago desea ofrecer (tarjeta, PayPal, etc.)?
- ¿Necesita funcionalidades adicionales como registro de usuarios o gestión de promociones?

---

## Transcripción detallada

Entrada: `transcripciones/panaderia-descripcion-detallada.json` · modelo `gpt-4o-mini` (openai) · tokens: 2581 entrada / 431 salida

<details>
<summary>Transcripción enviada</summary>

```text
Reunión de toma de requisitos — Panadería artesanal "Forn de la Plaça" — 15/09/2026

Asistentes: Núria (propietaria), Jordi (encargado del obrador), Ana (consultora), Pablo (tech lead).

Núria: Somos una panadería artesanal con una tienda en Girona. Cada vez más clientes nos piden por WhatsApp y por Instagram, y lo apuntamos a mano en una libreta. Queremos una tienda online propia para que la gente pida y recoja en tienda o se lo llevemos a casa.

Ana: ¿Qué vendéis y cuántos productos hay?

Núria: Unos 60 productos: panes, bollería, pasteles por encargo y algunos productos de temporada. Algunos tienen variantes, por ejemplo el pan de payés en 500 g y 1 kg, y las tartas por número de raciones.

Jordi: Lo importante es el stock diario. Cada día hacemos una cantidad limitada de cada cosa. Cuando se acaban las 30 barras de espelta, no se pueden pedir más para ese día. Y los pasteles por encargo necesitan 48 horas de antelación.

Ana: ¿Cómo sería la entrega?

Núria: Recogida en tienda en franjas de media hora, de 8:00 a 14:00 y de 17:00 a 20:00. Y reparto a domicilio en un radio de 5 km con una tarifa fija de 3 €, gratis a partir de 25 €. El reparto lo hace nuestro chico con la furgoneta, solo por las mañanas.

Pablo: ¿Pagos?

Núria: Tarjeta y Bizum. Usamos Stripe en la web antigua, que ya no funciona. También queremos que los clientes puedan crear cuenta y repetir su pedido habitual; hay mucha gente que pide lo mismo cada semana. Nos gustaría que pudieran programar un pedido recurrente semanal.

Jordi: Para nosotros lo clave es un panel en el obrador: cada mañana a las 6 quiero ver la lista de producción del día ordenada por producto, con las cantidades sumadas de todos los pedidos, y poder imprimirla. Y marcar los pedidos como preparados y entregados.

Ana: ¿Notificaciones?

Núria: Email de confirmación y un aviso cuando el pedido esté listo. Con email basta para empezar; WhatsApp sería un plus.

Pablo: ¿Tenéis diseño? ¿Integraciones con el TPV o con contabilidad?

Núria: Tenemos logo, fotos profesionales de los productos y una guía de marca. No tenemos diseño de la web. El TPV es un Square básico, no hace falta integrarlo por ahora. La contabilidad la lleva una gestoría; solo necesitamos exportar las ventas del mes a Excel.

Ana: ¿Idiomas, plazos y presupuesto?

Núria: Catalán y castellano. Nos gustaría tenerlo antes de la campaña de Navidad, en unas 8 semanas. El presupuesto máximo son 12.000 €. No tenemos hosting ni dominio nuevo; hay que contratarlo todo.

Pablo: ¿Volumen esperado?

Núria: Ahora unos 40 pedidos al día por WhatsApp; en Navidad puede subir a 150. Nada masivo.

Ana: Perfecto, con esto preparamos la estimación.
```

</details>

### Estimación devuelta

## Estimación: Tienda Online para Panadería "Forn de la Plaça"

### Resumen del alcance
Desarrollo de una tienda online para la panadería "Forn de la Plaça" que permita a los clientes realizar pedidos para recoger en tienda o recibir a domicilio. Incluye gestión de stock, programación de pedidos recurrentes, panel de producción y notificaciones por email.

### Desglose de tareas:
1. Análisis funcional y modelado de datos: 20 horas
2. Diseño UI/UX (wireframes + diseño visual): 50 horas
3. Backend API (gestión de productos, stock, pedidos, usuarios): 70 horas
4. Implementación de la tienda online (frontend): 80 horas
5. Integración de pagos (Stripe): 25 horas
6. Funcionalidad de pedidos recurrentes: 30 horas
7. Panel de producción para el obrador: 40 horas
8. Notificaciones por email (confirmación y aviso de pedido listo): 20 horas
9. Testing y QA: 30 horas
10. Despliegue y configuración de hosting y dominio: 15 horas

**Total estimado: 410 horas**
**Coste estimado: 20.500 €** (a 50 €/hora)
**Equipo recomendado: 1 desarrollador backend + 1 desarrollador frontend + 1 diseñador UX (part-time)**
**Duración estimada: 10-12 semanas**

### Supuestos y riesgos
- La tienda online no requiere integración con el TPV ni con la contabilidad en esta fase.
- Se asume que el hosting y dominio se contratarán sin complicaciones adicionales.
- Riesgo: el volumen de pedidos puede requerir ajustes en la gestión de stock y en la infraestructura de la tienda.

### Preguntas pendientes para el cliente
- ¿Se necesita algún tipo de gestión de usuarios más allá de la creación de cuentas?
- ¿Qué información específica debe incluir la lista de producción que se imprimirá?
- ¿Hay algún requisito adicional para la exportación de ventas a Excel?
