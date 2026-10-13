# Stress del CAG — baseline cuantitativo (sesión 06)

_Generado el 2026-10-13 01:46 UTC por `evals/stress/report.py` a partir de `results.csv` (900 filas, 718 turnos con llamada real al LLM)._


## Configuración de la tirada

| Parámetro | Valor |
|---|---|
| Proveedor / modelo | openai / gpt-4o-mini |
| Plantilla | estimacion/v4 |
| Ventana (SESSION_MAX_TURNS) | 6 |
| Tope de adjunto (ATTACHMENT_MAX_CHARS) | 60.000 caracteres |
| Summarizer | modo `llm`, activo desde el turno 7 |
| Escenarios | `contradiction` (Proyecto que se contradice), `growing` (Proyecto que crece), `pivot` (Proyecto que pivota) |
| Tamaños de adjunto | 0 KB, 5 KB, 20 KB, 50 KB, 100 KB |
| Repeticiones | 3 |
| Presupuestos | latencia ≤ 4000 ms · coste ≤ 0.01 USD por turno |
| Inicio / fin | 2026-10-12T23:52:30.976456+00:00 / 2026-10-13T01:38:17.855030+00:00 |

Las repeticiones miden las **ráfagas**: la segunda y tercera pasada de cada conversación envían exactamente los mismos prompts, así que son el test de la caché exacta (y, si cambia alguna palabra, de la semántica). Por eso la tabla resumen separa la latencia de todos los turnos de la de los turnos **sin caché**, que es la curva real del LLM.

## 1. Tabla resumen (escenario × tamaño de adjunto)

| Escenario | Adjunto | Turnos | P50 lat (ms) | P95 lat (ms) | P50 sin caché | P95 sin caché | Coste total (USD) | Hit exacto | Hit semántico | Recall medio | Latency budget OK | Cost budget OK |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| contradiction | 0 KB | 60 | 7.330 | 10.944 | 8.507 | 10.952 | 0.0914 | 22 % | 0 % | 99 % | 28 % | 100 % |
| contradiction | 5 KB | 60 | 8.273 | 10.754 | 8.616 | 11.162 | 0.0973 | 20 % | 0 % | 94 % | 22 % | 100 % |
| contradiction | 20 KB | 60 | 8.693 | 11.286 | 9.286 | 11.413 | 0.1011 | 20 % | 0 % | 99 % | 23 % | 100 % |
| contradiction | 50 KB | 60 | 8.441 | 10.746 | 8.815 | 10.954 | 0.1043 | 20 % | 0 % | 99 % | 20 % | 100 % |
| contradiction | 100 KB | 60 | 9.173 | 11.865 | 10.005 | 11.956 | 0.1114 | 20 % | 0 % | 94 % | 20 % | 100 % |
| growing | 0 KB | 60 | 8.485 | 11.170 | 9.083 | 11.923 | 0.1007 | 20 % | 0 % | 100 % | 23 % | 100 % |
| growing | 5 KB | 60 | 9.750 | 12.463 | 10.388 | 12.575 | 0.1047 | 20 % | 0 % | 100 % | 23 % | 100 % |
| growing | 20 KB | 60 | 10.305 | 13.111 | 10.536 | 13.197 | 0.1141 | 20 % | 0 % | 100 % | 20 % | 100 % |
| growing | 50 KB | 60 | 8.970 | 11.799 | 9.689 | 11.800 | 0.1141 | 20 % | 0 % | 100 % | 20 % | 100 % |
| growing | 100 KB | 60 | 9.668 | 13.608 | 10.559 | 14.233 | 0.1223 | 20 % | 0 % | 100 % | 22 % | 100 % |
| pivot | 0 KB | 60 | 7.807 | 10.887 | 8.396 | 11.017 | 0.0921 | 22 % | 0 % | 100 % | 27 % | 100 % |
| pivot | 5 KB | 60 | 6.684 | 7.478 | 6.813 | 7.614 | 0.0804 | 20 % | 0 % | 100 % | 30 % | 100 % |
| pivot | 20 KB | 60 | 7.331 | 8.940 | 7.506 | 8.997 | 0.0894 | 20 % | 0 % | 100 % | 25 % | 100 % |
| pivot | 50 KB | 60 | 9.003 | 10.808 | 9.393 | 10.818 | 0.1125 | 20 % | 0 % | 100 % | 20 % | 100 % |
| pivot | 100 KB | 60 | 7.217 | 10.358 | 7.407 | 11.182 | 0.0955 | 20 % | 0 % | 100 % | 27 % | 100 % |

## 2. Curva 1 — latencia vs tokens de entrada (solo turnos sin caché)

| tokens_in | Turnos | P50 latencia (ms) | P95 latencia (ms) | tokens_out medio | Coste medio (USD) | Con summarizer |
|---|---|---|---|---|---|---|
| 2.000–3.999 | 3 | 4.062 | 4.078 | 401 | 0.000750 | 0 % |
| 4.000–5.999 | 15 | 3.863 | 4.555 | 457 | 0.001028 | 0 % |
| 6.000–7.999 | 14 | 4.096 | 5.163 | 503 | 0.001350 | 0 % |
| 8.000–9.999 | 375 | 8.054 | 10.003 | 923 | 0.001920 | 97 % |
| 10.000–11.999 | 252 | 10.559 | 12.542 | 1.277 | 0.002384 | 98 % |
| 12.000–13.999 | 23 | 12.688 | 14.315 | 1.469 | 0.002747 | 87 % |
| 14.000–15.999 | 3 | 4.040 | 4.273 | 470 | 0.002559 | 0 % |
| 16.000–17.999 | 12 | 4.615 | 6.811 | 496 | 0.002859 | 0 % |
| 18.000–19.999 | 13 | 4.840 | 5.785 | 510 | 0.003158 | 0 % |
| 20.000–21.999 | 8 | 4.842 | 6.358 | 546 | 0.003469 | 0 % |

## 3. Curva 2 — coste acumulado vs turno (sin adjunto, primera repetición)

| Turno | contradiction: acumulado (turno · tokens_in) | growing: acumulado (turno · tokens_in) | pivot: acumulado (turno · tokens_in) |
|---|---|---|---|
| 1 | 0.00075 (0.00075 · 3.377 tok) | 0.00075 (0.00075 · 3.399 tok) | 0.00075 (0.00075 · 3.404 tok) |
| 2 | 0.00163 (0.00088 · 4.208 tok) | 0.00165 (0.00090 · 4.242 tok) | 0.00163 (0.00089 · 4.215 tok) |
| 3 | 0.00264 (0.00101 · 4.932 tok) | 0.00269 (0.00104 · 5.020 tok) | 0.00266 (0.00103 · 4.976 tok) |
| 4 | 0.00378 (0.00114 · 5.701 tok) | 0.00386 (0.00118 · 5.823 tok) | 0.00381 (0.00115 · 5.755 tok) |
| 5 | 0.00507 (0.00129 · 6.487 tok) | 0.00519 (0.00132 · 6.647 tok) | 0.00509 (0.00128 · 6.589 tok) |
| 6 | 0.00651 (0.00144 · 7.329 tok) | 0.00667 (0.00148 · 7.535 tok) | 0.00652 (0.00142 · 7.399 tok) |
| 7 | 0.00816 (0.00165 · 8.109 tok) | 0.00837 (0.00171 · 8.377 tok) | 0.00814 (0.00162 · 8.155 tok) |
| 8 | 0.00988 (0.00172 · 8.473 tok) | 0.01019 (0.00182 · 8.741 tok) | 0.00986 (0.00172 · 8.457 tok) |
| 9 | 0.01168 (0.00179 · 8.731 tok) | 0.01211 (0.00192 · 9.088 tok) | 0.01166 (0.00180 · 8.717 tok) |
| 10 | 0.01356 (0.00188 · 9.024 tok) | 0.01413 (0.00202 · 9.375 tok) | 0.01355 (0.00189 · 8.970 tok) |
| 11 | 0.01550 (0.00194 · 9.270 tok) | 0.01623 (0.00211 · 9.679 tok) | 0.01549 (0.00194 · 9.210 tok) |
| 12 | 0.01751 (0.00201 · 9.513 tok) | 0.01842 (0.00218 · 9.975 tok) | 0.01749 (0.00200 · 9.388 tok) |
| 13 | 0.01959 (0.00208 · 9.685 tok) | 0.02068 (0.00227 · 10.212 tok) | 0.01953 (0.00203 · 9.573 tok) |
| 14 | 0.02167 (0.00208 · 9.849 tok) | 0.02302 (0.00234 · 10.474 tok) | 0.02162 (0.00210 · 9.753 tok) |
| 15 | 0.02379 (0.00212 · 9.922 tok) | 0.02541 (0.00239 · 10.751 tok) | 0.02378 (0.00216 · 9.961 tok) |
| 16 | 0.02596 (0.00217 · 10.084 tok) | 0.02787 (0.00246 · 11.031 tok) | 0.02600 (0.00222 · 10.152 tok) |
| 17 | 0.02819 (0.00223 · 10.274 tok) | 0.03038 (0.00250 · 11.289 tok) | 0.02827 (0.00227 · 10.339 tok) |
| 18 | 0.03047 (0.00228 · 10.471 tok) | 0.03293 (0.00255 · 11.489 tok) | 0.03058 (0.00231 · 10.502 tok) |
| 19 | 0.03282 (0.00235 · 10.651 tok) | 0.03553 (0.00260 · 11.667 tok) | 0.03293 (0.00235 · 10.691 tok) |
| 20 | 0.03521 (0.00239 · 10.906 tok) | 0.03816 (0.00263 · 11.850 tok) | 0.03533 (0.00241 · 10.860 tok) |

## 4. Curva 3 — MemoryDriftMetric vs N (media sobre tamaños y repeticiones)

Recall = fracción de los hechos declarados en turnos k < N que aparecen literalmente (case-insensitive) en el resumen, las anclas o `project_metadata` del turno N. No mira el historial: mide lo que sobrevive a la ventana.

| N (turnos) | contradiction: recall (hechos esperados · nombre en título) | growing: recall (hechos esperados · nombre en título) | pivot: recall (hechos esperados · nombre en título) |
|---|---|---|---|
| 1 | – (0 hechos · título OK 100 %) | – (0 hechos · título OK 100 %) | – (0 hechos · título OK 100 %) |
| 3 | 90 % (4 hechos · título OK 100 %) | 100 % (5 hechos · título OK 100 %) | 100 % (5 hechos · título OK 100 %) |
| 6 | 94 % (7 hechos · título OK 100 %) | 100 % (8 hechos · título OK 100 %) | 100 % (7 hechos · título OK 100 %) |
| 10 | 100 % (10 hechos · título OK 100 %) | 100 % (12 hechos · título OK 100 %) | 100 % (11 hechos · título OK 100 %) |
| 20 | 100 % (19 hechos · título OK 100 %) | 100 % (22 hechos · título OK 100 %) | 100 % (21 hechos · título OK 100 %) |

### Hechos retirados y dónde viven los recordados

| Escenario | N final | Conversaciones con hechos retirados aún presentes | Hechos retirados presentes | Dónde se encuentran los hechos recordados (conteo) |
|---|---|---|---|---|
| pivot | 20 | 100 % | React Native | summary+anchors+metadata: 116, anchors: 87, anchors+metadata: 82, summary+anchors: 30 |
| contradiction | 20 | 100 % | 12 semanas; 30.000, 12 semanas | anchors+metadata: 139, summary+anchors+metadata: 111, anchors: 22, summary+anchors: 13 |

## 5. Adjuntos grandes (turno 1 de cada conversación)

| Adjunto | Chars extraídos | P50 tokens_in | P50 latencia (ms) | P95 latencia (ms) | Coste medio turno 1 (USD) | Tier extended | Recall marcadores |
|---|---|---|---|---|---|---|---|
| 0 KB | 0 | 3.399 | 4.062 | 4.078 | 0.000750 | 0.00 | – |
| 5 KB | 5.085 | 4.624 | 3.902 | 4.008 | 0.000971 | 0.00 | 93 % |
| 20 KB | 20.184 | 8.186 | 3.923 | 4.183 | 0.001497 | 1.00 | 80 % |
| 50 KB | 49.979 | 15.180 | 4.040 | 4.273 | 0.002559 | 1.00 | 87 % |
| 100 KB | 60.047 | 17.538 | 3.917 | 4.357 | 0.002900 | 1.00 | 40 % |

## 6. Lectura

**Dónde empieza a romperse mi CAG.** Con la ventana de 6 turnos, el primer síntoma no es un error sino un coste: el summarizer entra en el turno 7 (el primero que expulsa un turno de la ventana) y a partir de ahí cada turno paga dos llamadas. El coste del turno 20 multiplica por 3.30 el del turno 1 (tokens de entrada ×3.31) y la latencia por 2.79; el 94 % de los turnos sin caché supera el presupuesto de latencia y el 0 % el de coste. En las ráfagas (repeticiones 2 y 3: los mismos prompts otra vez) la caché exacta absorbe el 100 % de los turnos 1-6 con 0 tokens, 0 USD y latencia de milisegundos, pero solo el 0 % de los turnos 7-20: en cuanto el summarizer pliega el primer turno, el resumen que escribe el LLM cambia de una pasada a otra (temperatura 0,2) y el system prompt deja de ser idéntico, así que ni la caché exacta ni la semántica (que exige el mismo system y el mismo historial) lo atrapan; la semántica no llegó a acertar ninguna vez. La latencia no la manda la entrada: los turnos con 18.476 tokens de entrada de media y una sola llamada (turno 1 con adjunto grande) responden en 4.615 ms P50, mientras que los turnos con summarizer, con menos entrada (9.872 tokens), tardan 9.217 ms: encadenan dos llamadas (3.154 ms del summarizer de media) y generan más salida (1.100 tokens frente a 510) porque la estimación crece con el proyecto. Lo que no se rompe en 20 turnos es la memoria literal: ningún escenario cae bajo el 60 % y el recall del fact-tracker en el turno 20 es 100 % en `contradiction`, 100 % en `growing`, 100 % en `pivot`. Anclas, resumen y `project_metadata` se solapan (la mayoría de los hechos vive en dos o tres sitios a la vez), así que perder uno de los tres no deja hueco; lo que sí falla es retirar lo que ya no vale (ver más abajo). El nombre del proyecto sigue en el título de la estimación en el 100 % de las conversaciones en el turno 20.

**Qué dimensión domina y cuándo saltar a RAG.** Entre coste (+230 %), latencia (+179 %) y pérdida de memoria (recall intacto), la dimensión que más se degrada entre el turno 1 y el 20 es **coste**. Coste y latencia crecen de forma acotada porque la ventana y el resumen ponen un techo al contexto; lo que no está acotado es lo que se pierde para mantener ese techo: el resumen está limitado a 1.500 caracteres y las anclas a 24, así que cada turno nuevo desplaza hechos antiguos. Los hechos que sobreviven lo hacen por la memoria estructurada (`project_metadata`) o por una ancla literal; los que se reformulan en el resumen cuentan como perdidos. En `pivot`, el 100 % de las conversaciones llega al turno 20 con el stack retirado (React Native) todavía presente: `project_metadata` acumula tecnologías pero no sabe retirar ninguna. Con adjuntos, la latencia P50 del turno 1 pasa de 4.062 ms sin adjunto a 3.917 ms con 100 KB, y el recall de los marcadores del PDF baja hasta 40 % con 100 KB (con los que caben enteros, 5 KB, 20 KB, 50 KB, el modelo ya deja fuera marcadores: recall medio 87 %); el tope `ATTACHMENT_MAX_CHARS=60.000` trunca el de 100 KB, así que los marcadores del 75 % y del final del documento nunca llegan al modelo. El caso límite que justifica RAG es el que combina las dos curvas: un proyecto que supera la ventana (N > 6) y además arrastra documentación que no cabe (> 60.000 caracteres). Ahí el CAG solo puede elegir entre pagar el contexto completo en cada turno o resumirlo y perder recall; recuperar por relevancia solo lo que cada turno necesita es la única opción que no sacrifica ninguna de las tres curvas.

## 7. Notas de método

- `turn_observed` se lee del campo `turn_observed` de la respuesta de `POST /sessions/{id}/estimate` y se contrasta con `GET /sessions/{id}.last_turn_observed`: es el mismo diccionario que se emite al log con structlog. No hay que parsear stdout ni reconciliar timestamps, y el runner funciona igual contra Docker, uvicorn o in-process.
- `LatencyBudgetMetric`, `CostBudgetMetric` y `MemoryDriftMetric` viven en `evals/stress/metrics.py` porque evalúan el shape de `turn_observed` y del snapshot de sesión, no una estimación suelta como las tres de `evals/metrics.py`; comparten `MetricResult` y `run_all_metrics`.
- `latency_ms` y `cost_usd` del turno incluyen la llamada del summarizer cuando la hay (`summarizer_*` las desglosa). Un acierto de caché registra 0 tokens y 0 USD.
- Los PDFs son sintéticos y deterministas (`evals/stress/fixtures/build_pdfs.py`); llevan cinco marcadores repartidos por el documento para medir el recall del adjunto en la estimación. No se versionan: el runner los regenera.
- La caché de respuestas vive en memoria del proceso del servicio, así que el servicio se arranca justo antes de la tirada (caché vacía). Si se reutiliza un servicio que ya respondió a esos mismos prompts (por ejemplo tras una pasada de humo), la repetición 1 acierta en caché desde el primer turno y deja de medir al LLM.
- No se ha tocado ninguna constante entre medidas (ventana de 6, resumen de 1.500 caracteres, 24 anclas, tope de adjunto de 60.000).
