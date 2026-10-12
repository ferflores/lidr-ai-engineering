# Stress del CAG — baseline cuantitativo (sesión 06)

_Generado el 2026-10-12 17:20 UTC por `evals/stress/report.py` a partir de `results.csv` (900 filas, 300 turnos con llamada real al LLM)._

> **Aviso:** esta tirada usó el LLM simulado (`--fake`). Las tablas demuestran que el pipeline funciona de punta a punta; los números no son un baseline y hay que regenerarlos con `--http` contra el servicio con API key.

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
| Inicio / fin | 2026-10-12T17:19:17.097232+00:00 / 2026-10-12T17:19:33.823550+00:00 |

Las repeticiones miden las **ráfagas**: la segunda y tercera pasada de cada conversación envían exactamente los mismos prompts, así que son el test de la caché exacta (y, si cambia alguna palabra, de la semántica). Por eso la tabla resumen separa la latencia de todos los turnos de la de los turnos **sin caché**, que es la curva real del LLM.

## 1. Tabla resumen (escenario × tamaño de adjunto)

| Escenario | Adjunto | Turnos | P50 lat (ms) | P95 lat (ms) | P50 sin caché | P95 sin caché | Coste total (USD) | Hit exacto | Hit semántico | Recall medio | Latency budget OK | Cost budget OK |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| contradiction | 0 KB | 60 | 2 | 6 | 5 | 6 | 0.0271 | 67 % | 0 % | 94 % | 100 % | 100 % |
| contradiction | 5 KB | 60 | 3 | 5 | 3 | 5 | 0.0293 | 67 % | 0 % | 94 % | 100 % | 100 % |
| contradiction | 20 KB | 60 | 3 | 6 | 3 | 6 | 0.0327 | 67 % | 0 % | 94 % | 100 % | 100 % |
| contradiction | 50 KB | 60 | 3 | 6 | 3 | 6 | 0.0394 | 67 % | 0 % | 94 % | 100 % | 100 % |
| contradiction | 100 KB | 60 | 3 | 6 | 4 | 6 | 0.0416 | 67 % | 0 % | 94 % | 100 % | 100 % |
| growing | 0 KB | 60 | 3 | 5 | 3 | 6 | 0.0291 | 67 % | 0 % | 92 % | 100 % | 100 % |
| growing | 5 KB | 60 | 3 | 5 | 3 | 4 | 0.0305 | 67 % | 0 % | 92 % | 100 % | 100 % |
| growing | 20 KB | 60 | 3 | 6 | 3 | 6 | 0.0339 | 67 % | 0 % | 92 % | 100 % | 100 % |
| growing | 50 KB | 60 | 3 | 5 | 3 | 5 | 0.0406 | 67 % | 0 % | 92 % | 100 % | 100 % |
| growing | 100 KB | 60 | 3 | 8 | 6 | 8 | 0.0429 | 67 % | 0 % | 92 % | 100 % | 100 % |
| pivot | 0 KB | 60 | 3 | 4 | 3 | 5 | 0.0281 | 67 % | 0 % | 100 % | 100 % | 100 % |
| pivot | 5 KB | 60 | 3 | 6 | 5 | 6 | 0.0300 | 67 % | 0 % | 100 % | 100 % | 100 % |
| pivot | 20 KB | 60 | 2 | 5 | 4 | 5 | 0.0334 | 67 % | 0 % | 100 % | 100 % | 100 % |
| pivot | 50 KB | 60 | 3 | 6 | 3 | 6 | 0.0401 | 67 % | 0 % | 100 % | 100 % | 100 % |
| pivot | 100 KB | 60 | 3 | 7 | 4 | 7 | 0.0424 | 67 % | 0 % | 100 % | 100 % | 100 % |

## 2. Curva 1 — latencia vs tokens de entrada (solo turnos sin caché)

| tokens_in | Turnos | P50 latencia (ms) | P95 latencia (ms) | tokens_out medio | Coste medio (USD) | Con summarizer |
|---|---|---|---|---|---|---|
| 2.000–3.999 | 6 | 4 | 15 | 120 | 0.000623 | 0 % |
| 4.000–5.999 | 21 | 3 | 17 | 120 | 0.000840 | 0 % |
| 6.000–7.999 | 219 | 3 | 6 | 355 | 0.001230 | 96 % |
| 8.000–9.999 | 12 | 3 | 67 | 120 | 0.001455 | 0 % |
| 10.000–11.999 | 6 | 3 | 5 | 120 | 0.001662 | 0 % |
| 14.000–15.999 | 3 | 96 | 106 | 120 | 0.002460 | 0 % |
| 16.000–17.999 | 12 | 3 | 4 | 120 | 0.002644 | 0 % |
| 18.000–19.999 | 15 | 6 | 175 | 120 | 0.002922 | 0 % |
| 20.000–21.999 | 6 | 4 | 7 | 120 | 0.003157 | 0 % |

## 3. Curva 2 — coste acumulado vs turno (sin adjunto, primera repetición)

| Turno | contradiction: acumulado (turno · tokens_in) | growing: acumulado (turno · tokens_in) | pivot: acumulado (turno · tokens_in) |
|---|---|---|---|
| 1 | 0.00058 (0.00058 · 3.391 tok) | 0.00059 (0.00059 · 3.418 tok) | 0.00058 (0.00058 · 3.414 tok) |
| 2 | 0.00124 (0.00066 · 3.915 tok) | 0.00125 (0.00067 · 3.955 tok) | 0.00125 (0.00066 · 3.937 tok) |
| 3 | 0.00196 (0.00072 · 4.336 tok) | 0.00198 (0.00073 · 4.421 tok) | 0.00198 (0.00073 · 4.383 tok) |
| 4 | 0.00275 (0.00079 · 4.782 tok) | 0.00279 (0.00080 · 4.871 tok) | 0.00277 (0.00079 · 4.804 tok) |
| 5 | 0.00361 (0.00086 · 5.240 tok) | 0.00366 (0.00087 · 5.312 tok) | 0.00364 (0.00087 · 5.322 tok) |
| 6 | 0.00453 (0.00092 · 5.672 tok) | 0.00459 (0.00094 · 5.763 tok) | 0.00457 (0.00093 · 5.756 tok) |
| 7 | 0.00556 (0.00103 · 6.174 tok) | 0.00566 (0.00106 · 6.299 tok) | 0.00562 (0.00105 · 6.271 tok) |
| 8 | 0.00662 (0.00105 · 6.250 tok) | 0.00675 (0.00110 · 6.380 tok) | 0.00669 (0.00107 · 6.329 tok) |
| 9 | 0.00770 (0.00108 · 6.319 tok) | 0.00789 (0.00114 · 6.510 tok) | 0.00779 (0.00110 · 6.402 tok) |
| 10 | 0.00880 (0.00111 · 6.382 tok) | 0.00905 (0.00116 · 6.563 tok) | 0.00891 (0.00112 · 6.461 tok) |
| 11 | 0.00993 (0.00113 · 6.444 tok) | 0.01025 (0.00119 · 6.655 tok) | 0.01008 (0.00117 · 6.555 tok) |
| 12 | 0.01108 (0.00115 · 6.528 tok) | 0.01147 (0.00122 · 6.748 tok) | 0.01127 (0.00119 · 6.611 tok) |
| 13 | 0.01226 (0.00118 · 6.590 tok) | 0.01272 (0.00125 · 6.804 tok) | 0.01248 (0.00122 · 6.686 tok) |
| 14 | 0.01347 (0.00121 · 6.672 tok) | 0.01399 (0.00127 · 6.868 tok) | 0.01373 (0.00124 · 6.758 tok) |
| 15 | 0.01470 (0.00123 · 6.746 tok) | 0.01529 (0.00130 · 6.957 tok) | 0.01499 (0.00126 · 6.825 tok) |
| 16 | 0.01596 (0.00125 · 6.808 tok) | 0.01663 (0.00134 · 7.055 tok) | 0.01628 (0.00129 · 6.884 tok) |
| 17 | 0.01724 (0.00128 · 6.873 tok) | 0.01800 (0.00137 · 7.148 tok) | 0.01759 (0.00131 · 6.953 tok) |
| 18 | 0.01854 (0.00130 · 6.943 tok) | 0.01937 (0.00138 · 7.201 tok) | 0.01892 (0.00134 · 7.022 tok) |
| 19 | 0.01986 (0.00132 · 6.996 tok) | 0.02075 (0.00138 · 7.217 tok) | 0.02028 (0.00136 · 7.081 tok) |
| 20 | 0.02121 (0.00134 · 7.050 tok) | 0.02214 (0.00138 · 7.233 tok) | 0.02164 (0.00136 · 7.105 tok) |

## 4. Curva 3 — MemoryDriftMetric vs N (media sobre tamaños y repeticiones)

Recall = fracción de los hechos declarados en turnos k < N que aparecen literalmente (case-insensitive) en el resumen, las anclas o `project_metadata` del turno N. No mira el historial: mide lo que sobrevive a la ventana.

| N (turnos) | contradiction: recall (hechos esperados · nombre en título) | growing: recall (hechos esperados · nombre en título) | pivot: recall (hechos esperados · nombre en título) |
|---|---|---|---|
| 1 | – (0 hechos · título OK 100 %) | – (0 hechos · título OK 100 %) | – (0 hechos · título OK 100 %) |
| 3 | 75 % (4 hechos · título OK 100 %) | 80 % (5 hechos · título OK 100 %) | 100 % (5 hechos · título OK 100 %) |
| 6 | 86 % (7 hechos · título OK 100 %) | 88 % (8 hechos · título OK 100 %) | 100 % (7 hechos · título OK 100 %) |
| 10 | 100 % (10 hechos · título OK 100 %) | 100 % (12 hechos · título OK 100 %) | 100 % (11 hechos · título OK 100 %) |
| 20 | 100 % (19 hechos · título OK 100 %) | 95 % (22 hechos · título OK 100 %) | 100 % (21 hechos · título OK 100 %) |

### Hechos retirados y dónde viven los recordados

| Escenario | N final | Conversaciones con hechos retirados aún presentes | Hechos retirados presentes | Dónde se encuentran los hechos recordados (conteo) |
|---|---|---|---|---|
| pivot | 20 | 100 % | React Native | summary+anchors: 138, anchors: 102, summary+anchors+metadata: 75 |
| contradiction | 20 | 100 % | 30.000, 12 semanas | summary+anchors: 150, anchors: 75, summary+anchors+metadata: 60 |

## 5. Adjuntos grandes (turno 1 de cada conversación)

| Adjunto | Chars extraídos | P50 tokens_in | P50 latencia (ms) | P95 latencia (ms) | Coste medio turno 1 (USD) | Tier extended | Recall marcadores |
|---|---|---|---|---|---|---|---|
| 0 KB | 0 | 3.414 | 5 | 15 | 0.000583 | 0.00 | – |
| 5 KB | 5.085 | 4.704 | 17 | 17 | 0.000777 | 0.00 | 0 % |
| 20 KB | 20.184 | 8.479 | 46 | 67 | 0.001343 | 1.00 | 0 % |
| 50 KB | 49.979 | 15.928 | 96 | 106 | 0.002460 | 1.00 | 0 % |
| 100 KB | 60.047 | 18.446 | 157 | 175 | 0.002838 | 1.00 | 0 % |

## 6. Lectura

**Dónde empieza a romperse mi CAG.** Con la ventana de 6 turnos, el primer síntoma no es un error sino un coste: el summarizer entra en el turno 7 (el primero que expulsa un turno de la ventana) y a partir de ahí cada turno paga dos llamadas. El coste del turno 20 multiplica por 2.34 el del turno 1 (tokens de entrada ×2.09) y la latencia por 0.43; el 0 % de los turnos sin caché supera el presupuesto de latencia y el 0 % el de coste. La memoria se degrada en silencio: en `contradiction` el recall del fact-tracker no baja del 60 % en 20 turnos y termina en 100 %; en `growing` el recall del fact-tracker no baja del 60 % en 20 turnos y termina en 95 %; en `pivot` el recall del fact-tracker no baja del 60 % en 20 turnos y termina en 100 %. El nombre del proyecto sigue en el título de la estimación en el 100 % de las conversaciones en el turno 20.

**Qué dimensión domina y cuándo saltar a RAG.** Entre coste (+134 %), latencia (−57 %) y pérdida de memoria (−2 % de recall), la dimensión que más se degrada entre el turno 1 y el 20 es **coste**. Coste y latencia crecen de forma acotada porque la ventana y el resumen ponen un techo al contexto; lo que no está acotado es lo que se pierde para mantener ese techo: el resumen está limitado a 1.500 caracteres y las anclas a 24, así que cada turno nuevo desplaza hechos antiguos. Los hechos que sobreviven lo hacen por la memoria estructurada (`project_metadata`) o por una ancla literal; los que se reformulan en el resumen cuentan como perdidos. En `pivot`, el 100 % de las conversaciones llega al turno 20 con el stack retirado (React Native) todavía presente: `project_metadata` acumula tecnologías pero no sabe retirar ninguna. Con adjuntos, la latencia P50 del turno 1 pasa de 5 ms sin adjunto a 157 ms con 100 KB, y el recall de los marcadores del PDF baja hasta 0 % con 5 KB; el tope `ATTACHMENT_MAX_CHARS=60.000` trunca el de 100 KB, así que los marcadores del 75 % y del final del documento nunca llegan al modelo. El caso límite que justifica RAG es el que combina las dos curvas: un proyecto que supera la ventana (N > 6) y además arrastra documentación que no cabe (> 60.000 caracteres). Ahí el CAG solo puede elegir entre pagar el contexto completo en cada turno o resumirlo y perder recall; recuperar por relevancia solo lo que cada turno necesita es la única opción que no sacrifica ninguna de las tres curvas. **Esta tirada usó un LLM simulado: las cifras validan el pipeline, no son un baseline.**

## 7. Notas de método

- `turn_observed` se lee del campo `turn_observed` de la respuesta de `POST /sessions/{id}/estimate` y se contrasta con `GET /sessions/{id}.last_turn_observed`: es el mismo diccionario que se emite al log con structlog. No hay que parsear stdout ni reconciliar timestamps, y el runner funciona igual contra Docker, uvicorn o in-process.
- `LatencyBudgetMetric`, `CostBudgetMetric` y `MemoryDriftMetric` viven en `evals/stress/metrics.py` porque evalúan el shape de `turn_observed` y del snapshot de sesión, no una estimación suelta como las tres de `evals/metrics.py`; comparten `MetricResult` y `run_all_metrics`.
- `latency_ms` y `cost_usd` del turno incluyen la llamada del summarizer cuando la hay (`summarizer_*` las desglosa). Un acierto de caché registra 0 tokens y 0 USD.
- Los PDFs son sintéticos y deterministas (`evals/stress/fixtures/build_pdfs.py`); llevan cinco marcadores repartidos por el documento para medir el recall del adjunto en la estimación. No se versionan: el runner los regenera.
- No se ha tocado ninguna constante entre medidas (ventana de 6, resumen de 1.500 caracteres, 24 anclas, tope de adjunto de 60.000).
