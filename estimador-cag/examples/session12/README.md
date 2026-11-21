# Pre-work 12 · Estimación de la plataforma RUTA

El servicio recibe la transcripción de una reunión y devuelve un presupuesto por componente, las referencias históricas y la traza de decisiones. En el caso RUTA el cliente solicita cuatro entregables: API logística, integración con SAP, aplicación móvil de reparto y panel de analítica.

- [Agente del servicio FastAPI](../../app/agent/)
- [Transcripción compleja de RUTA](sample_transcript_complex.txt)
- [Traza real de GPT-5 con el buscador híbrido](trace_complex.json)
- [Material original de LIDR, rama session_12](https://github.com/LIDR-academy-temp/ai-engineering/tree/session_12/estimator/exercises/session-12)

## Resultado de la ejecución real

El modelo `gpt-5-2025-08-07`, con esfuerzo `medium`, realizó siete búsquedas, pidió un cálculo consolidado y terminó en el sexto turno. Las cuatro primeras búsquedas correspondieron a los cuatro entregables de la reunión.

| Componente | Horas estimadas | Coste a 50 €/h |
|---|---:|---:|
| Backend de pedidos, rutas y seguimiento | 97,8 | 4.890 € |
| Integración con SAP | 115,0 | 5.750 € |
| App de reparto con modo offline | 149,5 | 7.475 € |
| Panel de analítica | Pendiente | Pendiente |
| **Subtotal con referencias** | **362,3** | **18.115 €** |

El agente volvió a buscar evidencia para el panel. Los resultados describían operaciones logísticas y facturación, sin un precedente comparable de analítica con KPIs y alertas. El JSON conserva esa partida con `unbudgeted: true` y marca el conjunto como `complete: false`.

Hay otra limitación para discutir en el directo. Las referencias del backend describen piezas distintas de un mismo proyecto histórico. Aplicar la mediana las trata como alternativas y puede subestimar el alcance combinado. Para convertir el ejercicio en una oferta habría que estimar esas piezas por separado y revisar posibles solapamientos.

## Qué hace el agente

GPT-5 decide qué buscar y cuándo consolidar. El servicio Python recorre las llamadas devueltas por la Responses API, ejecuta las herramientas, devuelve cada resultado con su `call_id` y continúa con `previous_response_id`. El flujo termina cuando el modelo envía su mensaje final tras un cálculo consolidado. El límite de seguridad es de 12 turnos. La traza guarda el objetivo público de cada acción, sus argumentos y el resultado observado.

`search_budgets` delega en el buscador de la sesión 10: recuperación vectorial y full-text, fusión RRF y reranking con el cross-encoder. Devuelve el identificador del componente histórico, su alcance, sector y horas. El agente puede reformular la búsqueda si las referencias son débiles.

`calculate_estimate` comprueba que cada referencia y sus horas se recuperaron para ese mismo componente. Usa la mediana de horas para reducir el efecto de un presupuesto atípico, añade el 15 % de contingencia del ejercicio y redondea cada partida a una décima de hora. Convierte las horas a euros con la tarifa del proyecto (50 €/h por defecto). Un componente sin referencia queda pendiente, aporta cero al subtotal y hace que la estimación se marque como incompleta.

## Ejecución

Desde `estimador-cag/`, con las dependencias del proyecto y `OPENAI_API_KEY` configuradas, se puede depurar el caso simple con el dataset de referencia y ejecutar el caso complejo con el buscador del proyecto:

```sh
uv run python examples/session12/run.py examples/session12/sample_transcript_simple.txt --model gpt-5-mini --retrieval reference
uv run python examples/session12/run.py examples/session12/sample_transcript_complex.txt --model gpt-5 --retrieval hybrid --output examples/session12/trace_complex.json
```

El modo `hybrid` necesita PostgreSQL con pgvector, el corpus ingestado y el reranker del proyecto. La [guía del servicio](../../README.md) describe cómo levantarlo. El modo `reference` usa la función de recuperación de LIDR incluida junto a los dos archivos de entrada. Ambos modos llaman a la API de OpenAI con esfuerzo `medium`. La opción `--retrieval auto` usa el corpus del proyecto si encuentra chunks en la base y, si la base está vacía o fuera de línea, selecciona el de referencia. El JSON guarda cuál se usó, el modelo devuelto por la API y la huella SHA-256 de la transcripción.

El backend de negocio puede hacer una sola petición al servicio:

```sh
jq -n --rawfile transcript examples/session12/sample_transcript_complex.txt \
  '{transcript: $transcript, hourly_rate_eur: 50}' |
  curl --fail-with-body -sS http://localhost:8000/api/v1/agent/estimate \
    -H 'Content-Type: application/json' --data-binary @-
```

## Qué comprueba la prueba de integración

La prueba ejecuta el SDK de OpenAI con un transporte HTTP simulado, el mismo bucle del servicio, el buscador de referencia del curso y el cálculo real. Predefine cuatro consultas para poder verificar de forma repetible la correspondencia entre llamadas y respuestas, las referencias, el total y la parada. El campo `execution` distingue esta prueba de una corrida en vivo.

```sh
uv run python -m pytest -q tests/test_estimation_agent.py
uv run python -m tests.test_estimation_agent > examples/session12/trace_complex_offline.json
```
