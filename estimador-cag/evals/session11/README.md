# Sesión 11 · Citas por partida y evaluación Ragas

La pregunta de esta entrega es si cada partida se puede rastrear hasta el presupuesto que la respalda y cuánta información útil encuentra el estimador para cinco solicitudes de clientes.

## Cómo funciona

1. El servicio recibe la transcripción en `POST /api/v1/estimate` con `response_format: "cited"`. Utiliza el recuperador híbrido de la sesión 10 y entrega al generador los cinco primeros fragmentos después del reranking.
2. La Responses API genera partidas mediante `responses.parse`. El esquema exige el componente, las horas, una explicación, el estado de respaldo y las citas. Cada cita identifica el fragmento, su presupuesto padre y un extracto literal.
3. Python compara las citas con los fragmentos que recibió esa petición. Comprueba el identificador, el documento padre, la presencia textual del extracto y las horas históricas. Una cita colgante o una cifra sin respaldo deja la partida fuera del subtotal. El informe conserva el motivo y el `request_id` permite encontrar el evento correspondiente en los logs.
4. Ragas evalúa las mismas cinco preguntas de la sesión 10. Cada muestra conserva la pregunta, el texto de la respuesta, los fragmentos ordenados y la respuesta de referencia. Las referencias se prepararon antes de la ejecución, a partir de los componentes y horas del corpus. Son criterios de trabajo para este ejercicio, abiertos a revisión por un estimador humano.

[Esquema, prompt y verificación](../../app/services/cited_estimate.py) · [Golden set con las cinco referencias](../retrieval/golden_set.json) · [Pruebas de integridad](../../tests/test_cited_estimation.py)

## Reproducción

Desde `estimador-cag/`, con Python 3.11, la base pgvector del proyecto cargada con el corpus y la clave de OpenAI configurada en `.env`:

```sh
uv run python -m evals.session11.collect
uv run --env-file .env --script evals/session11/score.py
uv run python -m evals.session11.citation_example > evals/session11/citation_report.json
uv run pytest -q tests/test_cited_estimation.py
```

El primer comando prepara `samples.json`. El segundo utiliza su propio entorno de dependencias para ejecutar Ragas 0.4.4 y guarda las cuatro métricas de cada consulta y su media en `scores.json`. El tercero toma una estimación real y crea una copia con un identificador deliberadamente inexistente para demostrar el error. El archivo conserva ambos informes etiquetados.

Las horas respaldadas son referencias históricas de componentes comparables. La comprobación determinista detecta una cifra alterada, pero la pertinencia del precedente requiere juicio sobre el alcance. Las puntuaciones del juez también pueden variar entre ejecuciones, incluso conservando las mismas muestras.
