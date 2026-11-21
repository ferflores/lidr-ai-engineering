# Sesión 11 · Citas por partida y evaluación Ragas

¿Cuánto de un presupuesto generado se puede rastrear hasta una referencia y qué parte del encargo quedó sin cubrir? El ejercicio usa las cinco solicitudes de clientes de la sesión 10 para medir ambas cosas.

## Entrega

- [Tabla de las cuatro métricas por consulta y sus medias](baseline.md)
- [Ejemplo real del verificador y prueba de una cita colgante](citation_report.json)
- [Cinco estimaciones con sus fragmentos, citas y referencias esperadas](samples.json)
- [Golden set](../retrieval/golden_set.json) · [Esquema, prompt y verificador](../../app/services/cited_estimate.py) · [Pruebas](../../tests/test_cited_estimation.py)

## Qué muestran los resultados

En la flota, el juez dio 0,846 de fidelidad y 0,000 de cobertura: el estimador citó precedentes de seguimiento y optimización de rutas, mientras que el mapa del gestor, las alertas de desvío y el historial requerían otros precedentes. En el caso industrial, las tres cifras verificadas fueron 200 h de captura de datos, 110 h de OEE y 60 h de SAP, pero la respuesta añadió dos partidas ajenas al encargo. En el banco, la precisión del juez fue 0,000 pese a que el contexto contenía las referencias de alta y back office: esa discrepancia merece revisión manual antes de usar el indicador para aprobar un despliegue.

## Cómo funciona

1. El servicio recibe una transcripción en `POST /api/v1/estimate` con `response_format: "cited"`. Su buscador híbrido recupera candidatos y el reranker selecciona los cinco primeros.
2. La Responses API genera partidas mediante `responses.parse` y un esquema Pydantic v2. Cada partida contiene su alcance, horas, justificación, estado de respaldo y citas. La cita identifica el fragmento, su presupuesto padre y un extracto literal.
3. Python comprueba los dos identificadores, busca el extracto en el texto entregado al modelo y compara las horas con la cifra del precedente. Una partida sin evidencia suficiente queda con horas nulas y fuera del subtotal. El informe conserva el motivo y el `request_id` permite localizarlo en los logs.
4. Ragas recibe cuatro elementos por caso: la pregunta, la respuesta del estimador, los cinco fragmentos ordenados y una referencia preparada a partir del corpus antes de ejecutar la evaluación. Las referencias son criterios de trabajo abiertos a revisión por un estimador humano.

## Reproducción

Desde `estimador-cag/`, con Python 3.11 y la clave de OpenAI en `.env`:

```sh
uv run --env-file .env python -m evals.session11.public_snapshot
uv run --env-file .env --script evals/session11/score.py
uv run python -m evals.session11.citation_example > evals/session11/citation_report.json
uv run pytest -q tests/test_cited_estimation.py
```

La corrida publicada reproduce el orden exacto de la configuración D del [informe de recuperación de la sesión 10](../retrieval/REPORT.md). Reconstruye cada pasaje desde el [corpus de ejemplo](../../data/budgets_sample.json) con el mismo formateador del servicio y comprueba las huellas SHA-256 del corpus, el ranking y las preguntas. De esta forma se pueden evaluar nuevas respuestas con el mismo contexto. Para repetir también la búsqueda sobre un índice propio, arranca pgvector, carga el corpus y ejecuta `uv run --env-file .env python -m evals.session11.collect` en lugar del primer comando.

El segundo comando usa Ragas 0.4.4 y escribe tanto el JSON con precisión completa como la tabla Markdown. El tercero toma una estimación de la corrida y sustituye deliberadamente un ID por otro inexistente en una copia. El informe distingue el caso observado y el cambio controlado.

Las horas representan precedentes históricos. La comprobación automática puede validar la fuente y aun así aceptar un precedente con alcance distinto, o sumar dos alternativas que se solapan. Las calificaciones del juez también pueden variar al repetirlas sobre las mismas muestras.
