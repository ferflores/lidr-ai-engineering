# Evaluación Ragas · sesión 11

| Consulta | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---:|---:|---:|---:|
| pharmacy-online-orders | 0.444 | 0.414 | 0.000 | 1.000 |
| credit-card-onboarding-tool | 0.786 | 0.393 | 0.000 | 0.400 |
| van-fleet-tracking | 0.846 | 0.673 | 0.000 | 0.000 |
| plant-oee-sap | 0.545 | 0.560 | 1.000 | 0.750 |
| insurance-customer-portal | 0.824 | 0.599 | 0.250 | 0.800 |
| **Media** | 0.689 | 0.528 | 0.250 | 0.590 |

Juez `gpt-4.1-mini-2025-04-14`. Embeddings `text-embedding-3-small`. Ragas 0.4.4.

[Puntuaciones sin redondear](scores.json) · [Respuestas y fragmentos](samples.json)

Las medias dan el mismo peso a cada consulta.
