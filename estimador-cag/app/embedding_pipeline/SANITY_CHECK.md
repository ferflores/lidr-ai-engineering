# Sanity check de los embeddings

Paso 7 del pre-work de la sesión 07. Comprueba que el pipeline funciona de punta a punta y que los embeddings
discriminan entre textos cercanos y lejanos. No es una validación formal de calidad de retrieval: eso llega
más adelante con `recall@k` y NDCG.

**Cómo se generó:** `uv run python scripts/compare.py --text-a "..." --text-b "..."` con
`text-embedding-3-small` (1536 dimensiones, sin truncar) el **2026-10-16**. Coste de las tres parejas:
53 tokens de entrada, 0,00000106 USD.

## Resultados

| Pareja | Textos | Esperado | Obtenido |
|---|---|---|---|
| **A** — semánticamente cercanos | "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" vs "Authorization service using JSON Web Tokens for a banking application" | > 0.6 | **0.5957** |
| **B** — no relacionados | el texto 1 de la pareja A vs "Database migration from MySQL to PostgreSQL with zero downtime" | < 0.4 | **0.1920** |
| **C** — genéricos y ambiguos | "Backend services" vs "API development" | sin expectativa | **0.5407** |

Los tres vectores vienen normalizados por el modelo: las normas medidas caen entre 0,9999 y 1,0003 (la
desviación es precisión de punto flotante en el JSON de la API). Con norma 1, coseno y producto escalar son
intercambiables, que es lo que dice la model card de OpenAI.

## Lectura

**La estructura es correcta pero los umbrales no son los que sugiere la intuición.** El orden es el esperado
(A > C > B, con B muy por debajo) y la separación entre cercanos y lejanos es amplia: 0.5957 frente a 0.1920,
casi 0.4 de margen. El pipeline discrimina. Pero la pareja A se queda en 0.5957, **por debajo del 0.6
orientativo del enunciado**, pese a que los dos textos significan prácticamente lo mismo. Si alguien hubiera
puesto un umbral de corte en 0.6 "porque suena a muy similar", este par habría quedado fuera del retrieval.

**El resultado que más llama la atención es la pareja C.** Dos expresiones genéricas de dos palabras,
"Backend services" y "API development", dan 0.5407: casi lo mismo que dos descripciones técnicas completas que
sí son equivalentes (0.5957). La diferencia entre ambas parejas es de solo 0.055. La explicación es que el
coseno mide proximidad en el espacio, no cantidad de información compartida: dos textos vagos del mismo campo
caen en una zona densa y cercana del espacio, y la especificidad no suma distancia por sí sola. La consecuencia
práctica es directa: **un valor alto de similitud no significa que el chunk recuperado sea informativo**, y los
chunks genéricos del corpus pueden competir con los específicos en cualquier consulta de su dominio.

**Dentro del corpus real el suelo sube todavía más.** Midiendo entre chunks del `data/budgets_sample.json`
ingerido, dos componentes sin relación (autenticación OAuth frente a impresión de etiquetas GS1, sectores
distintos) dan 0.4313, por encima del 0.4 que el enunciado propone como techo de "no relacionado". Es el efecto
que anticipa el artículo 1 con la maldición de la dimensionalidad, agravado aquí porque todos los chunks
comparten la misma plantilla y la misma cabecera de contexto: esa estructura común es un suelo de similitud que
todos arrastran. Para comparación, dos componentes de pagos de clientes distintos (PSD2 open banking y 3-D
Secure) dan 0.6765 y dos de autenticación de sectores distintos dan 0.5382.

**Conclusión operativa:** los umbrales hay que calibrarlos sobre este corpus, no sobre las cifras del
enunciado. Con estos datos, el rango útil de decisión está entre 0.43 (ruido del propio corpus) y 0.68
(componentes del mismo dominio), que es una ventana estrecha. Para el directo queda la pregunta de si conviene
rankear por top-k en lugar de por umbral absoluto, y si la cabecera de contexto, que claramente ayuda a
discriminar entre sectores (ver abajo), está a la vez comprimiendo el rango de similitudes del corpus.

## Comprobación extra: ¿sirve la cabecera de contexto?

El chunker antepone a cada componente dos líneas con el contexto del presupuesto padre. Medido con un brief en
español del sector financiero ("Necesitamos un servicio de autenticación con flujos OAuth para una app móvil
del sector financiero") contra el mismo componente de autenticación con y sin esa cabecera:

| | Componente de fintech | Componente de e-commerce | Margen |
|---|---|---|---|
| **Con** cabecera | 0.4920 | 0.3834 | **+0.1086** |
| **Sin** cabecera | 0.3971 | 0.3857 | +0.0114 |

Sin la cabecera los dos componentes de autenticación quedan prácticamente empatados frente a la consulta
(0.3971 vs 0.3857): el retriever no tendría forma de preferir el del sector correcto. Con ella, el margen se
multiplica por más de nueve. Es la justificación medida de la decisión 2 del chunker, y encaja con los 15-25
puntos de accuracy que el material atribuye a los *contextual chunk headers*.
