# Calibración del umbral de la caché semántica

Medido el **2026-10-16** con `text-embedding-3-small` sobre peticiones del dominio del estimador.
Coste de la medición: menos de 0,00001 USD.

El material de la sesión 04 recomienda un umbral de **0,90-0,93** para un estimador. Sobre este corpus ese
umbral deja la caché semántica prácticamente inservible, y conviene saber por qué antes de activarla.

## Los pares medidos

Cinco parejas que piden **lo mismo** escrito de otra forma, y cinco que piden **cosas distintas**.
`jaccard` es el backend actual del proyecto, `coseno` el de embeddings.

| | Pareja | Jaccard | Coseno |
|---|---|---|---|
| = | tienda online de la panadería / comercio electrónico para el obrador | 0,000 | **0,594** |
| = | añade el módulo de pagos con Stripe / incluye la pasarela de pago de Stripe | 0,167 | 0,758 |
| = | landing page con formulario / página de aterrizaje con formulario | 0,455 | 0,840 |
| = | app móvil para pedir cita / aplicación para smartphone donde reservar cita | 0,250 | 0,841 |
| = | cuánto cuesta la tienda online / cuánto costaría la tienda online | 0,667 | 0,913 |
| ≠ | tienda online / migrar la base de datos a PostgreSQL | 0,154 | 0,182 |
| ≠ | añade pagos con Stripe / quita la sección de blog del alcance | 0,077 | 0,315 |
| ≠ | landing page / ERP completo para una fábrica | 0,154 | 0,332 |
| ≠ | app móvil para pedir cita / panel de administración web | 0,143 | 0,393 |
| ≠ | añade el módulo de pagos con Stripe / añade el módulo de facturación con Holded | 0,556 | **0,627** |

## Qué pasa con cada umbral

| Umbral | Reformulaciones que acierta | Peticiones distintas que serviría mal |
|---|---|---|
| 0,90 (lo que recomienda el curso) | **1 de 5** | 0 de 5 |
| 0,85 | 1 de 5 | 0 de 5 |
| 0,80 | 3 de 5 | 0 de 5 |
| **0,75 (el que se ha fijado)** | **4 de 5** | **0 de 5** |
| 0,70 | 4 de 5 | 0 de 5 |
| 0,60 | 4 de 5 | 1 de 5 |
| 0,55 | 5 de 5 | 1 de 5 |

## Lectura

**El umbral del material no se puede copiar.** Con 0,90 la caché semántica solo reconoce la pareja que ya
reconocería casi cualquier método, la que solo cambia un verbo. Las reformulaciones de verdad, las que
cambian el vocabulario entero, caen entre 0,59 y 0,84. Es exactamente el mismo hallazgo que dejó el sanity
check de la sesión 07: los umbrales se calibran sobre el corpus propio.

**Las dos clases se solapan, así que ningún umbral es perfecto.** La peor reformulación equivalente da
0,594 y la peor pareja distinta da 0,627. El margen es **negativo**: cualquier umbral que acierte la
primera servirá mal la segunda. El par que lo provoca es revelador, «añade el módulo de pagos con Stripe»
frente a «añade el módulo de facturación con Holded»: misma estructura, mismo tipo de petición, producto
distinto. Para un humano son peticiones claramente distintas; en el espacio vectorial están cerca.

**0,75 es el punto elegido** porque maximiza aciertos sin ningún falso positivo en la muestra, y deja un
margen de 0,12 con la peor pareja distinta. No es un número universal: con otro corpus habría que repetir
esta tabla.

**Qué cuesta tenerlo activo.** Cada consulta a la caché embebe el mensaje del usuario, unos 400-700 ms
medidos y 0,02 USD por millón de tokens. Eso se paga en todas las peticiones, acierten o no; lo que se
ahorra es la llamada al LLM de las que aciertan. Con el coste por estimación en torno a 0,0005 USD y el
embedding en 0,0000003 USD, el embedding se amortiza con una tasa de aciertos muy baja; lo que no se
amortiza tan fácil es la **latencia** añadida a los fallos.

## Cómo reproducirlo

```bash
SEMANTIC_CACHE_BACKEND=embeddings uv run python -c "
from app.config import get_settings
from app.services.response_cache import cosine
from app.services.semantic_cache_backend import make_cache_embedder
embed = make_cache_embedder(get_settings())
print(cosine(embed('tu petición A'), embed('tu petición B')))
"
```
