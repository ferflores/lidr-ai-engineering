# Transcripciones de ejemplo

Tres transcripciones de reunión con cliente, usadas como entrada del diagnóstico de la sesión 09
([`../../arquitectura-actual.md`](../../arquitectura-actual.md), en la raíz del repo).

| Fichero | Perfil | Caracteres |
|---|---|---|
| `01_clear.txt` | El cliente describe con claridad qué necesita y menciona explícitamente tecnologías (Spring Boot, PostgreSQL, Kafka, AWS) y sector (fintech / pagos). | ~2.900 |
| `02_ambiguous.txt` | El cliente divaga, mezcla temas (historia de la empresa, reparto, facturación, personal, una web fallida) y solo en **dos frases** dice qué quiere construir: portal B2B con tarifas por cliente y petición de presupuesto. **Es la que se usa en el trace.** | ~8.400 |
| `03_hard.txt` | El cliente enumera varias features posibles (teleconsulta, portal de resultados, identidad, cita online, facturación a mutuas, cuadro de mandos), cambia de prioridad a mitad de la conversación y termina sin cerrar el alcance. | ~5.000 |

## Procedencia

El enunciado de la sesión 09 dice que estas tres transcripciones vienen con el material del curso, pero no
llegaron al repo ni a `docs/material/`. Están **reconstruidas** a partir de las descripciones del propio
enunciado (clara / ambigua / difícil) y con el mismo estilo que las transcripciones de las sesiones 02–05
(`estimador-cag/transcripciones/`): reunión real en español, con interrupciones, datos contradictorios y
nombres de empresa ficticios. Si aparece el material original, el trace se repite con él: los comandos de
la sección 2 del entregable no dependen del contenido del fichero.

El contenido es ficticio. Cualquier parecido con una empresa real es casual.
