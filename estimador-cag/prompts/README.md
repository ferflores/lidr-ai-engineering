# Prompts como artefactos versionados

Aquí viven los prompts del estimador, **fuera del código Python**. Cada versión es una carpeta con tres
ficheros y se carga con `app/services/prompts.py`:

```
prompts/
└── estimacion/                 # familia de plantillas (una por caso de uso)
    ├── v1/
    │   ├── template.toml       # metadatos: nombre, versión, descripción, fecha, changelog, variables
    │   ├── system.md.j2        # system prompt (Jinja2)
    │   └── user.md.j2          # mensaje de usuario (Jinja2)
    ├── v2/
    │   └── ...
    └── v3/                     # conversación con memoria: bloque <project_metadata> (sesión 05)
        └── ...
```

## Reglas

- **Una versión nunca se edita una vez publicada.** Para cambiar el prompt se crea `vN+1`, se explica el
  cambio en su `changelog` y se activa con `PROMPT_VERSION=vN+1` en `.env`. Así una estimación antigua se
  puede reproducir con la versión que la generó, y dos versiones se pueden comparar con la misma entrada
  (`prompt_version` en el cuerpo de `POST /api/v1/estimate`).
- **`[variables]` es el contrato de la plantilla.** Declara qué variables necesita cada fichero; si el código
  no las pasa, el registro falla antes de llamar al LLM. Jinja2 se carga con `StrictUndefined`, así que una
  variable no declarada pero usada también falla, en vez de renderizarse vacía.
- **El system prompt debe ser estable entre llamadas** (rol, reglas, formato, ejemplos CAG) y los datos de
  cada petición van en el mensaje de usuario. `v1` mezclaba la tarifa en el system prompt; `v2` lo corrige.
  Un prefijo estable es lo que permite cachear el prompt en el proveedor.
- **Lo que varía por turno va al final del system prompt.** `v3` inyecta la memoria del proyecto
  (`<project_metadata>`) después de los ejemplos CAG: el prefijo sigue siendo idéntico entre turnos y
  sesiones (cacheable en el proveedor) y solo cambia la cola. Si el bloque fuera al principio, cada cambio
  de memoria invalidaría la caché de todo el system prompt.
- Sintaxis Jinja2 con `trim_blocks` y `lstrip_blocks` activados (las líneas con `{% ... %}` no dejan huecos).
  Filtro extra `num`: imprime un número sin decimales innecesarios (`50.0 -> 50`, `47.5 -> 47.5`).

## Versiones

| Versión | Qué cambia |
|---------|------------|
| `v1`    | El prompt original de las sesiones 02 y 03, migrado tal cual desde `llm_service.py`. |
| `v2`    | Parámetros tipados (nombre, tipo, tarifa, plazo, enfoque, nivel de detalle) en el mensaje de usuario; dos formatos de salida. |
| `v3`    | `v2` + sección "Conversación con memoria" (cada respuesta es la estimación completa y actualizada; el dato más reciente manda) y bloque `<project_metadata>` al final del system prompt con los hechos del proyecto en curso (vacío en el primer turno). Explica los adjuntos delimitados por `--- attachment: <fichero> ---`. El mensaje de usuario no cambia. **Versión por defecto.** |
