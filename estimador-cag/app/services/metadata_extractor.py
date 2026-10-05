"""Extracción heurística de `ProjectMetadata` a partir de cada turno.

Después de cada respuesta del LLM hay que actualizar la memoria del proyecto. Dos formas de hacerlo:

- **Heurística** (este módulo): parsear la estimación, la transcripción y los adjuntos con expresiones
  regulares y listas cerradas. Cero llamadas extra, determinista, se prueba sin LLM.
- **LLM extractor**: una segunda llamada con un prompt que devuelve el JSON de `ProjectMetadata`. Más
  robusto ante texto libre, pero duplica llamadas, latencia y coste por turno, y añade otra salida que validar.

Aquí se elige la heurística porque el prompt `v2`/`v3` ya obliga al modelo a un **formato de salida
fijo** (título `## Estimación: ...`, sección `### Resumen del alcance`, líneas `**Equipo recomendado: ...**`
y `**Total estimado: N horas**`). Ese formato es un contrato que ya se paga en el prompt: parsearlo es
barato y fiable. Lo que la heurística no cubre (tecnologías fuera del vocabulario, matices del alcance)
es exactamente lo que justificaría pasar al extractor LLM más adelante; la función `extract_project_metadata`
es el único punto que habría que sustituir.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from app.schemas.estimation import EstimationRequest
from app.services.attachments import ExtractedAttachment
from app.services.sessions import ProjectMetadata

# Vocabulario cerrado de tecnologías, plataformas y herramientas que suelen aparecer en reuniones de
# alcance. Se busca con límites de palabra y sin distinguir mayúsculas; el nombre canónico es el de la lista.
TECHNOLOGIES: tuple[str, ...] = (
    # Lenguajes y frameworks
    "Python", "FastAPI", "Django", "Flask", "Node.js", "Express", "NestJS", "TypeScript", "JavaScript",
    "React Native", "React", "Next.js", "Vue", "Nuxt", "Angular", "Svelte", "Java", "Spring Boot", "Kotlin",
    "Swift", "Flutter", ".NET", "C#", "PHP", "Laravel", "Ruby on Rails", "Go",
    # Plataformas
    "iOS", "Android", "WordPress", "WooCommerce", "Shopify", "PrestaShop", "Magento", "Webflow",
    # Datos
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "SQLite", "Supabase", "Firebase",
    "BigQuery", "Snowflake", "Airflow", "dbt", "Kafka", "RabbitMQ", "Power BI", "Tableau", "Looker",
    # Infraestructura
    "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Terraform", "Vercel", "Netlify", "Heroku",
    # Pagos, CRM, marketing y mensajería
    "Stripe", "PayPal", "Bizum", "Redsys", "HubSpot", "Salesforce", "Mailchimp", "SendGrid", "Twilio",
    "WhatsApp", "Telegram", "Slack", "Zapier", "n8n", "Google Maps", "Google Analytics", "Auth0", "Keycloak",
    # Diseño, ofimática y gestión
    "Figma", "Excel", "Google Sheets", "SAP", "Odoo", "Holded", "Notion", "Jira",
    # IA
    "OpenAI", "Anthropic", "LangChain", "Streamlit", "GraphQL", "OAuth",
)

# Los nombres más largos primero: "React Native" se reconoce antes que "React" y se enmascara.
_TECH_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (name, re.compile(rf"(?<![\w.#+-]){re.escape(name)}(?![\w-])", re.IGNORECASE))
    for name in sorted(TECHNOLOGIES, key=len, reverse=True)
]

_TITLE = re.compile(r"^##\s*Estimaci[oó]n:\s*(.+?)\s*$", re.MULTILINE)
_SCOPE = re.compile(r"^###\s*Resumen del alcance\s*\n(.+?)(?=\n#{2,3}\s|\n\*\*|\Z)", re.MULTILINE | re.DOTALL)
_TEAM = re.compile(r"\*\*Equipo recomendado:\s*(.+?)\*\*", re.DOTALL)
_TOTAL = re.compile(r"\*\*Total estimado:\s*([\d.,]+)\s*h", re.IGNORECASE)
_NUMBER_WORDS = {"un": 1, "una": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6}


def extract_project_metadata(
    request: EstimationRequest,
    estimation: str,
    attachments: Sequence[ExtractedAttachment] = (),
    current: ProjectMetadata | None = None,
) -> ProjectMetadata:
    """Hechos del proyecto que aporta un turno: lo que dijo quien pide (parámetros), lo que se habló
    (transcripción y adjuntos) y lo que concluyó el modelo (estimación). Se fusionan con la memoria anterior
    mediante `ProjectMetadata.merged`.

    El nombre del proyecto es especial: lo fija el formulario o, si no, el título de la primera estimación,
    y después NO se toma del título de cada respuesta (probado con gpt-4o-mini: reformulaba el título en
    cada turno y la memoria iba a la deriva). Solo el formulario lo cambia.
    """
    sources = [request.transcription, estimation, *(a.text for a in attachments)]
    team = _first(_TEAM, estimation)
    if request.project_name:
        project_name = request.project_name
    elif current is not None and current.project_name:
        project_name = None  # se conserva el de la memoria
    else:
        project_name = _first(_TITLE, estimation)
    return ProjectMetadata(
        project_name=project_name,
        project_type=request.project_type,
        assumed_team=team,
        assumed_team_size=team_size(team) if team else None,
        mentioned_technologies=find_technologies(sources),
        agreed_scope=_scope(estimation),
        hourly_rate_eur=request.hourly_rate_eur,
        deadline_weeks=request.deadline_weeks,
        last_total_hours=parse_hours(_first(_TOTAL, estimation)),
        attached_documents=[a.filename for a in attachments],
    )


def find_technologies(texts: Iterable[str]) -> list[str]:
    """Tecnologías del vocabulario presentes en los textos, en orden de aparición y sin duplicados."""
    found: list[str] = []
    for text in texts:
        masked = text
        hits: list[tuple[int, str]] = []
        for name, pattern in _TECH_PATTERNS:  # los nombres largos primero: "React Native" antes que "React"
            match = pattern.search(masked)
            if match is None:
                continue
            hits.append((match.start(), name))
            masked = pattern.sub(lambda m: " " * len(m.group(0)), masked)  # y lo encontrado se enmascara
        for _, name in sorted(hits):
            if name not in found:
                found.append(name)
    return found


def team_size(team: str) -> int | None:
    """Personas en "1 desarrollador full-stack, 1 diseñador UX/UI" -> 2; sin cifras, cuenta los perfiles."""
    numbers = [int(n) for n in re.findall(r"\b(\d+)\b", team)]
    words = [_NUMBER_WORDS[w] for w in re.findall(r"\b(un|una|uno|dos|tres|cuatro|cinco|seis)\b", team.lower())]
    if numbers or words:
        return sum(numbers) + sum(words) or None
    profiles = [p for p in re.split(r",|\by\b|\+|;", team) if p.strip()]
    return len(profiles) or None


def parse_hours(value: str | None) -> float | None:
    """"1.250" (miles en español) -> 1250; "47,5" -> 47.5; "80" -> 80."""
    if not value:
        return None
    cleaned = value.strip().rstrip(".,")
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", cleaned):
        cleaned = cleaned.replace(".", "")
    else:
        cleaned = cleaned.replace(".", "") if cleaned.count(".") > 1 else cleaned
        cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _scope(estimation: str) -> str | None:
    match = _SCOPE.search(estimation)
    if not match:
        return None
    scope = " ".join(line.strip() for line in match.group(1).strip().splitlines() if line.strip())
    return scope or None


def _first(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1).strip() if match else None
