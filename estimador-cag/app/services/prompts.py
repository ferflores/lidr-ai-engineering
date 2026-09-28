"""Registro de plantillas de prompt: carga `prompts/<nombre>/<versión>/` y las renderiza.

Los prompts son artefactos versionados que viven fuera del código (ver `prompts/README.md`).
Cada versión tiene un `template.toml` (metadatos y contrato de variables) y dos plantillas Jinja2:
`system.md.j2` y `user.md.j2`. Este módulo es el único punto del código que sabe dónde están y cómo
se renderizan; el servicio LLM solo pide "la plantilla `estimacion` en la versión X" y le pasa variables.

Fallos que se detectan antes de llamar al LLM (y por tanto antes de gastar tokens):
- versión inexistente -> `PromptTemplateNotFound`
- variable declarada en `[variables]` que no se pasa, o variable usada en la plantilla y no declarada
  (`StrictUndefined`) -> `PromptRenderError`
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import jinja2

from app.config import BASE_DIR, Settings, get_settings

PROMPTS_DIR = BASE_DIR / "prompts"
DEFAULT_TEMPLATE_NAME = "estimacion"


class PromptTemplateError(Exception):
    """Error base del registro de plantillas."""


class PromptTemplateNotFound(PromptTemplateError):
    """No existe la plantilla o la versión pedida."""


class PromptRenderError(PromptTemplateError):
    """La plantilla no se pudo renderizar: falta alguna variable o la sintaxis es incorrecta."""


def _num(value: object) -> str:
    """Filtro Jinja2 `num`: número sin decimales innecesarios (50.0 -> 50, 47.5 -> 47.5)."""
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


@lru_cache
def _environment() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(PROMPTS_DIR),
        undefined=jinja2.StrictUndefined,  # una variable no definida es un error, no un hueco en blanco
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
        autoescape=False,  # son prompts de texto, no HTML
    )
    env.filters["num"] = _num
    return env


@dataclass(frozen=True)
class PromptTemplate:
    """Una versión concreta de una plantilla, ya cargada y lista para renderizar."""

    name: str
    version: str
    description: str
    created: date | None
    changelog: tuple[str, ...]
    system_variables: tuple[str, ...]
    user_variables: tuple[str, ...]
    path: Path

    @property
    def id(self) -> str:
        return f"{self.name}/{self.version}"

    @property
    def system_source(self) -> str:
        """Texto crudo de `system.md.j2` (para mostrarlo en la interfaz)."""
        return (self.path / "system.md.j2").read_text(encoding="utf-8")

    @property
    def user_source(self) -> str:
        """Texto crudo de `user.md.j2` (para mostrarlo en la interfaz)."""
        return (self.path / "user.md.j2").read_text(encoding="utf-8")

    def render_system(self, **variables: object) -> str:
        return self._render("system", self.system_variables, variables)

    def render_user(self, **variables: object) -> str:
        return self._render("user", self.user_variables, variables)

    def _render(self, kind: str, declared: tuple[str, ...], variables: dict[str, object]) -> str:
        missing = sorted(set(declared) - set(variables))
        if missing:
            raise PromptRenderError(
                f"Faltan variables para {self.id}/{kind}.md.j2: {', '.join(missing)} "
                f"(declaradas en template.toml [variables].{kind})"
            )
        template_path = f"{self.name}/{self.version}/{kind}.md.j2"
        try:
            return _environment().get_template(template_path).render(**variables).strip()
        except jinja2.UndefinedError as exc:
            raise PromptRenderError(f"La plantilla {template_path} usa una variable no definida: {exc}") from exc
        except jinja2.TemplateError as exc:
            raise PromptRenderError(f"Error en la plantilla {template_path}: {exc}") from exc


def _version_key(version: str) -> tuple[int, str]:
    digits = version[1:] if version.startswith("v") else version
    return (int(digits), version) if digits.isdigit() else (-1, version)


def list_versions(name: str = DEFAULT_TEMPLATE_NAME) -> list[str]:
    """Versiones disponibles de una plantilla, de la más antigua a la más reciente (`v1`, `v2`, ...)."""
    family = PROMPTS_DIR / name
    if not family.is_dir():
        raise PromptTemplateNotFound(f"No existe la plantilla '{name}' en {PROMPTS_DIR}")
    versions = [p.name for p in family.iterdir() if p.is_dir() and (p / "template.toml").is_file()]
    return sorted(versions, key=_version_key)


def latest_version(name: str = DEFAULT_TEMPLATE_NAME) -> str:
    versions = list_versions(name)
    if not versions:
        raise PromptTemplateNotFound(f"La plantilla '{name}' no tiene ninguna versión en {PROMPTS_DIR / name}")
    return versions[-1]


@lru_cache
def load_template(name: str = DEFAULT_TEMPLATE_NAME, version: str = "") -> PromptTemplate:
    """Carga (y cachea) una versión de la plantilla. Con `version` vacía, la más reciente."""
    version = version or latest_version(name)
    path = PROMPTS_DIR / name / version
    if not (path / "template.toml").is_file():
        raise PromptTemplateNotFound(
            f"No existe la versión '{version}' de la plantilla '{name}'. Disponibles: {', '.join(list_versions(name))}"
        )
    for required in ("system.md.j2", "user.md.j2"):
        if not (path / required).is_file():
            raise PromptTemplateNotFound(f"A la plantilla {name}/{version} le falta {required}")

    meta = tomllib.loads((path / "template.toml").read_text(encoding="utf-8"))
    prompt = meta.get("prompt", {})
    variables = meta.get("variables", {})
    if prompt.get("name") != name or prompt.get("version") != version:
        raise PromptTemplateError(
            f"{path / 'template.toml'} declara {prompt.get('name')}/{prompt.get('version')} "
            f"pero está en la carpeta {name}/{version}"
        )
    created = prompt.get("created")
    return PromptTemplate(
        name=name,
        version=version,
        description=str(prompt.get("description", "")),
        created=created if isinstance(created, date) else None,
        changelog=tuple(str(line) for line in prompt.get("changelog", [])),
        system_variables=tuple(variables.get("system", [])),
        user_variables=tuple(variables.get("user", [])),
        path=path,
    )


def get_template(settings: Settings | None = None, version: str | None = None) -> PromptTemplate:
    """La plantilla de estimación en la versión pedida o, si no se pide, en la configurada (`PROMPT_VERSION`)."""
    settings = settings or get_settings()
    return load_template(DEFAULT_TEMPLATE_NAME, version or settings.prompt_version)
