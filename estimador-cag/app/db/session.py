"""Motor async de SQLAlchemy y la dependencia que abre una sesión por petición.

- Driver `asyncpg` (el recomendado por SQLAlchemy 2.0 para Postgres), vía la URL
  `postgresql+asyncpg://...` de `DATABASE_URL`.
- Un motor por URL y por proceso (`lru_cache`): el pool de conexiones se reutiliza entre peticiones en vez
  de abrir una conexión nueva cada vez.
- `get_db_session` **no** conecta: `async_sessionmaker` crea la sesión en memoria y la conexión se pide al
  pool en la primera consulta. Eso importa porque FastAPI resuelve las dependencias antes de validar el
  body: si conectara aquí, una petición mal formada contra una base caída daría 503 en vez del 422 que
  corresponde (el mismo problema que ya se documentó con `get_embedder` en la sesión 07).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings, get_settings


@lru_cache
def get_engine(database_url: str, echo: bool = False) -> AsyncEngine:
    """Motor async compartido por proceso. `pool_pre_ping` descarta conexiones que el servidor ya cerró."""
    return create_async_engine(database_url, echo=echo, pool_pre_ping=True)


@lru_cache
def get_sessionmaker(database_url: str) -> async_sessionmaker[AsyncSession]:
    """Fábrica de sesiones. `expire_on_commit=False` deja leer los objetos después del commit."""
    return async_sessionmaker(get_engine(database_url), expire_on_commit=False, autoflush=False)


async def get_db_session(settings: Settings = Depends(get_settings)) -> AsyncIterator[AsyncSession]:
    """Una sesión por petición; se cierra al terminar (y devuelve la conexión al pool)."""
    async with get_sessionmaker(settings.database_url)() as session:
        yield session
