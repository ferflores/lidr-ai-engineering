"""Entorno de Alembic (plantilla async) adaptado al proyecto.

Dos cambios sobre la plantilla que genera `alembic init -t async`:

1. **La URL sale de `DATABASE_URL`**, no de `alembic.ini`. El `.ini` versionado no lleva credenciales (su
   `sqlalchemy.url` queda vacío) y la URL llega por entorno, igual que la usa la aplicación: así migraciones
   y servicio apuntan siempre a la misma base, dentro y fuera de compose. Se reutiliza `Settings` para no
   duplicar el valor por defecto ni la lectura del `.env`.

2. **El tipo `vector` se registra en el dialecto** (`ischema_names`) antes de configurar el contexto. Sin
   esto, al reflejar la base Alembic no sabe qué es una columna `vector` y `alembic check` / `--autogenerate`
   producen migraciones inconsistentes (intentan recrear o alterar columnas que ya están bien).
"""

import asyncio
from logging.config import fileConfig

import pgvector.sqlalchemy
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from app.config import get_settings

# `app` es importable porque alembic.ini trae `prepend_sys_path = .` y los comandos se lanzan desde la raíz
# del proyecto (o desde /app en el contenedor).
from app.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# El esquema que describen los modelos. Es lo que compara `alembic check` con la base real.
target_metadata = Base.metadata


def get_database_url() -> str:
    """La URL de `DATABASE_URL` (vía Settings, que también lee `.env`). `-x url=...` la sobrescribe."""
    override = context.get_x_argument(as_dictionary=True).get("url")
    return override or get_settings().database_url


def _register_pgvector_types(connection: Connection) -> None:
    """Enseña al dialecto a reconocer las columnas `vector` al reflejar la base."""
    connection.dialect.ischema_names["vector"] = pgvector.sqlalchemy.Vector


def run_migrations_offline() -> None:
    """Modo offline: renderiza el SQL sin conectarse (`alembic upgrade head --sql`).

    Lo usa `scripts/verificar.sh` para comprobar en CI que la migración es válida y coincide con los modelos
    sin necesidad de un Postgres levantado.
    """
    context.configure(
        url=get_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    _register_pgvector_types(connection)
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Motor async propio de las migraciones (`NullPool`: el proceso se muere al terminar)."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_database_url()
    connectable = async_engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
