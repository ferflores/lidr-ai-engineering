"""Capa de persistencia: PostgreSQL + pgvector (sesión 08).

`models.py` define el esquema (dos tablas, `documents` y `chunks`), `session.py` el motor async y la
dependencia de FastAPI que abre una sesión por petición. El esquema real lo crean las migraciones de
Alembic (`alembic/versions/`), no `create_all`: los modelos y las migraciones se mantienen alineados y
`scripts/verificar.sh` lo comprueba renderizando la migración en modo offline.
"""

from app.db.models import EMBEDDING_DIMENSIONS, Base, Chunk, Document
from app.db.session import get_db_session, get_engine, get_sessionmaker

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "Base",
    "Chunk",
    "Document",
    "get_db_session",
    "get_engine",
    "get_sessionmaker",
]
