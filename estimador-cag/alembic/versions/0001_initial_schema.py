"""Esquema inicial: extensión vector + documents + chunks (sin índice vectorial).

Revision ID: 0001_initial_schema
Revises: None
Create Date: 2026-10-23

Primera migración del proyecto (pre-work sesión 08). Crea:

- la extensión `vector` (la imagen pgvector/pgvector:pg16 la trae precompilada);
- `documents`: un documento ingestado, con su ruta de origen, su tipo y metadata JSONB;
- `chunks`: sus fragmentos vectorizados, con `ON DELETE CASCADE` sobre `documents.id`;
- los índices **no vectoriales**: la ruta de origen, la foreign key, el tipo de chunk y un GIN sobre el JSONB.

**No crea ningún índice sobre `chunks.embedding`**: a propósito. Sin índice, Postgres resuelve la búsqueda
semántica con un sequential scan sobre toda la tabla, y esa es la medida de partida contra la que la sesión
en directo comparará un HNSW. Añadirlo aquí sería perder el baseline (ver README).
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None

# Dimensión de text-embedding-3-small. Duplicada respecto a app.db.models.EMBEDDING_DIMENSIONS a propósito:
# una migración es un registro histórico y no debe cambiar de resultado porque el código evolucione. El test
# `test_migration_dimension_matches_models` comprueba que ambas siguen de acuerdo.
EMBEDDING_DIMENSIONS = 1536


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("source_path", sa.Text, nullable=False),
        sa.Column("document_type", sa.String(50), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("metadata", postgresql.JSONB, server_default="{}", nullable=False),
    )
    op.create_index("ix_documents_source_path", "documents", ["source_path"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("document_id", sa.BigInteger, sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_type", sa.String(50), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=True),
        sa.Column("metadata", postgresql.JSONB, server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_chunk_type", "chunks", ["chunk_type"])
    op.create_index("ix_chunks_metadata_gin", "chunks", ["metadata"], postgresql_using="gin")


def downgrade() -> None:
    # Las tablas se borran en orden inverso (chunks depende de documents). La extensión no se elimina: puede
    # haber otros esquemas usándola y `DROP EXTENSION` es más destructivo que lo que esta migración creó.
    op.drop_index("ix_chunks_metadata_gin", table_name="chunks")
    op.drop_index("ix_chunks_chunk_type", table_name="chunks")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("ix_documents_source_path", table_name="documents")
    op.drop_table("documents")
