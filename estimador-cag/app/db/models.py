"""Esquema relacional de los presupuestos vectorizados: `documents` (1) → `chunks` (N).

Decisiones (justificadas en el README, sección "Persistencia en PostgreSQL + pgvector"):

- **Dos tablas.** Un presupuesto produce N chunks. Una sola tabla desnormalizada repetiría la metadata del
  presupuesto en cada fila y perdería la integridad referencial; con `ON DELETE CASCADE`, borrar un
  presupuesto borra sus chunks sin código que lo orqueste.
- **`metadata` JSONB en las dos tablas.** Lo estable y siempre presente va en columnas tipadas
  (`document_type`, `chunk_type`, fechas); lo variable y lo que el chunker puede enriquecer, en JSONB con
  índice GIN.
- **`Vector(1536)`.** Dimensión de `text-embedding-3-small`, fijada a propósito: cambiarla obliga a
  re-embeber el corpus entero, no es un parámetro de ejecución.
- **`embedding` nullable.** Deja la puerta abierta a insertar el chunk y rellenar el vector después
  (ingesta asíncrona). En este ejercicio se insertan juntos, en la misma transacción.
- **Sin índice vectorial.** Deliberado: el sequential scan es el baseline del directo.

Nota sobre el nombre del atributo: en SQLAlchemy `metadata` está reservado en las clases declarativas
(`Base.metadata`), así que el atributo Python se llama `doc_metadata` / `chunk_metadata` y se mapea
explícitamente a la columna `metadata`, que es el nombre que fija el enunciado.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Dimensión de text-embedding-3-small. Está aquí y en la migración 0001 (una migración es un registro
# histórico y no se reescribe); `tests/test_db_models.py` comprueba que no se desincronizan.
EMBEDDING_DIMENSIONS = 1536

# Vocabulario de `chunks.chunk_type`. Hoy solo hay un tipo: un componente de presupuesto. La columna existe
# porque el chunker de otros documentos (transcripciones, especificaciones) producirá otros tipos, y porque
# el directo filtra por ella (`WHERE chunk_type = 'budget_component'`).
CHUNK_TYPE_BUDGET_COMPONENT = "budget_component"

# Vocabulario de `documents.document_type`, el que usa el enunciado para los presupuestos históricos.
DOCUMENT_TYPE_HISTORICAL_BUDGET = "historical_budget"


class Base(DeclarativeBase):
    pass


class Document(Base):
    """Un documento ingestado. En este ejercicio, un presupuesto histórico."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    document_type: Mapped[str] = mapped_column(String(50), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    doc_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, server_default="{}", nullable=False)

    # `passive_deletes=True`: el borrado en cascada lo hace Postgres con la foreign key, no el ORM cargando
    # los hijos en memoria para borrarlos uno a uno.
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (Index("ix_documents_source_path", "source_path"),)


class Chunk(Base):
    """Un fragmento vectorizado de un documento. Para un presupuesto, un componente."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_type: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # nullable: el chunk puede existir antes que su vector (ingesta asíncrona, sesiones posteriores).
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=True)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, server_default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_chunk_type", "chunk_type"),
        # GIN sobre el JSONB: permite consultar por claves arbitrarias (`metadata @> '{"client_sector": ...}'`)
        # sin migrar el esquema cada vez que el chunker añade un campo.
        Index("ix_chunks_metadata_gin", "metadata", postgresql_using="gin"),
    )
