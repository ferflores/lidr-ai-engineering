"""Full-text search column on chunks: generated tsvector + GIN index (session 10 pre-work).

Revision ID: 0002_chunks_fulltext
Revises: 0001_initial_schema
Create Date: 2026-11-09

Adds the lexical branch of hybrid search to `chunks`:

- `content_tsv`: a **generated, stored** `tsvector` computed from `content`. Stored (not virtual) because
  Postgres can only index stored generated columns, and the whole point of the column is the GIN index.
  Being generated, it can never drift from `content`: an ingest writes `content` and the database derives the
  lexemes; nothing in the application code has to remember to keep the two in sync.
- `ix_chunks_content_tsv_gin`: GIN index over the tsvector, the index type Postgres recommends for `@@`
  full-text queries.

Text search configuration: **`english`**. The exercise statement says the dataset is in Spanish, but that is
true of the official course dataset, not of this project's corpus: every budget in `data/budgets_sample.json`
(project summaries, component names and descriptions) is written in English, and the chunk text the chunker
renders around it is English too. The configuration has to match the language of the *content* being
indexed, because it drives stemming and stop words: with `spanish`, "payments" would not stem to the same
lexeme as "payment", and English stop words ("the", "with", "for") would be indexed as if they carried
meaning. Queries are normalised with the same configuration at search time (see
`app/embedding_pipeline/retrieval/lexical.py`). If the corpus ever mixes languages, the right move is a
per-row `regconfig` column rather than a global guess.

The configuration string is duplicated in `app.db.models.TEXT_SEARCH_CONFIG` on purpose: a migration is a
historical record and must not change its result when the code evolves. `tests/test_retrieval.py` checks the
two stay in agreement.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_chunks_fulltext"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None

TEXT_SEARCH_CONFIG = "english"
TSVECTOR_EXPRESSION = f"to_tsvector('{TEXT_SEARCH_CONFIG}'::regconfig, content)"
INDEX_NAME = "ix_chunks_content_tsv_gin"


def upgrade() -> None:
    # The two-argument form of to_tsvector (explicit regconfig) is IMMUTABLE, which is what a generated column
    # requires; the one-argument form depends on the session's default_text_search_config and would be rejected.
    op.add_column(
        "chunks",
        sa.Column(
            "content_tsv",
            postgresql.TSVECTOR,
            sa.Computed(TSVECTOR_EXPRESSION, persisted=True),
            nullable=True,
        ),
    )
    op.create_index(INDEX_NAME, "chunks", ["content_tsv"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_index(INDEX_NAME, table_name="chunks")
    op.drop_column("chunks", "content_tsv")
