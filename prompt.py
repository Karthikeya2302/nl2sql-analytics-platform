from __future__ import annotations

from typing import Sequence

from matching import Relationship, format_relationships_for_prompt
from schema import TableSchema, format_schema_for_prompt


SYSTEM_INSTRUCTIONS = """You are an expert data analyst who writes correct SQL for DuckDB.

Rules:
- Output ONLY a single SQL query (no markdown, no explanation, no code fences).
- Use only the tables/columns provided in the schema.
- Prefer SELECT queries. Do NOT write INSERT/UPDATE/DELETE/CREATE/DROP/ALTER/PRAGMA/ATTACH/COPY.
- If the question is ambiguous, make a reasonable assumption and proceed.
- Always qualify columns with table names when multiple tables are involved.
- Use DuckDB-compatible functions.
"""


def build_prompt(
    user_question: str,
    schema: dict[str, TableSchema],
    relationships: Sequence[Relationship],
) -> str:
    schema_txt = format_schema_for_prompt(schema)
    rel_txt = format_relationships_for_prompt(relationships)

    return f"""{SYSTEM_INSTRUCTIONS}

SCHEMA:
{schema_txt}

POTENTIAL RELATIONSHIPS / JOIN HINTS:
{rel_txt}

USER QUESTION:
{user_question}

SQL:
""".strip()

