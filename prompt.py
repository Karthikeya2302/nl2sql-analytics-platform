from __future__ import annotations

from pathlib import Path
from typing import Sequence

from matching import Relationship, format_relationships_for_prompt
from schema import TableSchema, format_schema_for_prompt

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")


def build_prompt(
    user_question: str,
    schema: dict[str, TableSchema],
    relationships: Sequence[Relationship],
) -> str:
    system_instructions = load_prompt("sql_generation.txt")
    schema_txt = format_schema_for_prompt(schema)
    rel_txt = format_relationships_for_prompt(relationships)

    return f"""{system_instructions}
SCHEMA:
{schema_txt}

POTENTIAL RELATIONSHIPS / JOIN HINTS:
{rel_txt}

USER QUESTION:
{user_question}

SQL:
""".strip()
