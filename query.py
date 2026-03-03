from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import duckdb
import pandas as pd


_DISALLOWED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|PRAGMA|ATTACH|DETACH|COPY|CALL|EXPORT|IMPORT)\b",
    re.IGNORECASE,
)


def make_connection() -> duckdb.DuckDBPyConnection:
    # In-memory, local-only.
    return duckdb.connect(database=":memory:")


def register_tables(conn: duckdb.DuckDBPyConnection, tables: Dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        conn.register(name, df)


def sanitize_sql(sql: str) -> str:
    s = (sql or "").strip()
    # If the model accidentally returns fenced code or extra text, try to extract SQL-ish portion.
    s = s.strip("`").strip()
    s = re.sub(r"^sql\s*", "", s, flags=re.IGNORECASE).strip()
    return s


def validate_sql(sql: str) -> None:
    s = sanitize_sql(sql)
    if not s:
        raise ValueError("Empty SQL.")
    if ";" in s:
        # Avoid multiple statements.
        raise ValueError("SQL contains ';' which may indicate multiple statements.")
    if _DISALLOWED.search(s):
        raise ValueError("Disallowed SQL keyword detected. Only SELECT-style queries are permitted.")
    if not re.match(r"^\s*select\b", s, flags=re.IGNORECASE):
        raise ValueError("Only SELECT queries are permitted.")


def run_query(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
) -> pd.DataFrame:
    validate_sql(sql)
    return conn.execute(sanitize_sql(sql)).df()

