from __future__ import annotations

from typing import Dict, List
from urllib.parse import urlparse

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from matching import Relationship
from query import sanitize_sql, validate_sql
from schema import ColumnInfo, TableSchema


_PG_TYPE_MAP: Dict[str, str] = {
    "integer": "int64",
    "bigint": "int64",
    "smallint": "int64",
    "serial": "int64",
    "bigserial": "int64",
    "real": "float64",
    "double precision": "float64",
    "numeric": "float64",
    "decimal": "float64",
    "boolean": "bool",
    "date": "datetime64[ns]",
    "text": "object",
    "character varying": "object",
    "varchar": "object",
    "char": "object",
    "uuid": "object",
    "json": "object",
    "jsonb": "object",
}


def _pg_type_to_pandas(pg_type: str) -> str:
    t = pg_type.lower()
    if t in _PG_TYPE_MAP:
        return _PG_TYPE_MAP[t]
    if t.startswith("timestamp"):
        return "datetime64[ns]"
    if t.startswith("character"):
        return "object"
    return "object"


def connect_to_postgres(uri: str) -> Engine:
    engine = create_engine(uri)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine


def get_db_name(uri: str) -> str:
    try:
        return urlparse(uri).path.lstrip("/") or uri
    except Exception:
        return uri


def extract_db_schemas(engine: Engine) -> Dict[str, TableSchema]:
    with engine.connect() as conn:
        table_rows = conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
            "ORDER BY table_name"
        )).fetchall()
        table_names = [r[0] for r in table_rows]

        col_rows = conn.execute(text(
            "SELECT table_name, column_name, data_type "
            "FROM information_schema.columns "
            "WHERE table_schema = 'public' "
            "ORDER BY table_name, ordinal_position"
        )).fetchall()

    cols_by_table: Dict[str, List[tuple]] = {}
    for row in col_rows:
        cols_by_table.setdefault(row[0], []).append((row[1], row[2]))

    return {
        tname: TableSchema(
            table_name=tname,
            columns=[
                ColumnInfo(
                    name=cname,
                    pandas_dtype=_pg_type_to_pandas(dtype),
                    sample_values=[],
                )
                for cname, dtype in cols_by_table.get(tname, [])
            ],
            row_count=0,
        )
        for tname in table_names
    }


def extract_fk_relationships(engine: Engine) -> List[Relationship]:
    query = text("""
        SELECT
            kcu.table_name,
            kcu.column_name,
            ccu.table_name,
            ccu.column_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
           AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
           AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = 'public'
        ORDER BY kcu.table_name, kcu.column_name
    """)
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()
    return [
        Relationship(
            left_table=r[0],
            left_column=r[1],
            right_table=r[2],
            right_column=r[3],
            score=1.0,
            reason="foreign key constraint",
        )
        for r in rows
    ]


def run_pg_query(engine: Engine, sql: str) -> pd.DataFrame:
    validate_sql(sql)
    clean = sanitize_sql(sql)
    with engine.connect() as conn:
        return pd.read_sql(text(clean), conn)
