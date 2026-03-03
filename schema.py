from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    pandas_dtype: str
    sample_values: List[str]


@dataclass(frozen=True)
class TableSchema:
    table_name: str
    columns: List[ColumnInfo]
    row_count: int


def _safe_str(x) -> str:
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x)
    s = s.replace("\n", " ").replace("\r", " ").strip()
    return s[:80]


def normalize_table_name(filename: str) -> str:
    """
    Convert filename into a DuckDB-friendly table identifier.
    """
    base = filename.rsplit(".", 1)[0]
    out = []
    for ch in base:
        if ch.isalnum() or ch == "_":
            out.append(ch.lower())
        elif ch in (" ", "-", "."):
            out.append("_")
    name = "".join(out).strip("_")
    if not name:
        name = "table"
    if name[0].isdigit():
        name = f"t_{name}"
    return name


def extract_table_schema(df: pd.DataFrame, table_name: str, sample_size: int = 3) -> TableSchema:
    cols: List[ColumnInfo] = []
    for col in df.columns:
        series = df[col]
        sample_vals = []
        try:
            non_null = series.dropna().head(sample_size)
            sample_vals = [_safe_str(v) for v in non_null.tolist()]
        except Exception:
            sample_vals = []
        cols.append(
            ColumnInfo(
                name=str(col),
                pandas_dtype=str(series.dtype),
                sample_values=sample_vals,
            )
        )
    return TableSchema(table_name=table_name, columns=cols, row_count=int(len(df)))


def extract_schemas(tables: Dict[str, pd.DataFrame]) -> Dict[str, TableSchema]:
    return {name: extract_table_schema(df, name) for name, df in tables.items()}


def format_schema_for_prompt(schema: Dict[str, TableSchema]) -> str:
    lines: List[str] = []
    for table_name, ts in schema.items():
        lines.append(f"- {table_name} ({ts.row_count} rows)")
        for c in ts.columns:
            samples = ""
            if c.sample_values:
                samples = f" | samples: {', '.join([s for s in c.sample_values if s])}"
            lines.append(f"  - {c.name} ({c.pandas_dtype}){samples}")
    return "\n".join(lines)


def load_table_from_upload(uploaded_file) -> Tuple[str, pd.DataFrame]:
    """
    Streamlit UploadedFile -> (table_name, df)
    Supports CSV and Excel (first sheet by default).
    """
    fname = getattr(uploaded_file, "name", "table.csv")
    table_name = normalize_table_name(fname)

    if fname.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif fname.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        raise ValueError("Unsupported file type. Please upload CSV or Excel.")

    # Normalize column names to strings; keep original semantics (no aggressive renames).
    df.columns = [str(c).strip() for c in df.columns]
    return table_name, df

