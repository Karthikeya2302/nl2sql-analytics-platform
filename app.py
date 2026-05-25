from __future__ import annotations

import os
import time
from typing import Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from db_connector import (
    connect_to_postgres,
    extract_db_schemas,
    extract_fk_relationships,
    get_db_name,
    run_pg_query,
)
from llm import GroqLLM
from matching import infer_relationships
from prompt import build_prompt
from query import make_connection, register_tables, run_query
from schema import extract_schemas, load_table_from_upload


load_dotenv()


st.set_page_config(page_title="NL → SQL Analytics", layout="wide")


@st.cache_resource
def get_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _dedupe_table_name(name: str, existing: set[str]) -> str:
    if name not in existing:
        return name
    i = 2
    while f"{name}_{i}" in existing:
        i += 1
    return f"{name}_{i}"


def ingest_uploads(uploaded_files) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for uf in uploaded_files:
        tname, df = load_table_from_upload(uf)
        tname = _dedupe_table_name(tname, set(tables.keys()))
        tables[tname] = df
    return tables


st.title("Natural Language → SQL Analytics (Local)")
st.caption("Upload CSV/Excel files, ask a question in plain English, and get results via DuckDB.")

with st.sidebar:
    st.header("Settings")
    data_source = st.radio("Data source", ["Upload files", "Connect to database"])
    groq_model = st.text_input("Groq model", value=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    embed_model = st.text_input("Embedding model", value="all-MiniLM-L6-v2")
    show_prompt = st.checkbox("Show prompt (debug)", value=False)

    if data_source == "Connect to database":
        st.markdown("---")
        pg_uri = st.text_input(
            "PostgreSQL URI",
            type="password",
            placeholder="postgresql://user:pass@host:5432/dbname",
        )
        if st.button("Connect"):
            if not pg_uri.strip():
                st.error("Please enter a PostgreSQL URI.")
            else:
                try:
                    with st.spinner("Connecting..."):
                        engine = connect_to_postgres(pg_uri)
                        db_schemas = extract_db_schemas(engine)
                        db_fk_rels = extract_fk_relationships(engine)
                    st.session_state.db_engine = engine
                    st.session_state.db_schemas = db_schemas
                    st.session_state.db_fk_rels = db_fk_rels
                    st.session_state.db_name = get_db_name(pg_uri)
                except Exception as e:
                    st.error(f"Connection failed: {e}")
                    for key in ("db_engine", "db_schemas", "db_fk_rels", "db_name"):
                        st.session_state.pop(key, None)

        if st.session_state.get("db_engine") is not None:
            st.success(f"Connected to {st.session_state.db_name}")

# ── Data loading ──────────────────────────────────────────────────────────────
if data_source == "Upload files":
    uploads = st.file_uploader(
        "Upload one or more files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )
    if not uploads:
        st.info("Upload CSV/Excel files to begin.")
        st.stop()
    try:
        tables = ingest_uploads(uploads)
    except Exception as e:
        st.error(f"Failed to read uploaded files: {e}")
        st.stop()
    schemas = extract_schemas(tables)
    db_engine = None
else:
    if st.session_state.get("db_engine") is None:
        st.info("Connect to a PostgreSQL database in the sidebar to begin.")
        st.stop()
    tables: Dict[str, pd.DataFrame] = {}
    schemas = st.session_state.db_schemas
    db_engine = st.session_state.db_engine

# ── Display tables + schema ───────────────────────────────────────────────────
left, right = st.columns([1, 1])
with left:
    st.subheader("Tables")
    if tables:
        for tname, df in tables.items():
            with st.expander(f"{tname} — {len(df)} rows, {len(df.columns)} cols", expanded=False):
                st.dataframe(df.head(50), use_container_width=True)
    else:
        for tname, ts in schemas.items():
            with st.expander(f"{tname} — {len(ts.columns)} columns", expanded=False):
                st.write([c.name for c in ts.columns])

with right:
    st.subheader("Detected schema")
    schema_rows = []
    for tname, ts in schemas.items():
        for c in ts.columns:
            schema_rows.append(
                {
                    "table": tname,
                    "column": c.name,
                    "dtype": c.pandas_dtype,
                    "samples": ", ".join([s for s in c.sample_values if s][:3]),
                }
            )
    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, height=420)

# ── Relationship detection ────────────────────────────────────────────────────
st.subheader("Join hints (auto-detected)")
relationships = []

using_fk = data_source == "Connect to database" and bool(st.session_state.get("db_fk_rels"))

if using_fk:
    relationships = st.session_state.db_fk_rels
elif len(schemas) >= 2:
    try:
        embedder = None
        _load_err = None
        with st.spinner("Loading embedding model..."):
            for _attempt in range(3):
                try:
                    embedder = get_embedder(embed_model)
                    break
                except Exception as e:
                    _load_err = e
                    if _attempt < 2:
                        time.sleep(2)
        if embedder is None:
            raise _load_err
        relationships = infer_relationships(
            schemas,
            embedder=embedder,
        )
    except Exception as e:
        st.warning(f"Relationship detection unavailable: {e}")

if relationships:
    rel_df = pd.DataFrame(
        [
            {
                "left": f"{r.left_table}.{r.left_column}",
                "right": f"{r.right_table}.{r.right_column}",
                "score": round(r.score, 3),
                "reason": r.reason,
            }
            for r in relationships
        ]
    )
    st.dataframe(rel_df, use_container_width=True, height=220)
    if using_fk:
        st.caption("Relationships sourced from PostgreSQL foreign key constraints.")
else:
    st.caption("No relationships detected (or only one table uploaded).")

# ── Question + run ────────────────────────────────────────────────────────────
st.subheader("Ask a question")
question = st.text_area(
    "Example: What are total sales by region this month?",
    height=90,
    placeholder="Type your question in plain English...",
)

col_a, col_b = st.columns([1, 2])
with col_a:
    run_btn = st.button(
        "Generate SQL + Run",
        type="primary",
        use_container_width=True,
    )
with col_b:
    st.caption("Tip: If you want a chart/aggregation, ask for it explicitly (e.g., group by, top 10, trend by month).")

if run_btn:
    if not question.strip():
        st.error("Please enter a question.")
        st.stop()

    try:
        prompt = build_prompt(question, schemas, relationships)
        llm = GroqLLM.from_env(model=groq_model)

        with st.spinner("Generating SQL with Groq..."):
            sql = llm.chat_completion(prompt, temperature=0.0)

        st.subheader("Generated SQL")
        st.code(sql, language="sql")

        if show_prompt:
            with st.expander("Prompt (debug)", expanded=False):
                st.text(prompt)

        if db_engine is not None:
            with st.spinner("Executing in PostgreSQL..."):
                result_df = run_pg_query(db_engine, sql)
        else:
            with st.spinner("Executing in DuckDB..."):
                conn = make_connection()
                register_tables(conn, tables)
                result_df = run_query(conn, sql)

        st.subheader("Results")
        st.dataframe(result_df, use_container_width=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="results.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(str(e))
