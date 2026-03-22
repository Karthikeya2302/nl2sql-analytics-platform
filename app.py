from __future__ import annotations
 
import os
import datetime
from typing import Dict
 
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
 
from llm import GroqLLM
from matching import infer_relationships
from prompt import build_prompt
from query import make_connection, register_tables, run_query
from schema import extract_schemas, load_table_from_upload
 
 
load_dotenv()
 
 
# ── Query Limiting ────────────────────────────────────────────────────────────
DAILY_QUERY_LIMIT = 10
 
def get_query_count() -> int:
    if "query_date" not in st.session_state:
        st.session_state.query_date = str(datetime.date.today())
        st.session_state.query_count = 0
    if st.session_state.query_date != str(datetime.date.today()):
        st.session_state.query_date = str(datetime.date.today())
        st.session_state.query_count = 0
    return st.session_state.query_count
 
def increment_query_count() -> None:
    st.session_state.query_count = get_query_count() + 1
 
def queries_remaining() -> int:
    return max(0, DAILY_QUERY_LIMIT - get_query_count())
 
 
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
    groq_model = st.text_input("Groq model", value=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    embed_model = st.text_input("Embedding model", value="all-MiniLM-L6-v2")
    min_score = st.slider("Relationship min score", min_value=0.50, max_value=0.95, value=0.72, step=0.01)
    max_per_column = st.slider("Max relationships per column", min_value=0, max_value=5, value=2, step=1)
    show_prompt = st.checkbox("Show prompt (debug)", value=False)
 
    st.markdown("---")
    remaining = queries_remaining()
    st.metric("Queries remaining today", f"{remaining} / {DAILY_QUERY_LIMIT}")
    if remaining == 0:
        st.error("Daily limit reached. Come back tomorrow!")
    elif remaining <= 3:
        st.warning(f"Only {remaining} queries left!")
    else:
        st.success("Queries available ✅")
 
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
 
left, right = st.columns([1, 1])
with left:
    st.subheader("Tables")
    for tname, df in tables.items():
        with st.expander(f"{tname} — {len(df)} rows, {len(df.columns)} cols", expanded=False):
            st.dataframe(df.head(50), use_container_width=True)
 
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
 
st.subheader("Join hints (auto-detected)")
relationships = []
if len(tables) >= 2 and max_per_column > 0:
    try:
        embedder = get_embedder(embed_model)
        relationships = infer_relationships(
            schemas,
            embedder=embedder,
            max_per_column=int(max_per_column),
            min_score=float(min_score),
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
else:
    st.caption("No relationships detected (or only one table uploaded).")
 
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
        disabled=(queries_remaining() == 0)
    )
with col_b:
    st.caption("Tip: If you want a chart/aggregation, ask for it explicitly (e.g., group by, top 10, trend by month).")
 
if run_btn:
    if not question.strip():
        st.error("Please enter a question.")
        st.stop()
 
    if queries_remaining() == 0:
        st.error("Daily limit reached. Come back tomorrow!")
        st.stop()
 
    try:
        prompt = build_prompt(question, schemas, relationships)
        llm = GroqLLM.from_env(model=groq_model)
 
        with st.spinner("Generating SQL with Groq..."):
            sql = llm.chat_completion(prompt, temperature=0.0)
 
        increment_query_count()
 
        st.subheader("Generated SQL")
        st.code(sql, language="sql")
 
        if show_prompt:
            with st.expander("Prompt (debug)", expanded=False):
                st.text(prompt)
 
        with st.spinner("Executing in DuckDB..."):
            conn = make_connection()
            register_tables(conn, tables)
            result_df = run_query(conn, sql)
 
        st.subheader("Results")
        st.dataframe(result_df, use_container_width=True)
        st.caption(f"{len(result_df)} rows returned • {queries_remaining()} queries remaining today")
 
        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="results.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(str(e))
