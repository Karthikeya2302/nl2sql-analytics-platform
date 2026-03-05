# DataTalk — NL2SQL Analytics Platform

> Ask questions about your data in plain English. No SQL required.

Built for non-technical users like HR managers, finance teams, and business analysts who sit on valuable data but can't write SQL to get insights from it.

---

## What it does

Upload your CSV or Excel files, type a question like _"which department has the highest attrition this year?"_ and get back a result instantly.

Under the hood:
- Extracts schema from your uploaded files automatically
- Uses **sentence transformer embeddings + FAISS** to detect relationships between multiple tables silently
- Builds a context-aware prompt with schema, relationships, and sample values
- Sends it to **Groq LLaMA 3** which generates the SQL
- Executes the SQL on your data using **DuckDB** in memory
- Returns accurate results — not approximations

---

## Why not just use ChatGPT with my Excel?

General AI tools like ChatGPT read your data as text and approximate answers. They struggle with large datasets and can miss rows or miscalculate.

This tool actually **runs real SQL queries** on your data via DuckDB — so results are always accurate and it scales to large files.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq API (LLaMA 3.3 70B) |
| Embeddings | Sentence Transformers |
| Similarity Search | FAISS |
| Query Execution | DuckDB |
| Data Processing | Pandas |
| Environment | python-dotenv |

---

## Project Structure

```
nl2sql/
│
├── app.py          # Streamlit UI and main flow
├── schema.py       # Schema extraction from uploaded files
├── matching.py     # Embedding-based relationship detection
├── prompt.py       # Dynamic prompt builder
├── query.py        # SQL validation and DuckDB execution
├── llm.py          # Groq API wrapper
├── .env            # API keys (never commit this)
├── .gitignore      
└── requirements.txt
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/nl2sql.git
cd nl2sql
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

Create a `.env` file in the `nl2sql/` folder:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)

---

## Run

```bash
streamlit run app.py
```

---

## How it works

###Single table
Upload a CSV → ask a question → get results. That simple.

### Multiple tables
Upload multiple CSVs or Excel sheets → the app automatically:
1. Extracts schema from each table
2. Converts column descriptions into embeddings
3. Uses FAISS similarity search to detect which columns relate across tables (e.g. `customers.id` ↔ `transactions.customer_id`)
4. Passes detected relationships as JOIN hints to the LLM
5. LLM generates correct SQL with proper JOINs

No manual data modeling needed.

---

## Example Questions

**Basic**
- "Show me all records"
- "How many rows are there"
- "Show me top 5 rows by amount"

**Aggregations**
- "What is total sales by region"
- "Which city has the most customers"
- "Show me average order value by month"

**Multi-table**
- "Show me customer names with their total spending"
- "Which customer placed the most orders"
- "Top 5 customers by revenue"

---

## Safety

Only `SELECT` queries are allowed. The app blocks all write and DDL operations including `INSERT`, `UPDATE`, `DELETE`, `CREATE`, `DROP`, and `ALTER` before execution.

---

## Limitations

- Each table should be in its own sheet (for Excel files with multiple sheets, each sheet is treated as a separate table)
- Designed for structured tabular data only
- Relationship detection works best when column names follow consistent naming conventions

---

## Future Improvements

- [ ] Auto visualization of query results as charts
- [ ] Conversational follow-up questions
- [ ] Value overlap based relationship detection
- [ ] Support for larger files via chunked loading
- [ ] Feedback loop to improve SQL accuracy over time

---

## Author

Built by Karthikeya Thimirishetty — MS CS at UAB



