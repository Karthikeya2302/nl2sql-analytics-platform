from __future__ import annotations

import os
import threading

from dotenv import load_dotenv

load_dotenv()

from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
from fastapi import FastAPI, Request
import uvicorn

from db_connector import connect_to_postgres, extract_db_schemas, extract_fk_relationships, run_pg_query
from llm import GroqLLM
from prompt import build_prompt


slack_app = App(
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
)
handler = SlackRequestHandler(slack_app)
api = FastAPI()


@api.post("/slack/events")
async def slack_events(req: Request):
    return await handler.handle(req)


def _df_to_text_table(df, max_rows: int = 10) -> str:
    if df.empty:
        return "(no results)"
    subset = df.head(max_rows)
    cols = list(subset.columns)
    rows = [cols] + [[str(v) for v in row] for row in subset.itertuples(index=False)]
    widths = [max(len(r[i]) for r in rows) for i in range(len(cols))]
    lines = []
    for i, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            lines.append("  ".join("-" * widths[j] for j in range(len(cols))))
    return "\n".join(lines)


def _truncate(text: str, limit: int = 2900) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


@slack_app.command("/query")
def handle_query(ack, respond, command):
    question = (command.get("text") or "").strip()

    if not question:
        ack(text="Please provide a question after `/query`. Example: `/query total sales by region`")
        return

    ack(text=":hourglass_flowing_sand: Processing your query, hang tight...")

    def process():
        try:
            db_uri = os.environ["DATATALK_DB_URI"]
            engine = connect_to_postgres(db_uri)
            schemas = extract_db_schemas(engine)
            relationships = extract_fk_relationships(engine)

            llm = GroqLLM.from_env()
            prompt = build_prompt(question, schemas, relationships)
            sql = llm.chat_completion(prompt, temperature=0.0)

            result_df = run_pg_query(engine, sql)

            # Limit to 5 columns and 10 rows for display
            display_df = result_df.iloc[:, :5]
            table_text = _df_to_text_table(display_df)
            row_count = len(result_df)

            blocks = [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*{question}*"},
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"```{_truncate(table_text)}```"},
                },
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": f"Found {row_count} result(s)"}],
                },
            ]

            respond(blocks=blocks, text=f"Found {row_count} result(s)", replace_original=True)

        except Exception as e:
            respond(text=f":warning: Something went wrong: {e}", replace_original=True)

    threading.Thread(target=process, daemon=True).start()


if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=3000)
