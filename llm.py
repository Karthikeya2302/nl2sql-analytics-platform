from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


@dataclass(frozen=True)
class GroqConfig:
    api_key: str
    model: str = "llama-3.3-70b-versatile"
    base_url: str = "https://api.groq.com/openai/v1"
    timeout_s: int = 60


class GroqLLM:
    def __init__(self, cfg: GroqConfig):
        if not cfg.api_key:
            raise ValueError("Missing Groq API key. Set GROQ_API_KEY in your environment.")
        self.cfg = cfg

    @staticmethod
    def from_env(model: Optional[str] = None) -> "GroqLLM":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        m = (model or os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()
        return GroqLLM(GroqConfig(api_key=api_key, model=m))

    def chat_completion(self, prompt: str, temperature: float = 0.0) -> str:
        url = f"{self.cfg.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": "Return only the SQL query as plain text."},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(temperature),
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_s)
        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Groq API error: {resp.status_code} {resp.text}") from e

        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return (content or "").strip()

    def explain_query(self, sql: str, schema: dict, question: str) -> str:
        template = (Path(__file__).parent / "prompts" / "query_explanation.txt").read_text(encoding="utf-8")
        schema_summary = "\n".join(
            f"- {tname}: {', '.join(c.name for c in ts.columns)}"
            for tname, ts in schema.items()
        )
        prompt = f"{template}\n\nUser question: {question}\n\nSchema:\n{schema_summary}\n\nSQL:\n{sql}"
        url = f"{self.cfg.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": "You explain SQL queries in plain English for non-technical users."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_s)
        resp.raise_for_status()
        content = (
            resp.json().get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return (content or "").strip()
