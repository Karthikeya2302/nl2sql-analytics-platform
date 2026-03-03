from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from schema import TableSchema


@dataclass(frozen=True)
class Relationship:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    score: float
    reason: str


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def _is_id_like(col: str) -> bool:
    c = col.lower().strip()
    return c == "id" or c.endswith("_id") or c.endswith("id")


def build_column_corpus(schema: Dict[str, TableSchema]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Returns:
      - texts: list of natural language strings describing each column
      - keys: (table, column) pairs aligned with texts
    """
    texts: List[str] = []
    keys: List[Tuple[str, str]] = []
    for tname, ts in schema.items():
        for c in ts.columns:
            sample_hint = ""
            if c.sample_values:
                sample_hint = " examples: " + ", ".join([s for s in c.sample_values if s][:3])
            texts.append(f"table {tname}, column {c.name}, type {c.pandas_dtype}.{sample_hint}")
            keys.append((tname, c.name))
    return texts, keys


def infer_relationships(
    schema: Dict[str, TableSchema],
    embedder,
    max_per_column: int = 2,
    min_score: float = 0.72,
) -> List[Relationship]:
    """
    Lightweight relationship inference:
    - Uses embeddings + cosine similarity to propose join keys across tables.
    - Adds a small heuristic boost for id-like columns and exact-ish name patterns.
    """
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("FAISS is required for relationship inference. Install `faiss-cpu`.") from e

    texts, keys = build_column_corpus(schema)
    if not texts:
        return []

    emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    emb = emb.astype("float32")
    emb = _l2_normalize(emb)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    rels: List[Relationship] = []
    seen = set()

    # Search for nearest neighbors for each column; filter to other tables.
    k = min(len(keys), 12)
    scores, idxs = index.search(emb, k)

    for i, (t1, c1) in enumerate(keys):
        for jpos in range(1, k):
            j = int(idxs[i, jpos])
            if j < 0 or j >= len(keys):
                continue
            t2, c2 = keys[j]
            if t1 == t2:
                continue

            s = float(scores[i, jpos])
            reason_parts = ["semantic similarity"]

            # Heuristic boosts / reasons
            if _is_id_like(c1) and _is_id_like(c2):
                s = min(0.999, s + 0.05)
                reason_parts.append("id-like columns")
            if c1.lower() == c2.lower():
                s = min(0.999, s + 0.07)
                reason_parts.append("same column name")
            if c1.lower() == f"{t2.lower()}_id" or c2.lower() == f"{t1.lower()}_id":
                s = min(0.999, s + 0.07)
                reason_parts.append("foreign-key naming pattern")

            if s < min_score:
                continue

            # De-dup (unordered pair)
            a = (t1, c1, t2, c2)
            b = (t2, c2, t1, c1)
            if a in seen or b in seen:
                continue
            seen.add(a)

            rels.append(
                Relationship(
                    left_table=t1,
                    left_column=c1,
                    right_table=t2,
                    right_column=c2,
                    score=s,
                    reason=", ".join(reason_parts),
                )
            )

    # Sort by score; limit per column to keep prompt small.
    rels.sort(key=lambda r: r.score, reverse=True)

    per_col = {}
    out: List[Relationship] = []
    for r in rels:
        key = (r.left_table, r.left_column)
        if per_col.get(key, 0) >= max_per_column:
            continue
        per_col[key] = per_col.get(key, 0) + 1
        out.append(r)
    return out


def format_relationships_for_prompt(relationships: Sequence[Relationship]) -> str:
    if not relationships:
        return "None detected."
    lines: List[str] = []
    for r in relationships:
        lines.append(
            f"- {r.left_table}.{r.left_column} ↔ {r.right_table}.{r.right_column} (score {r.score:.2f}; {r.reason})"
        )
    return "\n".join(lines)

